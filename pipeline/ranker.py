import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from eval.split import make_val_split

from .artifacts import pipeline_signature
from .features import FEATURES, LEGACY_FEATURES
from .pool import PoolEngine


def predict_scores(model, frame, *, legacy=False):
    columns = LEGACY_FEATURES if legacy else FEATURES
    values = frame[columns] if legacy else frame[columns].fillna(-1)
    return model.predict(values)


def rank_items(model, frame, *, legacy=False):
    frame = frame.copy()
    frame["score"] = predict_scores(model, frame, legacy=legacy)
    return frame.sort_values(["score", "item_id"], ascending=[False, True]).item_id.tolist()


def load_ranker(path):
    from catboost import CatBoostRanker
    model = CatBoostRanker()
    model.load_model(str(path))
    return model


@dataclass
class RankerTrainer:
    root: object
    train: pd.DataFrame
    history: pd.DataFrame
    items: pd.DataFrame
    base: object
    weighted: object
    engine: PoolEngine
    pool_variant: str

    def train_ranker(self, fit_queries=2000, production=False):
        from catboost import CatBoostRanker, Pool
        fit_history, queries, relevant = make_val_split(self.train if production else self.history, val_size=.05, random_state=7)
        queries = queries.sample(min(fit_queries, len(queries)), random_state=11).reset_index(drop=True)
        folder = self.root / "data/pool_eval";folder.mkdir(exist_ok=True)
        prefix = "production" if production else "validation"
        key = pipeline_signature(f"{prefix}:{len(queries)}:{self.pool_variant}".encode())
        marker = folder / f"{prefix}_training_manifest.json"
        new_path, old_path = folder / f"{prefix}_training.parquet", folder / f"{prefix}_legacy_training.parquet"
        if marker.exists() and json.loads(marker.read_text()).get("signature") == key:
            training, legacy = pd.read_parquet(new_path), pd.read_parquet(old_path)
            print(f"{prefix}: обучающие признаки восстановлены из cache", flush=True)
        else:
            engine = PoolEngine(self.root, self.items, fit_history, self.base, self.weighted,
                char_matrix=self.engine.char_matrix, fields=self.engine.fields, dense=self.engine.dense)
            engine.char_right = self.engine.char_right
            chunk_folder = folder / f"{prefix}_chunks" / key
            chunk_folder.mkdir(parents=True, exist_ok=True)
            markers = sorted(chunk_folder.glob("*.json"))
            completed = set().union(*(set(json.loads(path.read_text())) for path in markers)) if markers else set()
            todo = queries.loc[~queries.context_id.isin(completed)].reset_index(drop=True)
            print(f"{prefix}: готово {len(completed)}, осталось {len(todo)} контекстов", flush=True)
            frames, legacy_frames, chunk_ids = [], [], []
            chunk_number = len(markers)
            def flush_chunk():
                nonlocal chunk_number
                name = f"{chunk_number:04d}"
                if frames:
                    pd.concat(frames, ignore_index=True).to_parquet(chunk_folder / f"{name}_new.parquet", index=False)
                if legacy_frames:
                    pd.concat(legacy_frames, ignore_index=True).to_parquet(chunk_folder / f"{name}_old.parquet", index=False)
                (chunk_folder / f"{name}.json").write_text(json.dumps(chunk_ids))
                frames.clear();legacy_frames.clear();chunk_ids.clear()
                chunk_number += 1
            with threadpool_limits(limits=4):
                for candidate in engine.iter_candidates(todo):
                    pool = candidate.pools[self.pool_variant]
                    frame = candidate.features[candidate.features.item_id.isin(set(pool))].copy()
                    frame["label"] = frame.item_id.isin(relevant[candidate.cid]).astype(np.int8)
                    if frame.label.any() and not frame.label.all():
                        frames.append(frame)
                    old_frame = candidate.features.loc[candidate.features.item_id.isin(candidate.pools["old_6_channels_union"]),
                        ["context_id", "item_id"] + LEGACY_FEATURES].copy()
                    old_frame["label"] = old_frame.item_id.isin(relevant[candidate.cid]).astype(np.int8)
                    if old_frame.label.any() and not old_frame.label.all():
                        legacy_frames.append(old_frame)
                    chunk_ids.append(candidate.cid)
                    if len(chunk_ids) == 500:
                        flush_chunk()
                if chunk_ids:
                    flush_chunk()
            frames = [pd.read_parquet(path) for path in sorted(chunk_folder.glob("*_new.parquet"))]
            legacy_frames = [pd.read_parquet(path) for path in sorted(chunk_folder.glob("*_old.parquet"))]
            if sum(frame.context_id.nunique() for frame in frames) < 10:
                raise ValueError("Слишком мало обучающих групп с позитивами")
            training = pd.concat(frames, ignore_index=True).sort_values(["context_id", "item_id"])
            frames.clear()
            legacy = pd.concat(legacy_frames, ignore_index=True).sort_values(["context_id", "item_id"])
            legacy_frames.clear()
            training.to_parquet(new_path, index=False)
            legacy.to_parquet(old_path, index=False)
            marker.write_text(json.dumps({"signature": key}))
        groups = training.context_id.unique()
        all_groups = queries.context_id.to_numpy().copy();rng = np.random.RandomState(13);rng.shuffle(all_groups)
        valid_groups = set(all_groups[:max(1, len(all_groups) // 5)])
        mask = training.context_id.isin(valid_groups)
        def dataset(df):
            return Pool(df[FEATURES].fillna(-1), df.label, group_id=df.context_id)
        model = CatBoostRanker(loss_function="PairLogit", iterations=250, depth=6,
            learning_rate=.08, random_seed=42, thread_count=4, allow_writing_files=False)
        model.fit(dataset(training.loc[~mask]), eval_set=dataset(training.loc[mask]),
            early_stopping_rounds=40, verbose=50)
        self.model = model
        model.save_model(str(self.root / ("data/pool_eval/production_ranker.cbm" if production else "data/pool_eval/updated_ranker.cbm")))
        old_mask = legacy.context_id.isin(valid_groups)
        def legacy_dataset(df):
            return Pool(df[LEGACY_FEATURES], df.label, group_id=df.context_id)
        self.old_model = CatBoostRanker(loss_function="PairLogit", iterations=250, depth=6,
            learning_rate=.08, random_seed=42, thread_count=4, allow_writing_files=False)
        self.old_model.fit(legacy_dataset(legacy.loc[~old_mask]), eval_set=legacy_dataset(legacy.loc[old_mask]),
            early_stopping_rounds=40, verbose=50)
        self.old_model.save_model(str(self.root / ("data/pool_eval/production_legacy_ranker.cbm" if production else "data/pool_eval/legacy_control_ranker.cbm")))
        self.training_info = {"requested_contexts": len(queries), "contexts_with_positive_in_pool": len(groups),
            "training_rows": len(training), "positives": int(training.label.sum()),
            "selected_pool": self.pool_variant, "trees": model.tree_count_}
        return self.training_info
