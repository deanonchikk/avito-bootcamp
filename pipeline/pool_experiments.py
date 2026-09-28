from collections import defaultdict
import json
from pathlib import Path
import pickle

import pandas as pd
from threadpoolctl import threadpool_limits

from .artifacts import benchmark_assets, load_bm25, pipeline_signature
from eval import make_val_split
from eval.experiments import BenchmarkProxy, RankingEvaluation, metric_table
from .features import CHANNELS, FEATURES, LEGACY_CHANNELS, LEGACY_FEATURES
from .pool import PoolEngine, QueryCandidates
from .ranker import RankerTrainer
from .retrieval import DenseChannel
from .submission import SubmissionWriter


class PoolExperiment:
    def __init__(self, root=".", sample_size=5000):
        self.root = Path(root).resolve()
        self.train = pd.read_parquet(self.root / "data/train.parquet")
        self.history, self.val_queries, self.val_relevant = make_val_split(self.train, random_state=42)
        self.sample = self.val_queries.sample(min(sample_size, len(self.val_queries)), random_state=1).reset_index(drop=True)
        self.relevant = {int(cid): self.val_relevant[int(cid)] for cid in self.sample.context_id}
        self.items = self.train.drop_duplicates("item_id").reset_index(drop=True)
        self.base = load_bm25(self.root / "data/bm25s_index")
        self.weighted = load_bm25(self.root / "data/bm25s_index_weighted/w311")
        self.engine = PoolEngine(self.root, self.items, self.history, self.base, self.weighted)
        self.predictions = defaultdict(dict)
        self.frames = []
        self.cold = {}
        self.pool_variant = "all_geo_weight_1"
        self.model = None
        print(f"history={len(self.history)}, eval={len(self.sample)}, items={len(self.items)}", flush=True)

    def compare(self):
        folder = self.root / "data/pool_eval";folder.mkdir(exist_ok=True)
        self.signature = pipeline_signature(self.sample.context_id.to_numpy().tobytes())
        manifest = folder / "comparison_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("signature") == self.signature:
            with (folder / "predictions.pkl").open("rb") as stream:
                self.predictions, self.cold, self.pool_variant = pickle.load(stream)
            self.features = pd.read_parquet(folder / "eval_features.parquet")
            self.table = pd.read_csv(folder / "channel_comparison.csv", index_col=0)
            print("Сравнение восстановлено из cache с тем же кодом и sample", flush=True)
            return self.table
        with threadpool_limits(limits=4):
            for candidates in self.engine.iter_candidates(self.sample):
                for name, pool in candidates.pools.items():
                    self.predictions[name][candidates.cid] = pool
                self.predictions["weighted_geo50"][candidates.cid] = candidates.geo50
                keep = set().union(*(candidates.pools[name] for name in [
                    "all_geo_weight_0.5", "all_geo_weight_1", "all_geo_weight_2", "old_6_channels_union"]))
                self.frames.append(candidates.features.loc[candidates.features.item_id.isin(keep)])
                self.cold[candidates.cid] = candidates.cold
        self.features = pd.concat(self.frames, ignore_index=True)
        self.frames.clear()
        self.table = metric_table(self.predictions, self.relevant)
        candidates = [name for name in self.table.index if name.startswith("all_geo_weight_")]
        self.pool_variant = max(candidates, key=lambda name: (self.table.loc[name, "Recall@500"], self.table.loc[name, "Recall@50"]))
        self.table.to_csv(folder / "channel_comparison.csv")
        self.features.to_parquet(folder / "eval_features.parquet", index=False)
        with (folder / "predictions.pkl").open("wb") as stream:
            pickle.dump((self.predictions, self.cold, self.pool_variant), stream)
        manifest.write_text(json.dumps({"signature": self.signature, "sample_size": len(self.sample)}))
        return self.table

    def train_ranker(self, fit_queries=2000, production=False):
        trainer = RankerTrainer(self.root, self.train, self.history, self.items,
            self.base, self.weighted, self.engine, self.pool_variant)
        self.training_info = trainer.train_ranker(fit_queries, production)
        self.model, self.old_model = trainer.model, trainer.old_model
        return self.training_info

    def evaluate_ranker(self):
        evaluation = RankingEvaluation(self.root, self.features, self.predictions,
            self.relevant, self.cold, self.pool_variant, self.model, self.old_model)
        return evaluation.evaluate_ranker()

    def evaluate_benchmark_proxy(self, sample_size=None):
        if hasattr(self, "features"):
            del self.features
        proxy = BenchmarkProxy(self.root, self.history, self.val_relevant,
            self.val_queries, self.engine, self.pool_variant, self.model, self.old_model)
        table = proxy.evaluate_benchmark_proxy(sample_size)
        self.benchmark_engine, self.final_policy = proxy.benchmark_engine, proxy.final_policy
        return table

    def make_submission(self, path="submissions/v2.csv"):
        writer = SubmissionWriter(self.root, self.benchmark_engine, self.train,
            self.pool_variant, self.model, self.final_policy, self.training_info)
        return writer.make_submission(path)
