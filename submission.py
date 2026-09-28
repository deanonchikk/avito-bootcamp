from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from features import FEATURES
from pool import PoolEngine
from ranker import rank_items
from retrieval import reciprocal_rank_fusion


def apply_policy(ranked, heuristic, geo50, policy, *, top_n=50):
    if policy == "weighted_geo50":
        return geo50
    if policy == "pool_heuristic":
        return heuristic
    for prefix, lists, weights in [
        ("ranker_heuristic_geo_rrf_", [ranked, heuristic, geo50], [1, None, .5]),
        ("ranker_heuristic_rrf_", [ranked, heuristic], [1, None]),
        ("ranker_geo_rrf_", [ranked, geo50], [1, None]),
    ]:
        if policy.startswith(prefix):
            weights[1] = float(policy.removeprefix(prefix))
            return reciprocal_rank_fusion(*lists, weights=weights, top_n=top_n)
    if policy != "updated_catboost":
        raise ValueError(f"Неизвестная политика выдачи: {policy}")
    return ranked


def validate_predictions(queries, predictions, allowed):
    if not queries.query_id.is_unique or set(predictions) != set(queries.context_id):
        raise ValueError("Набор предсказаний не соответствует уникальным query_id")
    for cid, items in predictions.items():
        if len(items) != 50 or len(set(items)) != 50 or not set(items) <= allowed:
            raise ValueError(f"Контекст {cid}: нужны 50 уникальных item_id из benchmark")


@dataclass
class SubmissionWriter:
    root: object
    benchmark_engine: PoolEngine
    train: pd.DataFrame
    pool_variant: str
    model: object
    final_policy: str
    training_info: dict

    def make_submission(self, path="submissions/v2.csv"):
        assets = self.benchmark_engine
        engine = PoolEngine(self.root, assets.items, self.train, assets.base, assets.weighted,
            char_matrix=assets.char_matrix, fields=assets.fields, dense=assets.dense)
        engine.char_right = assets.char_right
        queries = pd.read_parquet(self.root / "data/benchmark_queries.parquet")
        if not queries.query_id.is_unique:
            raise ValueError("query_id не уникальны")
        queries = queries.assign(context_id=np.arange(len(queries)))
        predictions = {}
        with threadpool_limits(limits=4):
            for candidate in engine.iter_candidates(queries):
                frame = candidate.features.loc[candidate.features.item_id.isin(candidate.pools[self.pool_variant])].copy()
                ranked = rank_items(self.model, frame)
                ranked = apply_policy(ranked, candidate.pools[self.pool_variant],
                                      candidate.geo50, self.final_policy)
                if len(ranked) < 50:
                    ranked = list(dict.fromkeys(ranked + candidate.pools[self.pool_variant]))
                predictions[candidate.cid] = ranked[:50]
        allowed = set(engine.ids)
        validate_predictions(queries, predictions, allowed)
        answer = pd.DataFrame({"query_id": queries.query_id,
            "answer": [" ".join(predictions[cid]) for cid in queries.context_id]})
        output = self.root / path
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Сабмит уже существует: {output}")
        answer.to_csv(output, index=False)
        metadata = {"file": path, "queries": len(answer), "items_per_query": 50,
            "pool_variant": self.pool_variant, "final_policy": self.final_policy,
            "features": FEATURES, "training": self.training_info,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        return metadata
