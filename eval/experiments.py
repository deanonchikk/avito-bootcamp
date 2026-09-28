from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipeline.artifacts import benchmark_assets
from .metrics import recall_at_50, recall_at_k
from pipeline.pool import PoolEngine
from pipeline.ranker import predict_scores, rank_items
from pipeline.retrieval import reciprocal_rank_fusion
from pipeline.submission import apply_policy


def metric_table(predictions, relevant):
    return pd.DataFrame([{"variant": name, "Recall@50": recall_at_50(pred, relevant),
        "Recall@500": recall_at_k(pred, relevant, 500) if name != "wide_union" else np.nan,
        "pool_ceiling": recall_at_k(pred, relevant, max((len(p) for p in pred.values()), default=1)),
        "mean_pool_size": np.mean([len(p) for p in pred.values()])}
        for name, pred in predictions.items()]).set_index("variant")


@dataclass
class RankingEvaluation:
    root: object
    features: pd.DataFrame
    predictions: dict
    relevant: dict
    cold: dict
    pool_variant: str
    model: object
    old_model: object

    def evaluate_ranker(self):
        old_model = self.old_model
        old_frames = []
        selected = []
        for cid, frame in self.features.groupby("context_id", sort=False):
            allowed = set(self.predictions[self.pool_variant][cid])
            selected.append(frame.loc[frame.item_id.isin(allowed)])
            old_allowed = set(self.predictions["old_6_channels_union"][cid])
            old_frames.append(frame.loc[frame.item_id.isin(old_allowed)])
        frame = pd.concat(selected, ignore_index=True)
        frame["score"] = predict_scores(self.model, frame)
        ranked = frame.sort_values(["context_id", "score", "item_id"], ascending=[True, False, True])
        pred = ranked.groupby("context_id", sort=False).item_id.agg(list).to_dict()
        self.predictions["updated_catboost"] = pred
        legacy = pd.concat(old_frames, ignore_index=True)
        legacy["score"] = predict_scores(old_model, legacy, legacy=True)
        self.predictions["old_control_catboost"] = legacy.sort_values(
            ["context_id", "score", "item_id"], ascending=[True, False, True]).groupby("context_id").item_id.agg(list).to_dict()
        for weight in [.25, .5, 1., 2.]:
            self.predictions[f"ranker_geo_rrf_{weight:g}"] = {cid: reciprocal_rank_fusion(
                pred.get(cid, []), self.predictions["weighted_geo50"][cid], weights=[1, weight], top_n=500)
                for cid in self.relevant}
        table = metric_table(self.predictions, self.relevant)
        cold_relevant = {cid: r for cid, r in self.relevant.items() if self.cold[cid]}
        warm_relevant = {cid: r for cid, r in self.relevant.items() if not self.cold[cid]}
        cohorts = pd.DataFrame([{"variant": name, "warm_recall50": recall_at_50(p, warm_relevant),
            "cold_recall50": recall_at_50(p, cold_relevant)} for name, p in self.predictions.items()]).set_index("variant")
        table.to_csv(self.root / "data/pool_eval/final_comparison.csv")
        cohorts.to_csv(self.root / "data/pool_eval/warm_cold.csv")
        return table, cohorts, {"warm_contexts": len(warm_relevant), "cold_contexts": len(cold_relevant)}


@dataclass
class BenchmarkProxy:
    root: object
    history: pd.DataFrame
    val_relevant: dict
    val_queries: pd.DataFrame
    engine: PoolEngine
    pool_variant: str
    model: object
    old_model: object

    def evaluate_benchmark_proxy(self, sample_size=None):
        items, base, weighted, fields, matrix = benchmark_assets(self.root)
        engine = PoolEngine(self.root, items, self.history, base, weighted, char_matrix=matrix,
            dense_name="e5_base_bench", fields=fields)
        if self.engine.dense.model is not None:
            engine.dense.model = self.engine.dense.model
        self.benchmark_engine = engine
        allowed = set(items.item_id)
        relevant = {cid: targets & allowed for cid, targets in self.val_relevant.items() if targets & allowed}
        queries = self.val_queries.loc[self.val_queries.context_id.isin(relevant)].copy()
        if sample_size:
            queries = queries.sample(min(sample_size, len(queries)), random_state=31)
            relevant = {cid: relevant[cid] for cid in queries.context_id}
        print(f"benchmark proxy: {len(queries)} контекстов, только релевантные item benchmark", flush=True)
        predictions = defaultdict(dict);cold = {}
        with threadpool_limits(limits=4):
            for candidate in engine.iter_candidates(queries):
                cid = candidate.cid
                frame = candidate.features.loc[candidate.features.item_id.isin(candidate.pools[self.pool_variant])].copy()
                ranked = rank_items(self.model, frame)
                predictions["updated_catboost"][cid] = ranked
                predictions["weighted_geo50"][cid] = candidate.geo50
                predictions["pool_heuristic"][cid] = candidate.pools[self.pool_variant]
                legacy = candidate.features.loc[candidate.features.item_id.isin(candidate.pools["old_6_channels_union"])].copy()
                predictions["old_catboost"][cid] = rank_items(self.old_model, legacy, legacy=True)
                for weight in [.25, .5, 1., 2.]:
                    for family in ["ranker_geo_rrf", "ranker_heuristic_rrf", "ranker_heuristic_geo_rrf"]:
                        name = f"{family}_{weight:g}"
                        predictions[name][cid] = apply_policy(ranked,
                            candidate.pools[self.pool_variant], candidate.geo50, name, top_n=500)
                cold[cid] = candidate.cold
        ids = np.array(list(relevant));np.random.RandomState(19).shuffle(ids)
        dev_ids, test_ids = set(ids[:int(len(ids) * .7)]), set(ids[int(len(ids) * .7):])
        subsets = {"all": relevant, "policy_dev": {cid: relevant[cid] for cid in dev_ids},
            "policy_holdout": {cid: relevant[cid] for cid in test_ids}}
        candidates = [name for name in predictions if name != "old_catboost"]
        def policy_metric(name, labels):
            warm = {cid: r for cid, r in labels.items() if not cold[cid]}
            cold_labels = {cid: r for cid, r in labels.items() if cold[cid]}
            return .37 * recall_at_50(predictions[name], warm) + .63 * recall_at_50(predictions[name], cold_labels)
        self.final_policy = max(candidates, key=lambda name: policy_metric(name, subsets["policy_dev"]))
        rows = []
        for subset, labels in subsets.items():
            for name, pred in predictions.items():
                warm = {cid: r for cid, r in labels.items() if not cold[cid]}
                cold_labels = {cid: r for cid, r in labels.items() if cold[cid]}
                rows.append({"subset": subset, "variant": name, "contexts": len(labels),
                    "cold_contexts": len(cold_labels), "Recall@50": recall_at_50(pred, labels),
                    "Recall@500": recall_at_k(pred, labels, 500), "warm_recall50": recall_at_50(pred, warm),
                    "cold_recall50": recall_at_50(pred, cold_labels), "benchmark_mix_recall50": policy_metric(name, labels)})
        table = pd.DataFrame(rows)
        table.to_csv(self.root / "data/pool_eval/benchmark_proxy.csv", index=False)
        print(f"Политика для submission: {self.final_policy}", flush=True)
        return table
