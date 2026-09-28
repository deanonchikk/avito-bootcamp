import heapq
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from preprocessing import build_e5_query_text, normalize_query

from .artifacts import (
    load_item_embeddings,
    load_query_embeddings,
    save_query_embeddings,
)
from .geo import build_geo_tree


def stable_topk(scores, item_ids, k=50, eligible=None, positive_only=True):
    values = np.asarray(scores)
    positions = np.arange(len(values)) if eligible is None else np.asarray(eligible, dtype=int)
    mask = np.isfinite(values[positions])
    if positive_only:
        mask &= values[positions] > 0
    positions = positions[mask]
    if k <= 0 or not len(positions):
        return np.empty(0, dtype=int)
    if len(positions) > k:
        threshold = np.partition(values[positions], len(positions) - k)[-k]
        greater = positions[values[positions] > threshold]
        ties = positions[values[positions] == threshold]
        tie_order = np.argsort(np.asarray(item_ids)[ties], kind="stable")
        positions = np.concatenate([greater, ties[tie_order[:k - len(greater)]]])
    order = np.lexsort((np.asarray(item_ids)[positions], -values[positions]))
    return positions[order][:k]


def reciprocal_rank_fusion(*ranked_lists, weights=None, k=60, top_n=50):
    weights = [1.0] * len(ranked_lists) if weights is None else weights
    if len(weights) != len(ranked_lists):
        raise ValueError("Для каждого списка нужен один вес")
    scores = defaultdict(float)
    for weight, ranked in zip(weights, ranked_lists):
        seen = set()
        rank = 0
        for iid in ranked:
            if iid in seen:
                continue
            seen.add(iid)
            rank += 1
            scores[iid] += weight / (k + rank)
    return sorted(scores, key=lambda iid: (-scores[iid], iid))[:top_n]


def sparse_topk(query_matrix, corpus_matrix, item_ids, k=50, batch_size=100):
    right = corpus_matrix.T.tocsr()
    for begin in range(0, query_matrix.shape[0], batch_size):
        similarities = (query_matrix[begin:begin + batch_size] @ right).tocsr()
        for j in range(similarities.shape[0]):
            row = similarities.getrow(j)
            positive = row.data > 0
            idx, values = row.indices[positive], row.data[positive]
            selected = stable_topk(values, np.asarray(item_ids)[idx], k)
            yield begin + j, idx[selected], values[selected]


class FuzzyLookup:
    def __init__(self, text_lookup, allowed_ids=None):
        self.allowed_ids = set(allowed_ids) if allowed_ids is not None else None
        self.lookup = text_lookup
        self.words = {q: frozenset(q.split()) for q in text_lookup}
        self.postings = defaultdict(set)
        for q in sorted(self.words):
            for word in sorted(self.words[q]):
                self.postings[word].add(q)

    @classmethod
    def from_history(cls, history, allowed_ids=None):
        lookup = defaultdict(Counter)
        for q, iid in zip(history.search_query.map(normalize_query), history.item_id):
            lookup[q][iid] += 1
        return cls(lookup, allowed_ids)

    def scores(self, text, min_jaccard=.34, n_rarest=2, max_candidates=2500):
        words = frozenset(normalize_query(text).split())
        if not words:
            return {}
        rarest = sorted(words, key=lambda w: (len(self.postings.get(w, ())), w))[:n_rarest]
        candidates = set().union(*(self.postings.get(w, set()) for w in rarest))
        similar = []
        for q in sorted(candidates):
            other = self.words[q]
            similarity = len(words & other) / len(words | other)
            if similarity >= min_jaccard:
                similar.append((similarity, q))
        selected = heapq.nsmallest(max_candidates, similar, key=lambda x: (-x[0], x[1]))
        item_scores = defaultdict(float)
        for similarity, q in selected:
            for iid, count in sorted(self.lookup[q].items()):
                if self.allowed_ids is None or iid in self.allowed_ids:
                    item_scores[iid] += count * similarity
        return dict(item_scores)

    def predict(self, text, k=50):
        scores = self.scores(text)
        return sorted(scores, key=lambda iid: (-scores[iid], iid))[:k]


class GeographicBM25:
    def __init__(self, items, retriever, centroids, radius_km=20):
        self.ids = items.item_id.to_numpy()
        if retriever.scores["num_docs"] != len(self.ids):
            raise ValueError("Размер индекса не совпадает с корпусом")
        self.retriever, self.centroids, self.radius_km = retriever, centroids, radius_km
        self.tree, tree_ids = build_geo_tree(items[["item_id", "item_latitude", "item_longitude"]])
        id_to_pos = {iid: i for i, iid in enumerate(self.ids)}
        self.tree_positions = np.array([id_to_pos[iid] for iid in tree_ids])

    @lru_cache(maxsize=256)
    def local_positions(self, location):
        if location not in self.centroids.index:
            return np.empty(0, dtype=int)
        coords = self.centroids.loc[location, ["item_latitude", "item_longitude"]].to_numpy(float)
        if not np.isfinite(coords).all():
            return np.empty(0, dtype=int)
        positions = self.tree.query_radius(np.radians(coords[None]), r=self.radius_km / 6371.0)[0]
        return self.tree_positions[positions]

    def scores(self, tokens):
        return self.retriever.get_scores(tokens) if tokens else np.zeros(len(self.ids), dtype=np.float32)

    def search(self, tokens, location, k=50, fallback=True, scores=None):
        values = self.scores(tokens) if scores is None else scores
        local = stable_topk(values, self.ids, k, eligible=self.local_positions(location))
        if fallback and len(local) < k:
            eligible = np.ones(len(self.ids), dtype=bool)
            eligible[local] = False
            extra = stable_topk(values, self.ids, k - len(local), eligible=np.flatnonzero(eligible))
            local = np.concatenate([local, extra])
        return self.ids[local].tolist(), values[local]


class DenseChannel:
    def __init__(self, root, items, name):
        import faiss
        faiss.omp_set_num_threads(4)
        self.root = Path(root)
        self.embeddings, self.ids = load_item_embeddings(self.root, name)
        dense_positions = {iid: i for i, iid in enumerate(self.ids)}
        self.item_to_dense = np.array([dense_positions[iid] for iid in items.item_id])
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.model = None

    def encode(self, queries):
        texts = [build_e5_query_text(q) for q in queries.search_query]
        cached = load_query_embeddings(self.root, texts)
        if cached is not None:
            return cached
        print(f"E5: кодирование {len(texts)} запросов", flush=True)
        if self.model is None:
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = SentenceTransformer("intfloat/multilingual-e5-base", device=device, local_files_only=True)
            self.model.max_seq_length = 512
        with threadpool_limits(limits=1):
            encoded = self.model.encode(texts, batch_size=32, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True)
        save_query_embeddings(self.root, texts, encoded)
        return encoded
