from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eval.split import DEFAULT_SEARCH_COLS
from preprocessing import extract_label_value, normalize_query, tokenize

from .artifacts import load_char_artifacts, load_field_indexes
from .features import LEGACY_CHANNELS, FeatureCatalog, build_features
from .geo import build_loc_centroid, haversine
from .retrieval import (
    DenseChannel,
    FuzzyLookup,
    GeographicBM25,
    reciprocal_rank_fusion,
    stable_topk,
)


@dataclass
class QueryCandidates:
    cid: int
    pools: dict
    features: pd.DataFrame
    geo50: list
    cold: bool


class PoolEngine:
    def __init__(self, root, items, history, baseline_index, weighted_index,
                 char_matrix=None, dense_name="e5_base_enriched", fields=None, dense=None):
        self.root = Path(root)
        self.items = items.drop_duplicates("item_id").reset_index(drop=True).copy()
        self.ids = self.items.item_id.to_numpy()
        self.pos = {iid: i for i, iid in enumerate(self.ids)}
        self.base, self.weighted = baseline_index, weighted_index
        self.centroids = build_loc_centroid(history)
        self.geo = GeographicBM25(self.items, self.weighted, self.centroids)
        self.history = history
        self.feature_catalog = FeatureCatalog(self.items, history)
        self.popularity = self.feature_catalog.popularity
        self.pop, self.pop_norm = self.feature_catalog.pop, self.feature_catalog.pop_norm
        self.fuzzy = FuzzyLookup.from_history(history, self.ids)
        self.known_texts = self.feature_catalog.known_texts
        self.lat = self.items.item_latitude.astype(float).to_numpy()
        self.lon = self.items.item_longitude.astype(float).to_numpy()
        self.locations = self.feature_catalog.locations
        self.vid, self.tip = self.feature_catalog.vid, self.feature_catalog.tip
        self.char_matrix = char_matrix
        self.char_right = None
        self.dense = dense if dense is not None else (DenseChannel(root, self.items, dense_name) if dense_name else None)
        self.fields = fields or load_field_indexes(self.root)
        self.context_lookup = defaultdict(Counter)
        contexts = history[DEFAULT_SEARCH_COLS].astype(str).agg("||".join, axis=1)
        for context, iid in zip(contexts, history.item_id):
            if iid in self.pos:
                self.context_lookup[context][iid] += 1
        self.title_words = self.feature_catalog.title_words
        self.numeric = self.feature_catalog.numeric
        self.location_lookup = self._lookup_values(self.locations)
        self.vid_lookup, self.tip_lookup = self._lookup_values(self.vid), self._lookup_values(self.tip)

    def _lookup_values(self, values):
        groups = defaultdict(list)
        for i, value in enumerate(values):
            if value:
                groups[value].append(i)
        return {v: sorted(pos, key=lambda p: (-self.pop[p], self.ids[p])) for v, pos in groups.items()}

    def char_vectors(self, queries):
        vectorizer, self.char_matrix = load_char_artifacts(self.root, self.char_matrix)
        if self.char_right is None:
            self.char_right = self.char_matrix.T.tocsr()
        return vectorizer.transform(queries.search_query.tolist()).astype(np.float32)

    def _rank(self, values, k, eligible=None):
        return stable_topk(values, self.ids, k, eligible=eligible)

    def _ids(self, positions):
        return self.ids[np.asarray(positions, dtype=int)].tolist()

    def iter_candidates(self, queries, batch_size=25):
        print(f"Подготовка каналов: {len(queries)} запросов", flush=True)
        qtokens = queries.search_query.apply(tokenize).tolist()
        qchar = self.char_vectors(queries)
        qdense = self.dense.encode(queries) if self.dense else None
        for begin in range(0, len(queries), batch_size):
            batch = queries.iloc[begin:begin + batch_size]
            char_sim = (qchar[begin:begin + len(batch)] @ self.char_right).tocsr()
            if self.dense:
                dense_sc, dense_idx = self.dense.index.search(qdense[begin:begin + len(batch)], 100)
            for j, row in enumerate(batch.itertuples()):
                tokens = qtokens[begin + j]
                base = self.base.get_scores(tokens) if tokens else np.zeros(len(self.ids), np.float32)
                weighted = self.geo.scores(tokens)
                field_scores = {name: index.get_scores(tokens) if tokens else np.zeros(len(self.ids), np.float32)
                    for name, index in self.fields.items()}
                field_pos = {name: self._rank(scores, 100) for name, scores in field_scores.items()}
                bpos, wpos = self._rank(base, 2000), self._rank(weighted, 2000)
                local = self._rank(weighted, 2000, self.geo.local_positions(row.search_location_id))
                local50, _ = self.geo.search(tokens, row.search_location_id, k=50, scores=weighted)
                near = np.empty(0, int)
                distance = None
                if row.search_location_id in self.centroids.index:
                    coords = self.centroids.loc[row.search_location_id, ["item_latitude", "item_longitude"]].to_numpy(float)
                    if np.isfinite(coords).all():
                        _, near_idx = self.geo.tree.query(np.radians(coords[None]), k=min(300, len(self.geo.tree_positions)))
                        near = self.geo.tree_positions[near_idx[0]]
                        distance = haversine(coords[0], coords[1], self.lat, self.lon)
                fuzzy_map = self.fuzzy.scores(row.search_query)
                predicted_microcat = self.feature_catalog.predict_subcategory(fuzzy_map)
                fuzzy_ids = sorted(fuzzy_map, key=lambda iid: (-fuzzy_map[iid], iid))[:100]
                fpos = np.array([self.pos[iid] for iid in fuzzy_ids], int)
                cr = char_sim.getrow(j)
                csel = stable_topk(cr.data, self.ids[cr.indices], 100)
                cpos = cr.indices[csel]
                dpos = np.empty(0, int)
                if self.dense:
                    dpos = np.array([self.pos[self.dense.ids[i]] for i in dense_idx[j] if i >= 0], int)
                text = normalize_query(row.search_query)
                exact = self.fuzzy.lookup.get(text, {})
                context = "||".join(str(getattr(row, col)) for col in DEFAULT_SEARCH_COLS)
                exact_context = self.context_lookup.get(context, {})
                lookup_scores = {iid: exact.get(iid, 0) + 2 * exact_context.get(iid, 0)
                    for iid in set(exact) | set(exact_context) if iid in self.pos}
                lookup_ids = sorted(lookup_scores, key=lambda iid: (-lookup_scores[iid], iid))[:100]
                lookup = np.array([self.pos[iid] for iid in lookup_ids], int)
                qvid, qtip = extract_label_value(row.search_infm_params_text, "Вид услуги"), extract_label_value(row.search_infm_params_text, "Тип услуги")
                filtered = self.tip_lookup.get(qtip, [])[:100] if qtip else self.vid_lookup.get(qvid, [])[:100]
                exact_geo = self.location_lookup.get(row.search_location_id, [])[:100]
                def union(*positions):
                    return np.unique(np.concatenate([np.asarray(p, dtype=int) for p in positions]))
                raw = union(bpos, near)
                revised = union(wpos, near)
                with_local = union(revised, local)
                with_fuzzy = union(with_local, fpos)
                with_char = union(with_fuzzy, cpos)
                full = union(with_char, bpos, dpos, lookup, filtered, exact_geo, *field_pos.values())
                char_values = np.zeros(len(self.ids), np.float32)
                char_values[cr.indices] = cr.data
                fuzzy_values = np.array([fuzzy_map.get(iid, 0.) for iid in self.ids[full]])
                fuzzy_values = np.log1p(fuzzy_values)
                fuzzy_values /= max(float(fuzzy_values.max()), 1.)
                dense_values = np.zeros(len(full), np.float32)
                if self.dense:
                    dense_values = self.dense.embeddings[self.dense.item_to_dense[full]] @ qdense[begin + j]
                dist = distance[full] if distance is not None else np.full(len(full), np.nan)
                vid = (self.vid[full] == qvid) if qvid else np.zeros(len(full), bool)
                tip = (self.tip[full] == qtip) if qtip else np.zeros(len(full), bool)
                category = np.where(tip, 1., np.where(vid, .5, 0.))
                bm = weighted[full] / max(float(weighted.max()), 1e-12)
                geo_original = np.nan_to_num(1 / (1 + dist))
                geo_radius = np.nan_to_num(1 / (1 + dist / 20.))
                score = bm + .5 * geo_original + .3 * category + .1 * self.pop_norm[full]
                char_normalized = char_values[full] / max(float(char_values.max()), 1e-12)
                dense_normalized = np.maximum(dense_values, 0) / max(float(np.max(dense_values)), 1e-12)
                offsets = {int(p): i for i, p in enumerate(full)}
                def select(candidates, scores, k=500):
                    indexes = np.array([offsets[int(p)] for p in candidates], dtype=int)
                    chosen = stable_topk(scores[indexes], self.ids[candidates], k, positive_only=False)
                    return self._ids(candidates[chosen])
                base_known = np.zeros(len(self.ids), np.float32);base_known[bpos] = base[bpos]
                original = base_known[full] / max(float(base.max()), 1e-12) + .3 * category + .1 * self.pop_norm[full]
                original += .5 * geo_original * np.isin(full, near)
                fixed = base_known[full] / max(float(base.max()), 1e-12) + .5 * geo_original + .3 * category + .1 * self.pop_norm[full]
                pools = {"old_score": select(raw, original), "fixed_geo_score": select(raw, fixed),
                    "weighted_bm25": select(revised, score), "local_bm25": select(with_local, score),
                    "plus_fuzzy": select(with_fuzzy, score + .1 * fuzzy_values),
                    "plus_char": select(with_char, score + .1 * fuzzy_values + .1 * char_normalized),
                    "all_channels": select(full, score + .1 * fuzzy_values + .1 * char_normalized + .1 * dense_normalized)}
                for weight in [.5, 1., 2.]:
                    revised_score = bm + weight * geo_radius + .3 * category + .1 * self.pop_norm[full]
                    revised_score += .1 * fuzzy_values + .1 * char_normalized + .1 * dense_normalized
                    pools[f"all_geo_weight_{weight:g}"] = select(full, revised_score)
                legacy_lists = [self._ids(p) for p in [bpos[:100], dpos, lookup, near[:100], np.asarray(filtered), np.asarray(exact_geo)]]
                pools["old_6_channels_rrf"] = reciprocal_rank_fusion(*legacy_lists, top_n=500)
                pools["old_6_channels_union"] = self._ids(union(bpos[:100], dpos, lookup, near[:100], filtered, exact_geo))
                pools["wide_union"] = self._ids(full)
                channel_positions = dict(zip(LEGACY_CHANNELS, [bpos[:100], dpos, lookup, near[:100], filtered, exact_geo]))
                channel_positions.update(weighted=wpos, local=local, fuzzy=fpos, char=cpos, **field_pos)
                channel_values = {"bm25": base[full], "weighted": weighted[full], "local": weighted[full],
                    "dense": dense_values, "lookup": np.array([lookup_scores.get(iid, np.nan) for iid in self.ids[full]]),
                    "geo": dist, "filter": self.pop[full], "geo_exact": self.pop[full],
                    "fuzzy": np.array([fuzzy_map.get(iid, 0) for iid in self.ids[full]]), "char": char_values[full]}
                channel_values.update({name: scores[full] for name, scores in field_scores.items()})
                frame = build_features(self.feature_catalog, row, full, bm=bm, dist=dist,
                    geo_radius=geo_radius, vid=vid, tip=tip, fuzzy_values=fuzzy_values,
                    char_values=char_values, dense_values=dense_values, text=text,
                    qvid=qvid, qtip=qtip, channel_positions=channel_positions,
                    channel_values=channel_values, field_scores=field_scores,
                    predicted_microcat=predicted_microcat)
                yield QueryCandidates(int(row.context_id), pools, frame, local50, text not in self.known_texts)
            if (begin + len(batch)) % 500 == 0:
                print(f"каналы: {begin + len(batch)}/{len(queries)}", flush=True)
