from collections import defaultdict

import numpy as np
import pandas as pd

from preprocessing import extract_label_value, normalize_query

FEATURES = ["bm25_norm", "geo_distance_km", "geo_bonus", "geo_exact", "vid_match",
    "tip_match", "log_popularity", "fuzzy_score", "char_score", "dense_score",
    "title_coverage", "item_price", "item_rating", "log_reviews", "subcat_match"]
LEGACY_CHANNELS = ["bm25", "dense", "lookup", "geo", "filter", "geo_exact"]
LEGACY_FEATURES = ["geo_distance_km", "geo_exact_match", "vid_uslugi_match", "tip_uslugi_match",
    "item_price", "item_rating", "item_rating_reviews_count", "item_is_phone_hidden",
    "item_is_message_forbidden", "item_popularity", "text_overlap"] + [
    f"{channel}_{suffix}" for channel in LEGACY_CHANNELS for suffix in ["rank", "present", "score"]]
CHANNELS = LEGACY_CHANNELS + ["weighted", "local", "fuzzy", "char", "title", "desc", "params"]
FEATURES = list(dict.fromkeys(FEATURES + LEGACY_FEATURES + [
    f"{channel}_{suffix}" for channel in CHANNELS for suffix in ["rank", "present", "score"]] +
    ["title_norm", "desc_norm", "params_norm", "query_words", "query_known", "delivery_search"]))


class FeatureCatalog:
    def __init__(self, items, history):
        self.ids = items.item_id.to_numpy()
        self.popularity = history.item_id.value_counts()
        self.pop = self.popularity.reindex(self.ids).fillna(0).to_numpy(float)
        self.pop_norm = np.log1p(self.pop) / max(np.log1p(self.pop.max()), 1.)
        self.known_texts = set(history.search_query.map(normalize_query))
        self.locations = items.item_location_id.to_numpy()
        self.vid = items.item_infm_params_text.apply(lambda t: extract_label_value(t, "Вид услуги")).to_numpy()
        self.tip = items.item_infm_params_text.apply(lambda t: extract_label_value(t, "Тип услуги")).to_numpy()
        self.title_words = [set(t.lower().split()) for t in items.item_title_raw.fillna("")]
        self.numeric = {name: pd.to_numeric(items[name], errors="coerce").to_numpy(float, copy=True)
            for name in ["item_price", "item_rating", "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden"]}
        self.numeric["item_price"][self.numeric["item_price"] == -1] = np.nan
        self.microcat = items.item_microcat_id.to_numpy()
        self.id_to_microcat = dict(zip(self.ids, self.microcat))

    def predict_subcategory(self, fuzzy_map):
        if not fuzzy_map:
            return None
        scores = defaultdict(float)
        for iid, score in fuzzy_map.items():
            mc = self.id_to_microcat.get(iid)
            if mc is not None:
                scores[mc] += score
        return max(scores, key=scores.get) if scores else None


def build_features(catalog, row, full, *, bm, dist, geo_radius, vid, tip,
                   fuzzy_values, char_values, dense_values, text, qvid, qtip,
                   channel_positions, channel_values, field_scores, predicted_microcat=None):
    qwords = set(text.split())
    subcat_match = (catalog.microcat[full] == predicted_microcat).astype(np.float32) \
        if predicted_microcat is not None else np.zeros(len(full), np.float32)
    overlap = np.array([len(qwords & catalog.title_words[p]) for p in full])
    coverage = overlap / max(len(qwords), 1)
    frame = pd.DataFrame({"context_id": row.context_id, "item_id": catalog.ids[full],
        "bm25_norm": bm, "geo_distance_km": dist, "geo_bonus": geo_radius,
        "geo_exact": (catalog.locations[full] == row.search_location_id).astype(np.float32),
        "vid_match": vid.astype(np.float32), "tip_match": tip.astype(np.float32),
        "log_popularity": np.log1p(catalog.pop[full]), "fuzzy_score": fuzzy_values,
        "char_score": char_values[full], "dense_score": dense_values,
        "title_coverage": coverage, "item_price": catalog.numeric["item_price"][full],
        "item_rating": catalog.numeric["item_rating"][full],
        "log_reviews": np.log1p(np.maximum(np.nan_to_num(catalog.numeric["item_rating_reviews_count"][full]), 0)),
        "geo_exact_match": (catalog.locations[full] == row.search_location_id).astype(np.float32),
        "vid_uslugi_match": vid.astype(float) if qvid else np.nan,
        "tip_uslugi_match": tip.astype(float) if qtip else np.nan,
        "item_rating_reviews_count": catalog.numeric["item_rating_reviews_count"][full],
        "item_is_phone_hidden": catalog.numeric["item_is_phone_hidden"][full],
        "item_is_message_forbidden": catalog.numeric["item_is_message_forbidden"][full],
        "item_popularity": catalog.pop[full], "text_overlap": overlap,
        "query_words": len(qwords), "query_known": text in catalog.known_texts,
        "delivery_search": row.search_is_delivery_search, "subcat_match": subcat_match})
    for name, positions in channel_positions.items():
        ranks = {int(p): rank for rank, p in enumerate(positions, 1)}
        values = np.array([ranks.get(int(p), 0) for p in full])
        frame[f"{name}_rank"] = values
        frame[f"{name}_present"] = values > 0
        frame[f"{name}_score"] = channel_values[name] if name not in LEGACY_CHANNELS else np.where(values > 0, channel_values[name], np.nan)
    for name, scores in field_scores.items():
        frame[f"{name}_norm"] = scores[full] / max(float(scores.max()), 1e-12)
    frame[FEATURES] = frame[FEATURES].astype(np.float32)
    return frame
