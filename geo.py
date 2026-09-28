import numpy as np
from sklearn.neighbors import BallTree


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_loc_centroid(df):
    sp = df.copy()
    sp["item_latitude"] = sp["item_latitude"].astype(float)
    sp["item_longitude"] = sp["item_longitude"].astype(float)
    return sp.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()


def build_geo_tree(items_df):
    items = items_df.copy()
    items["item_latitude"] = items["item_latitude"].astype(float)
    items["item_longitude"] = items["item_longitude"].astype(float)
    items = items.dropna(subset=["item_latitude", "item_longitude"]).reset_index(drop=True)
    item_ids = items["item_id"].tolist()
    tree = BallTree(np.radians(items[["item_latitude", "item_longitude"]].to_numpy()), metric="haversine")
    return tree, item_ids
