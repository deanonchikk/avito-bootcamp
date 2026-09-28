import hashlib
import json
from pathlib import Path
import pickle

import bm25s
import numpy as np
import pandas as pd
from scipy import sparse

from preprocessing import build_char_item_text, tokenize


def load_bm25(path):
    return bm25s.BM25.load(str(path), mmap=True, show_progress=False)


def load_field_indexes(root):
    return {name: load_bm25(Path(root) / f"data/bm25s_index_{name}")
            for name in ["title", "desc", "params"]}


def load_char_artifacts(root, matrix=None):
    folder = Path(root) / "data/tfidf_char_index"
    with (folder / "vectorizer.pkl").open("rb") as stream:
        vectorizer = pickle.load(stream)
    if matrix is None:
        matrix = sparse.load_npz(folder / "corpus_matrix.npz")
    return vectorizer, matrix


def load_item_embeddings(root, name):
    folder = Path(root) / "data/embeddings" / name
    embeddings = np.load(folder / "512.npy", mmap_mode="r")
    ids = np.load(folder / "512_item_ids.npy", allow_pickle=True)
    if embeddings.ndim != 2 or len(embeddings) != len(ids):
        raise ValueError("кол-во item_id не совпадает с матрицей эмбеддингов")
    return embeddings, ids


def query_embedding_path(root, texts):
    key = hashlib.sha256(json.dumps(["e5-base:512:plain:v1", texts], ensure_ascii=False).encode()).hexdigest()
    folder = Path(root) / "data/query_embeddings_cache"
    folder.mkdir(exist_ok=True)
    return folder / f"{key}.npy"


def load_query_embeddings(root, texts):
    path = query_embedding_path(root, texts)
    if not path.exists():
        return None
    encoded = np.load(path)
    if encoded.ndim != 2 or len(encoded) != len(texts):
        raise ValueError("Query embedding cache не соответствует запросам")
    return encoded


def save_query_embeddings(root, texts, encoded):
    np.save(query_embedding_path(root, texts), encoded)


def pipeline_signature(extra=b""):
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "pipeline").glob("*.py"))
    paths += sorted((root / "preprocessing").glob("*.py"))
    paths += sorted((root / "eval").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(extra)
    return digest.hexdigest()


def benchmark_assets(root):
    root = Path(root)
    items = pd.read_parquet(root / "data/benchmark_items.parquet").drop_duplicates("item_id").reset_index(drop=True)
    folder = root / "data/updated_benchmark_index"
    folder.mkdir(exist_ok=True)
    signature = hashlib.sha256("\n".join(items.item_id.astype(str)).encode()).hexdigest()
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["item_ids_sha256"] != signature:
        raise ValueError("Порядок benchmark-каталога изменился: нужен новый cache")
    manifest_path.write_text(json.dumps({"item_ids_sha256": signature, "count": len(items),
        "weighted_bm25": [3, 1, 1], "char": "train-vectorizer;title+desc[:200]+params[:100]"}, indent=2))
    paths = {name: folder / name for name in ["title", "desc", "params", "weighted"]}
    if not all((path / "params.index.json").exists() for path in paths.values()):
        token_path = folder / "field_tokens.pkl"
        if token_path.exists():
            with token_path.open("rb") as stream:
                tokens = pickle.load(stream)
        else:
            tokens = {}
            for name, col in [("title", "item_title_raw"), ("desc", "item_description_raw"), ("params", "item_infm_params_text")]:
                print(f"benchmark: токенизация {name}", flush=True)
                tokens[name] = []
                values = items[col].tolist()
                for start in range(0, len(values), 20000):
                    tokens[name].extend(tokenize(value) for value in values[start:start + 20000])
                    print(f"{name}: {min(start + 20000, len(values))}/{len(values)}", flush=True)
            with token_path.open("wb") as stream:
                pickle.dump(tokens, stream)
        for name, path in paths.items():
            if (path / "params.index.json").exists():
                continue
            print(f"benchmark: индекс {name}", flush=True)
            documents = tokens[name] if name != "weighted" else [t * 3 + d + p
                for t, d, p in zip(tokens["title"], tokens["desc"], tokens["params"])]
            index = bm25s.BM25()
            index.index(documents, show_progress=False)
            index.save(str(path))
            del index, documents
    matrix_path = folder / "char_matrix.npz"
    if matrix_path.exists():
        matrix = sparse.load_npz(matrix_path)
    else:
        print("benchmark: char TF-IDF", flush=True)
        with (root / "data/tfidf_char_index/vectorizer.pkl").open("rb") as stream:
            vectorizer = pickle.load(stream)
        texts = [build_char_item_text(title, description, params) for title, description, params in zip(
            items.item_title_raw, items.item_description_raw, items.item_infm_params_text)]
        matrix = vectorizer.transform(texts).astype(np.float32)
        sparse.save_npz(matrix_path, matrix)
    base = bm25s.BM25.load(str(root / "data/bm25s_index_bench"), mmap=True, show_progress=False)
    weighted = bm25s.BM25.load(str(paths["weighted"]), mmap=True, show_progress=False)
    fields = {name: bm25s.BM25.load(str(paths[name]), mmap=True, show_progress=False) for name in ["title", "desc", "params"]}
    return items, base, weighted, fields, matrix
