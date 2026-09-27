import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# копия preprocessing/tokenize.py (только гомоглиф-фикс, без лемматизации) -
# автономная, чтобы не тащить весь проект в kaggle-датасет
FORWARD_MAP = str.maketrans({
    "a": "а", "A": "А", "e": "е", "E": "Е", "o": "о", "O": "О",
    "p": "р", "P": "Р", "c": "с", "C": "С", "x": "х", "X": "Х",
    "y": "у", "Y": "У", "H": "Н", "K": "К", "M": "М", "T": "Т", "B": "В",
    "b": "в", "h": "н", "k": "к", "m": "м", "t": "т", "d": "д",
})
REVERSE_MAP = str.maketrans({v: k for k, v in {
    "a": "а", "A": "А", "e": "е", "E": "Е", "o": "о", "O": "О",
    "p": "р", "P": "Р", "c": "с", "C": "С", "x": "х", "X": "Х",
    "y": "у", "Y": "У", "H": "Н", "K": "К", "M": "М", "T": "Т", "B": "В",
    "b": "в", "h": "н", "k": "к", "m": "м", "t": "т", "d": "д",
}.items()})

WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9]+")
CYR_RE = re.compile(r"[а-яёА-ЯЁ]")
LAT_RE = re.compile(r"[a-zA-Z]")
DIGIT_RE = re.compile(r"[0-9]")


def _script_purity_score(s):
    n_cyr = len(CYR_RE.findall(s))
    n_lat = len(LAT_RE.findall(s))
    return min(n_cyr, n_lat)


def normalize_homoglyphs(token):
    if DIGIT_RE.search(token):
        return token.translate(REVERSE_MAP)
    forward = token.translate(FORWARD_MAP)
    reverse = token.translate(REVERSE_MAP)
    return min((forward, reverse), key=_script_purity_score)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    def _fix(match):
        word = match.group(0)
        if CYR_RE.search(word) and LAT_RE.search(word):
            return normalize_homoglyphs(word)
        return word

    return WORD_RE.sub(_fix, text)


print("/kaggle/input contents:")
for root, _dirs, files in os.walk("/kaggle/input"):
    for f in files:
        print(" ", os.path.join(root, f))

candidates = glob.glob("/kaggle/input/**/*.parquet", recursive=True)
if not candidates:
    raise FileNotFoundError("не нашёл ни одного .parquet под /kaggle/input, см. листинг выше")
DATA_PATH = candidates[0]
print("используем:", DATA_PATH)

items = pd.read_parquet(DATA_PATH)
items["text"] = (items["item_title_raw"] + ". " + items["item_description_raw"]).apply(clean_text)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = SentenceTransformer("cointegrated/rubert-tiny2", device=device)
model.max_seq_length = 1024

embeddings = model.encode(
    items["text"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

np.save("/kaggle/working/embeddings_rubert_tiny2_1024.npy", embeddings)
np.save("/kaggle/working/embeddings_rubert_tiny2_1024_item_ids.npy", items["item_id"].to_numpy())

print("done, shape:", embeddings.shape)
