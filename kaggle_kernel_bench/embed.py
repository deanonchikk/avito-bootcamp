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


TEXT_LABELS = [
    "Название услуги", "Специальность", "Куда выезжаете", "Ваши клиенты",
    "Кто оказывает услуги", "Где вы оказываете услуги", "Занятия",
    "Вид услуги", "Тип услуги", "Услуга", "Чем вы занимаетесь",
    "Марка авто", "Модель авто", "Тип автосервиса", "Тип обслуживаемой техники",
    "Где снимаете", "Дополнительно", "Как вы работаете",
]

FLAG_LABELS = [
    "Работа по договору", "Гарантия на выполнение работ", "Гарантия на работу", "Гарантия",
    "Готов закупить материалы", "Работаете с юрлицами и ИП", "Берёте ли срочные заказы",
    "Предоплата", "Работа в праздники и выходные", "Выезд",
]

NOISE_LABELS = [
    "Место оказания услуг", "Тип стоимости за услугу", "Тип стоимости за час", "Тип стоимости за м²",
    "Тип стоимости за единицу", "Начальная цена", "Опыт работы",
    "График работы, дни недели", "График работы от", "График работы до",
    "Время работы, с", "Время работы, до",
    "Время для связи, дни недели", "Время для связи от", "Время для связи до",
    "Время для связи, с", "Время для связи, до",
    "Стоимость", "Продолжительность", "Учебное учреждение", "Год окончания",
    "Бригада", "Выполняю заказы", "Рабочие дни",
    "Дни пн", "Дни вт", "Дни ср", "Дни чт", "Дни пт", "Дни сб", "Дни вс",
    "Срочная услуга (мультистатус)", "Рейтинг пользователя",
    "Минимальная сумма заказа", "Минимальное время аренды", "Минимальное время заказа", "Минимальное время",
    "Признак предзаполнения прайс листа", "Признак мигрированного прайс листа", "Признак",
    "Можно со своими запчастями", "Камера наблюдения в ремонтной зоне", "Камера наблюдения",
]

_ALL_LABELS = sorted(set(TEXT_LABELS + FLAG_LABELS + NOISE_LABELS), key=len, reverse=True)
_PARAMS_PATTERN = re.compile("(" + "|".join(re.escape(l) for l in _ALL_LABELS) + ")")


def extract_semantic_params(text):
    if not text:
        return ""
    parts = _PARAMS_PATTERN.split(text)
    out = []
    seen_flags = set()
    for i in range(1, len(parts), 2):
        label = parts[i]
        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if label in TEXT_LABELS:
            if value and value != "Своя услуга":
                out.append(value)
        elif label in FLAG_LABELS and label not in seen_flags:
            out.append(label)
            seen_flags.add(label)
    result = []
    for v in out:
        if not result or result[-1] != v:
            result.append(v)
    return " ".join(result)


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


def build_item_text(row):
    parts = [row["item_title_raw"], ". ", row["item_description_raw"]]
    extracted = extract_semantic_params(row.get("item_infm_params_text", ""))
    if extracted:
        parts.append(". " + extracted)
    return clean_text("".join(parts))


items["text"] = "passage: " + items.apply(build_item_text, axis=1)

n_gpus = torch.cuda.device_count()
print("GPUs available:", n_gpus)

model = SentenceTransformer("intfloat/multilingual-e5-base")
model.max_seq_length = 512

if n_gpus >= 1:
    model.half()

device_arg = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus >= 2 else ("cuda" if n_gpus == 1 else "cpu")

texts = items["text"].tolist()
CHUNK_SIZE = 10000
n_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
chunks = []
for i in range(n_chunks):
    chunk_texts = texts[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
    chunk_emb = model.encode(
        chunk_texts,
        batch_size=256,
        device=device_arg,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    chunks.append(chunk_emb)
    print(f"chunk {i + 1}/{n_chunks} готов ({(i + 1) * CHUNK_SIZE if i < n_chunks - 1 else len(texts)}/{len(texts)})", flush=True)

embeddings = np.concatenate(chunks)

embeddings = embeddings.astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

np.save("/kaggle/working/embeddings_e5_base_512.npy", embeddings)
np.save("/kaggle/working/embeddings_e5_base_512_item_ids.npy", items["item_id"].to_numpy())

print("done, shape:", embeddings.shape)
