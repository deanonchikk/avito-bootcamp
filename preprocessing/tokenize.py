from functools import lru_cache
import re

import pymorphy3

from .text import (
    CYR_RE, DIGIT_RE, FORWARD_MAP, LAT_RE, REVERSE_MAP, WORD_RE,
    clean_text, normalize_homoglyphs,
)

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

# список стоп-слов nltk.corpus.stopwords.words("russian") (151 слово)
# захардкожен, чтобы не тащить nltk как зависимость
STOP_WORDS = {
    "а", "без", "более", "больше", "будет", "будто", "бы", "был", "была", "были",
    "было", "быть", "в", "вам", "вас", "вдруг", "ведь", "во", "вот", "впрочем",
    "все", "всегда", "всего", "всех", "всю", "вы", "где", "да", "даже", "два",
    "для", "до", "другой", "его", "ее", "ей", "ему", "если", "есть", "еще",
    "ж", "же", "за", "зачем", "здесь", "и", "из", "или", "им", "иногда", "их",
    "к", "как", "какая", "какой", "когда", "конечно", "кто", "куда", "ли",
    "лучше", "между", "меня", "мне", "много", "может", "можно", "мой", "моя",
    "мы", "на", "над", "надо", "наконец", "нас", "не", "него", "нее", "ней",
    "нельзя", "нет", "ни", "нибудь", "никогда", "ним", "них", "ничего", "но",
    "ну", "о", "об", "один", "он", "она", "они", "опять", "от", "перед", "по",
    "под", "после", "потом", "потому", "почти", "при", "про", "раз", "разве",
    "с", "сам", "свою", "себе", "себя", "сейчас", "со", "совсем", "так",
    "такой", "там", "тебя", "тем", "теперь", "то", "тогда", "того", "тоже",
    "только", "том", "тот", "три", "тут", "ты", "у", "уж", "уже", "хорошо",
    "хоть", "чего", "чем", "через", "что", "чтоб", "чтобы", "чуть", "эти",
    "этого", "этой", "этом", "этот", "эту", "я",
}

GLUED_MAX_LEN = 18
CYR_ONLY_RE = re.compile(r"^[а-яё]+$")

_lemma_cache: dict[str, str] = {}
_is_known_cache: dict[str, bool] = {}


@lru_cache(maxsize=1)
def _get_morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


def _is_known(word: str) -> bool:
    if word not in _is_known_cache:
        _is_known_cache[word] = _get_morph().parse(word)[0].is_known
    return _is_known_cache[word]


def segment_glued(word: str, max_len: int = GLUED_MAX_LEN, min_piece: int = 3) -> list[str]:
    if len(word) <= max_len or not CYR_ONLY_RE.match(word) or _is_known(word):
        return [word]

    result = []
    i, n = 0, len(word)
    while i < n:
        matched = False
        upper = min(n, i + 25)
        for j in range(upper, i + min_piece - 1, -1):
            candidate = word[i:j]
            if _is_known(candidate):
                result.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            result.append(word[i:i + min_piece])
            i += min_piece
    return result


def lemmatize(token: str) -> str:
    if token not in _lemma_cache:
        _lemma_cache[token] = _get_morph().parse(token)[0].normal_form
    return _lemma_cache[token]


def tokenize(text: object) -> list[str]:
    if not isinstance(text, str):
        return []
    raw_tokens = TOKEN_RE.findall(text.lower())

    normalized = [
        normalize_homoglyphs(t) if CYR_RE.search(t) and LAT_RE.search(t) else t
        for t in raw_tokens
    ]
    segmented = [piece for t in normalized for piece in segment_glued(t)]
    kept = [t for t in segmented if len(t) >= 2 and t not in STOP_WORDS]
    return [lemmatize(t) for t in kept]
