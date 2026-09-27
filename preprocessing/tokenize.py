import re

import pymorphy3

FORWARD_MAP = str.maketrans({
    "a": "а", "A": "А",
    "e": "е", "E": "Е",
    "o": "о", "O": "О",
    "p": "р", "P": "Р",
    "c": "с", "C": "С",
    "x": "х", "X": "Х",
    "y": "у", "Y": "У",
    "H": "Н",
    "K": "К",
    "M": "М",
    "T": "Т",
    "B": "В",
    "b": "в",
    "h": "н",
    "k": "к",
    "m": "м",
    "t": "т",
    "d": "д",
})

REVERSE_MAP = str.maketrans({v: k for k, v in {
    "a": "а", "A": "А", "e": "е", "E": "Е", "o": "о", "O": "О",
    "p": "р", "P": "Р", "c": "с", "C": "С", "x": "х", "X": "Х",
    "y": "у", "Y": "У", "H": "Н", "K": "К", "M": "М", "T": "Т", "B": "В",
    "b": "в", "h": "н", "k": "к", "m": "м", "t": "т", "d": "д",
}.items()})

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")
WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9]+")
CYR_RE = re.compile(r"[а-яёА-ЯЁ]")
LAT_RE = re.compile(r"[a-zA-Z]")
DIGIT_RE = re.compile(r"[0-9]")

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

_morph = pymorphy3.MorphAnalyzer()
_lemma_cache: dict[str, str] = {}
_is_known_cache: dict[str, bool] = {}


def _script_purity_score(s: str) -> int:
    n_cyr = len(CYR_RE.findall(s))
    n_lat = len(LAT_RE.findall(s))
    return min(n_cyr, n_lat)


def normalize_homoglyphs(token: str) -> str:
    if DIGIT_RE.search(token):
        return token.translate(REVERSE_MAP)
    forward = token.translate(FORWARD_MAP)
    reverse = token.translate(REVERSE_MAP)
    return min((forward, reverse), key=_script_purity_score)


def _is_known(word: str) -> bool:
    if word not in _is_known_cache:
        _is_known_cache[word] = _morph.parse(word)[0].is_known
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
        _lemma_cache[token] = _morph.parse(token)[0].normal_form
    return _lemma_cache[token]


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""

    def _fix(match: re.Match) -> str:
        word = match.group(0)
        if CYR_RE.search(word) and LAT_RE.search(word):
            return normalize_homoglyphs(word)
        return word

    return WORD_RE.sub(_fix, text)


def tokenize(text) -> list[str]:
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
