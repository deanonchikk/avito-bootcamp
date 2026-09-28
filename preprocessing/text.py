import re

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


def as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def normalize_query(value: object) -> str:
    return as_text(value).lower().strip()


def _script_purity_score(text: str) -> int:
    return min(len(CYR_RE.findall(text)), len(LAT_RE.findall(text)))


def normalize_homoglyphs(token: str) -> str:
    if DIGIT_RE.search(token):
        return token.translate(REVERSE_MAP)
    forward = token.translate(FORWARD_MAP)
    reverse = token.translate(REVERSE_MAP)
    return min((forward, reverse), key=_script_purity_score)


def clean_text(value: object) -> str:
    text = as_text(value)

    def fix(match: re.Match) -> str:
        word = match.group(0)
        if CYR_RE.search(word) and LAT_RE.search(word):
            return normalize_homoglyphs(word)
        return word

    return WORD_RE.sub(fix, text)
