from .params import extract_semantic_params
from .text import as_text, clean_text
from .tokenize import tokenize

CHAR_MAX_DESC = 200
CHAR_MAX_PARAMS = 100


def build_bm25_text(title: object, description: object = "", params: object = "") -> str:
    return " ".join(as_text(field) for field in (title, description, params))


def build_bm25_tokens(
    title: object,
    description: object = "",
    params: object = "",
    *,
    weights: tuple[int, int, int] = (1, 1, 1),
) -> list[str]:
    if len(weights) != 3 or any(type(weight) is not int or weight < 0 for weight in weights):
        raise ValueError("Нужны три целых неотрицательных веса: title, description, params")
    result = []
    for field, weight in zip((title, description, params), weights):
        if weight:
            result.extend(tokenize(field) * weight)
    return result


def build_char_item_text(
    title: object,
    description: object = "",
    params: object = "",
    *,
    max_desc: int = CHAR_MAX_DESC,
    max_params: int = CHAR_MAX_PARAMS,
) -> str:
    if max_desc < 0 or max_params < 0:
        raise ValueError("Длины должны быть неотрицательными")
    return as_text(title) + ". " + as_text(description)[:max_desc] + " " + as_text(params)[:max_params]


def build_e5_item_text(title: object, description: object = "", params: object = "") -> str:
    text = as_text(title) + ". " + as_text(description)
    extracted = extract_semantic_params(params)
    if extracted:
        text += ". " + extracted
    return "passage: " + clean_text(text)


def build_e5_query_text(query: object) -> str:
    return "query: " + clean_text(query)
