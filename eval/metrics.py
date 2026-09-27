def recall_at_k(
    predictions: dict[int, list[str]],
    relevant: dict[int, set[str]],
    k: int = 50,
) -> float:
    scores = []
    for query_id, true_items in relevant.items():
        if not true_items:
            continue
        pred_items = set(predictions.get(query_id, [])[:k])
        scores.append(len(pred_items & true_items) / len(true_items))
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_50(predictions: dict[int, list[str]], relevant: dict[int, set[str]]) -> float:
    return recall_at_k(predictions, relevant, k=50)
