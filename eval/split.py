import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SEARCH_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def make_val_split(
    train: pd.DataFrame,
    search_cols: list[str] = DEFAULT_SEARCH_COLS,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, set[str]]]:
    context = train[search_cols].astype(str).agg("||".join, axis=1)
    unique_contexts = context.drop_duplicates()

    items_per_context = train.groupby(context)["item_id"].nunique()
    is_multi = (items_per_context.loc[unique_contexts] > 1).to_numpy()

    train_ctx, val_ctx = train_test_split(
        unique_contexts,
        test_size=val_size,
        random_state=random_state,
        stratify=is_multi,
    )

    train_part = train[context.isin(train_ctx)].reset_index(drop=True)

    val_ctx_set = set(val_ctx)
    first_occurrence_idx = context[context.isin(val_ctx_set)].drop_duplicates().index
    val_queries = train.loc[first_occurrence_idx, search_cols].reset_index(drop=True)
    val_queries["context_id"] = np.arange(len(val_queries))

    val_context_str = val_queries[search_cols].astype(str).agg("||".join, axis=1)
    relevant_by_context = train.loc[context.isin(val_ctx_set)].groupby(
        context, sort=False
    )["item_id"].agg(set)
    val_relevant = {
        context_id: relevant_by_context.loc[ctx_str]
        for context_id, ctx_str in zip(val_queries["context_id"], val_context_str)
    }

    return train_part, val_queries, val_relevant
