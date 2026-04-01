from __future__ import annotations

import numpy as np
import pandas as pd


def _sort_by_submit_order(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.copy()
    df_sorted["_sort_submit_date"] = pd.to_datetime(
        df_sorted["submit_date"], errors="coerce"
    ).fillna(pd.Timestamp.max)
    df_sorted["_sort_app_number"] = df_sorted.get("app_number", "").astype(str)
    df_sorted["_orig_index"] = df_sorted.index
    return df_sorted.sort_values(
        ["_sort_submit_date", "_sort_app_number", "_orig_index"],
        kind="mergesort",
    )


def _advisory_label(rate: float, count: int) -> str:
    if count < 5:
        return "недостаточно истории"
    if rate >= 0.75:
        return "история поддерживает"
    if rate <= 0.45:
        return "история предупреждает"
    return "история нейтральна"


def _advisory_note(source: str, count: int, rate: float) -> str:
    if source == "exact":
        scope = "точно такой же связке регион-направление-тип субсидии"
    elif source == "similar":
        scope = "похожей связке направление-тип субсидии"
    else:
        scope = "общей истории реестра"

    return (
        f"Историческая подсказка построена по {scope}: "
        f"{count} прошлых заявок, одобряемость {rate:.1%}."
    )


def build_history_advisory_batch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "history_match_source",
                "history_match_count",
                "history_approval_rate",
                "history_advisory_score",
                "history_recommendation",
                "history_note",
            ],
            index=df.index,
        )

    df_sorted = _sort_by_submit_order(df)

    exact_group = df_sorted.groupby(
        ["region", "direction", "subsidy_type"], sort=False
    )
    exact_count = exact_group.cumcount()
    exact_prev_approved = exact_group["is_approved"].cumsum() - df_sorted["is_approved"]
    exact_rate = (exact_prev_approved + 1) / (exact_count + 2)

    similar_group = df_sorted.groupby(["direction", "subsidy_type"], sort=False)
    similar_count = similar_group.cumcount()
    similar_prev_approved = (
        similar_group["is_approved"].cumsum() - df_sorted["is_approved"]
    )
    similar_rate = (similar_prev_approved + 1) / (similar_count + 2)

    global_count = pd.Series(np.arange(len(df_sorted)), index=df_sorted.index)
    global_prev_approved = (
        df_sorted["is_approved"].cumsum() - df_sorted["is_approved"]
    )
    global_rate = (global_prev_approved + 1) / (global_count + 2)

    use_exact = exact_count >= 5
    use_similar = (~use_exact) & (similar_count >= 10)

    source = np.where(use_exact, "exact", np.where(use_similar, "similar", "global"))
    count = np.where(use_exact, exact_count, np.where(use_similar, similar_count, global_count))
    rate = np.where(use_exact, exact_rate, np.where(use_similar, similar_rate, global_rate))

    advisory = pd.DataFrame(index=df_sorted.index)
    advisory["history_match_source"] = source
    advisory["history_match_count"] = count.astype(int)
    advisory["history_approval_rate"] = pd.Series(rate, index=df_sorted.index).round(4)
    advisory["history_advisory_score"] = (
        advisory["history_approval_rate"].astype(float) * 100
    ).round(1)
    advisory["history_recommendation"] = [
        _advisory_label(float(rate_value), int(count_value))
        for rate_value, count_value in zip(
            advisory["history_approval_rate"],
            advisory["history_match_count"],
        )
    ]
    advisory["history_note"] = [
        _advisory_note(str(source_value), int(count_value), float(rate_value))
        for source_value, count_value, rate_value in zip(
            advisory["history_match_source"],
            advisory["history_match_count"],
            advisory["history_approval_rate"],
        )
    ]

    result = advisory.set_index(df_sorted["_orig_index"])
    return result.reindex(df.index)


def build_history_advisory_single(
    row: dict | pd.Series,
    history_df: pd.DataFrame,
) -> dict[str, object]:
    row_series = row if isinstance(row, pd.Series) else pd.Series(row)
    request_row = pd.DataFrame(
        [
            {
                "app_number": "__request__",
                "region": str(row_series.get("region", "")).strip(),
                "direction": str(row_series.get("direction", "")).strip(),
                "subsidy_type": str(row_series.get("subsidy_type", "")).strip(),
                "submit_date": pd.to_datetime(
                    row_series.get("submit_date"), errors="coerce"
                ),
                "is_approved": 0,
            }
        ]
    )

    required_columns = request_row.columns.tolist()
    history = history_df.copy()
    for column in required_columns:
        if column not in history.columns:
            history[column] = pd.NA

    history["submit_date"] = pd.to_datetime(history["submit_date"], errors="coerce")
    request_date = request_row.loc[0, "submit_date"]
    if not pd.isna(request_date):
        history = history[history["submit_date"].isna() | (history["submit_date"] <= request_date)]

    combined = pd.concat([history[required_columns], request_row], ignore_index=True)
    advisory = build_history_advisory_batch(combined).iloc[-1]
    return {
        "history_match_source": str(advisory["history_match_source"]),
        "history_match_count": int(advisory["history_match_count"]),
        "history_approval_rate": float(advisory["history_approval_rate"]),
        "history_advisory_score": float(advisory["history_advisory_score"]),
        "history_recommendation": str(advisory["history_recommendation"]),
        "history_note": str(advisory["history_note"]),
    }
