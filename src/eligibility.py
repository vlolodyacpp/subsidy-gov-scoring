from __future__ import annotations

import pandas as pd

from src.normatives import check_deadline_compliance, get_normative_for_type
from src.scoring import DEADLINE_DISQUALIFICATION_REASON


DECISION_SUPPORT_NOTE = (
    "Система формирует рекомендацию для комиссии и не заменяет "
    "установленный Правилами порядок рассмотрения заявки."
)

APPENDIX_2_MANUAL_REVIEW_NOTE = (
    "Автопроверка не покрывает все критерии приложения 2 к Правилам. "
    "Проверки через ГИСС, ИСЖ, ИБСПР, ЕАСУ, ИСЕГКН и приложенные документы "
    "требуют ручной или интеграционной верификации."
)

MISSING_NORMATIVE_NOTE = (
    "Для этого типа субсидии не найден эталонный норматив в локальном справочнике. "
    "Нужна ручная сверка с приложением 1 к Правилам."
)

INVALID_AMOUNT_REASON = "Некорректная сумма заявки"
MISSING_FIELDS_REASON = "Неполные обязательные поля заявки"


def _coerce_series(row: dict | pd.Series) -> pd.Series:
    if isinstance(row, pd.Series):
        return row
    return pd.Series(row)


def evaluate_single_eligibility(
    row: dict | pd.Series,
    normative_lookup: dict[str, int] | None = None,
) -> dict[str, object]:
    row_series = _coerce_series(row)
    submit_date = row_series.get("submit_date")
    deadline_value = row_series.get("deadline_compliance")
    if pd.isna(deadline_value):
        deadline_value = check_deadline_compliance(submit_date)

    amount = pd.to_numeric(pd.Series([row_series.get("amount")]), errors="coerce").iloc[0]
    critical_fields = ["region", "direction", "subsidy_type"]
    missing_fields = [
        field
        for field in critical_fields
        if str(row_series.get(field, "")).strip() in {"", "None", "nan"}
    ]

    disqualified = False
    disqualification_reason = None
    if pd.notna(deadline_value) and float(deadline_value) <= 0:
        disqualified = True
        disqualification_reason = DEADLINE_DISQUALIFICATION_REASON
    elif pd.isna(amount) or float(amount) <= 0:
        disqualified = True
        disqualification_reason = INVALID_AMOUNT_REASON
    elif missing_fields:
        disqualified = True
        disqualification_reason = f"{MISSING_FIELDS_REASON}: {', '.join(missing_fields)}"

    normative_reference_found = True
    subsidy_type = str(row_series.get("subsidy_type", "")).strip()
    if subsidy_type and normative_lookup is not None:
        normative_reference_found = get_normative_for_type(subsidy_type, normative_lookup) is not None

    manual_review_required = not disqualified
    notes: list[str] = []
    if manual_review_required:
        notes.append(APPENDIX_2_MANUAL_REVIEW_NOTE)
        if not normative_reference_found:
            notes.append(MISSING_NORMATIVE_NOTE)

    return {
        "disqualified": bool(disqualified),
        "disqualification_reason": disqualification_reason,
        "eligibility_status": "failed" if disqualified else "preliminarily_eligible",
        "manual_review_required": bool(manual_review_required),
        "eligibility_note": " ".join(notes) if notes else None,
        "normative_reference_found": bool(normative_reference_found),
    }


def evaluate_batch_eligibility(
    df: pd.DataFrame,
    normative_lookup: dict[str, int] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "disqualified",
                "disqualification_reason",
                "eligibility_status",
                "manual_review_required",
                "eligibility_note",
                "normative_reference_found",
            ],
            index=df.index,
        )

    submit_dates = pd.to_datetime(df.get("submit_date"), errors="coerce")
    if "deadline_compliance" in df.columns:
        deadline_values = pd.to_numeric(df["deadline_compliance"], errors="coerce")
    else:
        deadline_values = submit_dates.apply(check_deadline_compliance)

    amounts = pd.to_numeric(df.get("amount"), errors="coerce")
    required_fields = df.reindex(columns=["region", "direction", "subsidy_type"]).fillna("")
    missing_fields_mask = (
        required_fields.astype(str).apply(lambda col: col.str.strip().eq("") | col.eq("None"))
    ).any(axis=1)
    invalid_amount_mask = amounts.isna() | (amounts <= 0)
    deadline_mask = deadline_values.fillna(0.5) <= 0

    disqualified = deadline_mask | invalid_amount_mask | missing_fields_mask
    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    reasons.loc[deadline_mask] = DEADLINE_DISQUALIFICATION_REASON
    reasons.loc[~deadline_mask & invalid_amount_mask] = INVALID_AMOUNT_REASON
    reasons.loc[~deadline_mask & ~invalid_amount_mask & missing_fields_mask] = MISSING_FIELDS_REASON

    if normative_lookup is not None and "subsidy_type" in df.columns:
        normative_reference_found = (
            df["subsidy_type"]
            .astype(str)
            .str.strip()
            .map(lambda subsidy_type: get_normative_for_type(subsidy_type, normative_lookup) is not None)
            .fillna(False)
        )
    else:
        normative_reference_found = pd.Series(True, index=df.index, dtype=bool)

    manual_review_required = ~disqualified
    eligibility_note = pd.Series(pd.NA, index=df.index, dtype="object")
    eligibility_note.loc[manual_review_required] = APPENDIX_2_MANUAL_REVIEW_NOTE
    eligibility_note.loc[
        manual_review_required & ~normative_reference_found
    ] = (
        APPENDIX_2_MANUAL_REVIEW_NOTE
        + " "
        + MISSING_NORMATIVE_NOTE
    )

    return pd.DataFrame(
        {
            "disqualified": disqualified.astype(bool),
            "disqualification_reason": reasons,
            "eligibility_status": pd.Series(
                ["failed" if flag else "preliminarily_eligible" for flag in disqualified],
                index=df.index,
                dtype="object",
            ),
            "manual_review_required": manual_review_required.astype(bool),
            "eligibility_note": eligibility_note,
            "normative_reference_found": normative_reference_found.astype(bool),
        },
        index=df.index,
    )
