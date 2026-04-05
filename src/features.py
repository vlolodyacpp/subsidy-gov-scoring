import pandas as pd
import numpy as np
import os
from functools import lru_cache

from src.normatives import (
    build_normative_lookup,
    check_deadline_compliance,
    get_normative_for_type,
)

SYNTHETIC_FEATURES_PATH = "data/cleaned/synthetic_features.csv"
PASTURE_NORMS_PATH = "data/cleaned/pasture_norms.csv"
MORTALITY_NORMS_PATH = "data/cleaned/mortality_norms.csv"
ELIGIBILITY_CRITERIA_PATH = "data/cleaned/eligibility_criteria.csv"

LEAKY_CONDITION_FEATURE_COLUMNS = [
    "pasture_compliance",
    "mortality_compliance",
    "grazing_utilization",
]
SAFE_CONDITION_FEATURE_COLUMNS = [
    "criteria_complexity",
    "direction_risk",
    "regional_pasture_capacity",
]
CONDITION_REFERENCE_COLUMNS = [
    "pasture_norm",
    "grazing_days",
    "mortality_mean",
    "mortality_max",
    "avg_criteria_count",
]

# новые признаки из cond-файлов (все [0, 1])
CONDITION_FEATURE_COLUMNS = [
    *LEAKY_CONDITION_FEATURE_COLUMNS,
    *SAFE_CONDITION_FEATURE_COLUMNS,
]

REGION_MAP = {
    "Акмолинская область": "Акмолинская",
    "Актюбинская область": "Актюбинская",
    "Алматинская область": "Алматинская",
    "Атырауская область": "Атырауская",
    "Восточно-Казахстанская область": "Восточно-Казахстанская",
    "Жамбылская область": "Жамбылская",
    "Западно-Казахстанская область": "Западно-Казахстанская",
    "Карагандинская область": "Карагандинская",
    "Костанайская область": "Костанайская",
    "Кызылординская область": "Кызылординская",
    "Мангистауская область": "Мангистауская",
    "Павлодарская область": "Павлодарская",
    "Северо-Казахстанская область": "Северо-Казахстанская",
    "Туркестанская область": "Туркестанская",
    "г.Шымкент": "Туркестанская",
    "область Абай": "Абай",
    "область Жетісу": "Жетісу",
    "область Ұлытау": "Ұлытау",
}

DIRECTION_MORTALITY_MAP = {
    "Субсидирование в скотоводстве": "Мясное и молочное скотоводство",
    "Субсидирование в овцеводстве": "Овцеводство и козоводство",
    "Субсидирование в козоводстве": "Овцеводство и козоводство",
    "Субсидирование в коневодстве": "Коневодство",
    "Субсидирование в верблюдоводстве": "Верблюдоводство",
    "Субсидирование в свиноводстве": "Свиноводство",
    "Субсидирование в птицеводстве": "Птицеводство",
    "Субсидирование в пчеловодстве": "Пчеловодство",
    "Субсидирование затрат по искусственному осеменению": "Мясное и молочное скотоводство",
}

DIRECTION_PASTURE_ANIMAL = {
    "Субсидирование в скотоводстве": "cattle",
    "Субсидирование в овцеводстве": "sheep_goats",
    "Субсидирование в козоводстве": "sheep_goats",
    "Субсидирование в коневодстве": "horses",
    "Субсидирование в верблюдоводстве": "camels",
    "Субсидирование в свиноводстве": "cattle",
    "Субсидирование в птицеводстве": "cattle",
    "Субсидирование в пчеловодстве": "cattle",
    "Субсидирование затрат по искусственному осеменению": "cattle",
}

DIRECTION_CRITERIA_MAP = {
    "Мясное и мясо-молочное скотоводство": "Субсидирование в скотоводстве",
    "Молочное и молочно-мясное скотоводство": "Субсидирование в скотоводстве",
    "Скотоводство": "Субсидирование в скотоводстве",
    "Овцеводство": "Субсидирование в овцеводстве",
    "Коневодство": "Субсидирование в коневодстве",
    "Верблюдоводство": "Субсидирование в верблюдоводстве",
    "Свиноводство": "Субсидирование в свиноводстве",
    "Мясное птицеводство": "Субсидирование в птицеводстве",
    "Яичное птицеводство": "Субсидирование в птицеводстве",
}


def _sort_by_submit_order(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.copy()
    df_sorted["_sort_submit_date"] = pd.to_datetime(
        df_sorted["submit_date"], errors="coerce"
    ).fillna(pd.Timestamp.max)
    df_sorted["_sort_app_number"] = df_sorted["app_number"].astype(str)
    df_sorted["_orig_index"] = df_sorted.index
    return df_sorted.sort_values(
        ["_sort_submit_date", "_sort_app_number", "_orig_index"],
        kind="mergesort",
    )


def _single_temporal_features(submit_date, submit_month: int | None = None) -> dict[str, float]:
    if pd.isna(submit_date):
        month = submit_month if submit_month is not None else 6
        angle = 2 * np.pi * (month - 1) / 12
        return {
            "submit_month_sin": float(np.sin(angle)),
            "submit_month_cos": float(np.cos(angle)),
        }

    month = int(getattr(submit_date, "month", submit_month or 6))
    angle = 2 * np.pi * (month - 1) / 12

    return {
        "submit_month_sin": float(np.sin(angle)),
        "submit_month_cos": float(np.cos(angle)),
    }


def _ratio_to_typicality(
    ratio_like: pd.Series | np.ndarray,
    neutral_value: float = 0.5,
) -> pd.Series:
    ratio_series = pd.to_numeric(
        pd.Series(ratio_like),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    ratio_series = ratio_series.clip(lower=1e-3)
    log_distance = np.abs(np.log(ratio_series))
    max_distance = float(np.log(4))
    typicality = 1 - np.minimum(log_distance, max_distance) / max_distance
    return typicality.fillna(neutral_value).astype(float)


def compute_approval_rates(df: pd.DataFrame) -> dict:
    # вычисление процентов одобрений по направлениям, типам субсидий, акиматам, и кросс-метрик
    rates = {}

    for col in ["direction", "subsidy_type", "akimat", "region"]:
        group = df.groupby(col)["is_approved"].agg(["mean", "count"])
        group.columns = ["approval_rate", "total_apps"]
        rates[col] = group.to_dict("index")

    # кросс-метрика: (region, direction) → approval_rate с лапласовым сглаживанием
    cross = df.groupby(["region", "direction"])["is_approved"].agg(["sum", "count"])
    cross.columns = ["n_approved", "n_total"]
    cross["smoothed_rate"] = (cross["n_approved"] + 1) / (cross["n_total"] + 2)
    rates["region_direction"] = {
        (r, d): {"approval_rate": row["smoothed_rate"], "total_apps": int(row["n_total"])}
        for (r, d), row in cross.iterrows()
    }

    return rates


def compute_amount_stats(df: pd.DataFrame) -> dict:
    # медиана и стандартное отклонение суммы по типу субсидии
    stats = {}
    for stype, group in df.groupby("subsidy_type"):
        amounts = group["amount"]
        stats[stype] = {
            "median": amounts.median(),
            "std": amounts.std() if len(amounts) > 1 else amounts.median(),
            "mean": amounts.mean(),
            "q25": amounts.quantile(0.25),
            "q75": amounts.quantile(0.75),
        }
    return stats


def compute_region_specialization(df: pd.DataFrame) -> dict:
    # доля направления в регионе
    region_direction_counts = df.groupby(["region", "direction"]).size()
    region_totals = df.groupby("region").size()

    spec = {}
    for (region, direction), count in region_direction_counts.items():
        total = region_totals[region]
        spec[(region, direction)] = count / total if total > 0 else 0

    return spec


def compute_historical_approval_rate(
    df: pd.DataFrame,
    group_cols: str | list[str],
    default_rate: float = 0.5,
) -> pd.Series:
    df_sorted = _sort_by_submit_order(df)
    grouped = df_sorted.groupby(group_cols, sort=False)
    prev_approved = grouped["is_approved"].cumsum() - df_sorted["is_approved"]
    prev_count = grouped.cumcount()
    rates = np.where(
        prev_count > 0,
        (prev_approved + 1) / (prev_count + 2),
        default_rate,
    )
    result = pd.Series(rates, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def compute_historical_region_specialization(df: pd.DataFrame) -> pd.Series:
    df_sorted = _sort_by_submit_order(df)
    prev_region_total = df_sorted.groupby("region", sort=False).cumcount()
    prev_region_direction = df_sorted.groupby(
        ["region", "direction"], sort=False
    ).cumcount()

    specialization = np.where(
        prev_region_total > 0,
        (prev_region_direction + 1) / (prev_region_total + 2),
        0.5,
    )
    result = pd.Series(specialization, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def compute_historical_amount_adequacy(df: pd.DataFrame) -> pd.Series:
    df_sorted = _sort_by_submit_order(df)
    grouped = df_sorted.groupby("subsidy_type", sort=False)["amount"]

    hist_median = grouped.transform(lambda s: s.expanding().median().shift(1))
    hist_std = grouped.transform(lambda s: s.expanding().std().shift(1))
    global_hist_median = df_sorted["amount"].expanding().median().shift(1)
    global_hist_std = df_sorted["amount"].expanding().std().shift(1)

    hist_median = hist_median.fillna(global_hist_median).fillna(df_sorted["amount"])
    hist_std = hist_std.fillna(global_hist_std)
    hist_std = hist_std.replace(0, np.nan).fillna(hist_median.abs().replace(0, np.nan))
    hist_std = hist_std.fillna(df_sorted["amount"].abs().replace(0, np.nan))
    hist_std = hist_std.fillna(1.0)

    z_scores = (df_sorted["amount"] - hist_median).abs() / hist_std
    adequacy = (1 - z_scores.clip(0, 3) / 3).clip(0, 1)
    result = pd.Series(adequacy, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def compute_historical_amount_to_median_ratio(df: pd.DataFrame) -> pd.Series:
    df_sorted = _sort_by_submit_order(df)
    grouped = df_sorted.groupby("subsidy_type", sort=False)["amount"]

    hist_median = grouped.transform(lambda s: s.expanding().median().shift(1))
    global_hist_median = df_sorted["amount"].expanding().median().shift(1)
    hist_median = hist_median.fillna(global_hist_median).replace(0, np.nan)
    hist_median = hist_median.fillna(df_sorted["amount"]).replace(0, 1.0)

    ratio = (df_sorted["amount"] / hist_median).clip(0, 5)
    result = pd.Series(ratio, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def compute_historical_group_count(
    df: pd.DataFrame,
    group_cols: str | list[str],
) -> pd.Series:
    df_sorted = _sort_by_submit_order(df)
    counts = df_sorted.groupby(group_cols, sort=False).cumcount().astype(float)
    result = pd.Series(counts, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def compute_budget_pressure(df: pd.DataFrame) -> pd.Series:
    # бюджетное давление без утечки в будущее.
    # Идея: какая доля запрошенных средств одобрялась в этой группе до сих пор?
    # fulfillment_rate = prev_approved_amount / prev_requested_amount
    #   ≈ 1.0 → почти всё одобрялось → давление низкое (score высокий)
    #   ≈ 0.5 → половина отклоняется → давление среднее
    #   → 0.0 → почти ничего не одобрялось → давление высокое (score низкий)
    df_sorted = _sort_by_submit_order(df)
    grouped = df_sorted.groupby(["region", "direction"], sort=False)

    df_sorted["_approved_amount"] = df_sorted["amount"] * df_sorted["is_approved"]
    prev_approved = grouped["_approved_amount"].cumsum() - df_sorted["_approved_amount"]
    prev_requested = grouped["amount"].cumsum() - df_sorted["amount"]

    pressure = np.where(
        prev_requested > 0,
        (prev_approved / prev_requested).clip(0, 1),
        0.5,
    )

    result = pd.Series(pressure, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def compute_queue_position(df: pd.DataFrame) -> pd.Series:
    # позиция в очереди подачи без знания будущего числа заявок в группе.
    # Логарифмическое затухание: ранние заявки получают больший score,
    # но затухание плавнее чем 1/sqrt, давая различимые значения даже
    # для поздних заявок в крупных группах.
    df_sorted = _sort_by_submit_order(df)
    prior_count = df_sorted.groupby(
        ["region", "direction", "subsidy_type"], sort=False
    ).cumcount()

    position = 1 / (1 + np.log1p(prior_count))
    result = pd.Series(position, index=df_sorted["_orig_index"])
    return result.reindex(df.index).astype(float)


def _count_criteria(text: str) -> int:
    if pd.isna(text) or not text:
        return 0
    return int(
        pd.Series([str(text)])
        .str.count(r"(?:^|\s)\d+\.\s")
        .fillna(0)
        .astype(int)
        .iloc[0]
    )


@lru_cache(maxsize=1)
def load_condition_context_tables() -> dict[str, object] | None:
    if not (
        os.path.exists(PASTURE_NORMS_PATH)
        and os.path.exists(MORTALITY_NORMS_PATH)
        and os.path.exists(ELIGIBILITY_CRITERIA_PATH)
    ):
        return None

    pasture = pd.read_csv(PASTURE_NORMS_PATH)
    pasture_avg = (
        pasture.groupby("oblast")
        .agg(
            cattle_norm=("cattle_restored", "mean"),
            sheep_goats_norm=("sheep_goats_restored", "mean"),
            horses_norm=("horses_restored", "mean"),
            camels_norm=("camels_restored", "mean"),
            grazing_days=("grazing_period_days", "mean"),
        )
        .reset_index()
    )

    mortality = pd.read_csv(MORTALITY_NORMS_PATH)
    mortality_agg = (
        mortality.groupby("direction")
        .agg(
            mortality_mean=("mortality_pct", "mean"),
            mortality_max=("mortality_pct", "max"),
        )
        .reset_index()
    )

    criteria = pd.read_csv(ELIGIBILITY_CRITERIA_PATH)
    criteria["criteria_count"] = criteria["criteria"].apply(_count_criteria)
    criteria["main_direction"] = criteria["direction"].map(DIRECTION_CRITERIA_MAP)
    criteria_by_direction = (
        criteria.groupby("main_direction")["criteria_count"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "main_direction": "direction",
                "criteria_count": "avg_criteria_count",
            }
        )
    )

    pasture_norm_values = []
    for animal in ["cattle", "sheep_goats", "horses", "camels"]:
        column_name = f"{animal}_norm"
        if column_name in pasture_avg.columns:
            pasture_norm_values.extend(pasture_avg[column_name].dropna().tolist())

    pasture_norm_median = float(np.median(pasture_norm_values)) if pasture_norm_values else 100.0
    grazing_days_median = float(pasture_avg["grazing_days"].median()) if "grazing_days" in pasture_avg.columns else 200.0
    mortality_mean_median = float(mortality_agg["mortality_mean"].median()) if "mortality_mean" in mortality_agg.columns else 30.0
    mortality_max_median = float(mortality_agg["mortality_max"].median()) if "mortality_max" in mortality_agg.columns else 80.0
    criteria_count_median = float(criteria_by_direction["avg_criteria_count"].median()) if "avg_criteria_count" in criteria_by_direction.columns else 5.0

    return {
        "pasture_avg": pasture_avg,
        "mortality_agg": mortality_agg,
        "criteria_by_direction": criteria_by_direction,
        "defaults": {
            "pasture_norm": pasture_norm_median,
            "grazing_days": grazing_days_median,
            "mortality_mean": mortality_mean_median,
            "mortality_max": mortality_max_median,
            "avg_criteria_count": criteria_count_median,
        },
        "max_values": {
            "pasture_norm": max(float(max(pasture_norm_values)), pasture_norm_median) if pasture_norm_values else pasture_norm_median,
            "mortality_mean": max(float(mortality_agg["mortality_mean"].max()), mortality_mean_median) if not mortality_agg.empty else mortality_mean_median,
            "avg_criteria_count": max(float(criteria_by_direction["avg_criteria_count"].max()), criteria_count_median) if not criteria_by_direction.empty else criteria_count_median,
        },
    }


def build_condition_context_features(df: pd.DataFrame) -> pd.DataFrame:
    tables = load_condition_context_tables()
    context = pd.DataFrame(index=df.index)
    if tables is None:
        for column_name in CONDITION_REFERENCE_COLUMNS:
            context[column_name] = np.nan
        for column_name in SAFE_CONDITION_FEATURE_COLUMNS:
            context[column_name] = 0.5
        return context

    pasture_avg = tables["pasture_avg"]
    mortality_agg = tables["mortality_agg"]
    criteria_by_direction = tables["criteria_by_direction"]
    defaults = tables["defaults"]
    max_values = tables["max_values"]

    working = df[["region", "direction"]].copy()
    working["oblast_key"] = working["region"].map(REGION_MAP)
    working["pasture_animal"] = working["direction"].map(DIRECTION_PASTURE_ANIMAL)
    working = working.merge(
        pasture_avg,
        left_on="oblast_key",
        right_on="oblast",
        how="left",
    )
    working["pasture_norm"] = np.nan
    for animal in ["cattle", "sheep_goats", "horses", "camels"]:
        mask = working["pasture_animal"] == animal
        if not mask.any():
            continue
        working.loc[mask, "pasture_norm"] = working.loc[mask, f"{animal}_norm"]

    working["pasture_norm"] = pd.to_numeric(working["pasture_norm"], errors="coerce").fillna(defaults["pasture_norm"])
    working["grazing_days"] = pd.to_numeric(working["grazing_days"], errors="coerce").fillna(defaults["grazing_days"])

    working["mortality_direction"] = working["direction"].map(DIRECTION_MORTALITY_MAP)
    working = working.merge(
        mortality_agg,
        left_on="mortality_direction",
        right_on="direction",
        how="left",
        suffixes=("", "_mort"),
    )
    working["mortality_mean"] = pd.to_numeric(working["mortality_mean"], errors="coerce").fillna(defaults["mortality_mean"])
    working["mortality_max"] = pd.to_numeric(working["mortality_max"], errors="coerce").fillna(defaults["mortality_max"])

    working = working.merge(
        criteria_by_direction,
        left_on="direction",
        right_on="direction",
        how="left",
    )
    working["avg_criteria_count"] = (
        pd.to_numeric(working["avg_criteria_count"], errors="coerce")
        .fillna(defaults["avg_criteria_count"])
    )

    context["pasture_norm"] = working["pasture_norm"].astype(float).to_numpy()
    context["grazing_days"] = working["grazing_days"].astype(float).to_numpy()
    context["mortality_mean"] = working["mortality_mean"].astype(float).to_numpy()
    context["mortality_max"] = working["mortality_max"].astype(float).to_numpy()
    context["avg_criteria_count"] = working["avg_criteria_count"].astype(float).to_numpy()

    max_criteria = max(float(max_values["avg_criteria_count"]), 1.0)
    max_mortality = max(float(max_values["mortality_mean"]), 0.1)
    max_pasture_norm = max(float(max_values["pasture_norm"]), 1.0)

    context["criteria_complexity"] = (
        1.0 - (context["avg_criteria_count"] / max_criteria)
    ).clip(0.0, 1.0).fillna(0.5)
    context["direction_risk"] = (
        1.0 - (context["mortality_mean"] / max_mortality)
    ).clip(0.0, 1.0).fillna(0.5)
    context["regional_pasture_capacity"] = (
        1.0 - (context["pasture_norm"] / max_pasture_norm)
    ).clip(0.0, 1.0).fillna(0.5)
    return context


def load_condition_features(
    path: str = SYNTHETIC_FEATURES_PATH,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Загружает condition-признаки из cleaned CSV для offline-экспериментов."""
    if not os.path.exists(path):
        return None
    cond = pd.read_csv(path)
    requested_columns = columns or CONDITION_FEATURE_COLUMNS
    keep = ["app_number"] + [c for c in requested_columns if c in cond.columns]
    return cond[keep]


def build_feature_tables(df: pd.DataFrame) -> dict:
    # сборка справочных таблиц для скоринга
    norm_lookup = build_normative_lookup()

    # предрассчитанные контекстные данные для single scoring
    # budget: одобренная сумма по (region, direction)
    approved_totals = (
        df[df["is_approved"] == 1]
        .groupby(["region", "direction"])["amount"]
        .sum()
        .to_dict()
    )
    # budget: кумулятивная сумма всех запросов по (region, direction)
    df_sorted = df.sort_values("submit_date")
    cum_requested = df_sorted.groupby(["region", "direction"])["amount"].cumsum()
    # для каждой группы: последнее значение = общий запрос
    total_requested = (
        df.groupby(["region", "direction"])["amount"]
        .sum()
        .to_dict()
    )

    # queue: размер каждой группы (region, direction, subsidy_type)
    queue_group_sizes = (
        df.groupby(["region", "direction", "subsidy_type"])
        .size()
        .to_dict()
    )

    # unit_count: значения по subsidy_type для перцентильного ранга
    effective_norms = df["subsidy_type"].map(
        lambda st: get_normative_for_type(st, norm_lookup) or 0
    )
    effective_norms = np.where(effective_norms > 0, effective_norms, df["normative"])
    effective_norms = pd.Series(effective_norms, index=df.index).replace(0, np.nan)
    unit_count_raw = df["amount"] / effective_norms.fillna(1)
    unit_count_by_type = {}
    for stype, group in unit_count_raw.groupby(df["subsidy_type"]):
        unit_count_by_type[stype] = sorted(group.dropna().tolist())

    return {
        "approval_rates": compute_approval_rates(df),
        "amount_stats": compute_amount_stats(df),
        "region_specialization": compute_region_specialization(df),
        "normative_lookup": norm_lookup,
        "approved_totals": approved_totals,
        "total_requested": total_requested,
        "queue_group_sizes": queue_group_sizes,
        "unit_count_by_type": unit_count_by_type,
    }


def extract_features(row: pd.Series, tables: dict) -> dict:
    # извлечение признаков из одной заявки
    ar = tables["approval_rates"]
    amt = tables["amount_stats"]
    spec = tables["region_specialization"]
    norm_lookup = tables["normative_lookup"]

    # группа 1: нормативное соответствие

    row_norm = pd.to_numeric(
        pd.Series([row.get("normative", 0)]),
        errors="coerce",
    ).fillna(0.0).iloc[0]
    original_norm = pd.to_numeric(
        pd.Series([row.get("normative_original", row_norm)]),
        errors="coerce",
    ).fillna(row_norm).iloc[0]

    # normative_match
    ref_norm = get_normative_for_type(row["subsidy_type"], norm_lookup)
    if ref_norm is None:
        normative_match = 0.5
    elif row_norm == ref_norm:
        normative_match = 1.0
    elif ref_norm > 0 and abs(row_norm - ref_norm) / ref_norm < 0.05:
        normative_match = 0.8
    else:
        normative_match = 0.0

    if ref_norm is None:
        normative_original_match = 0.5
        normative_reference_gap = 1.0
        normative_reference_typicality = 0.5
    else:
        original_deviation = abs(original_norm - ref_norm) / ref_norm if ref_norm > 0 else 1.0
        normative_original_match = (
            1.0
            if original_deviation == 0
            else 0.8
            if original_deviation < 0.05
            else 0.0
        )
        normative_reference_gap = float(np.clip(original_deviation, 0, 3))
        normative_reference_typicality = float(
            _ratio_to_typicality(
                [original_norm / ref_norm if ref_norm > 0 else 1.0],
                neutral_value=1.0,
            ).iloc[0]
        )

    # amount_normative_integrity
    effective_norm = ref_norm if ref_norm else row_norm
    if effective_norm and effective_norm > 0:
        unit_count_val = row["amount"] / effective_norm
        remainder = unit_count_val - round(unit_count_val)
        amount_integrity = max(0, 1 - abs(remainder) * 4)
    else:
        amount_integrity = 0.5

    # amount_adequacy
    stype_stats = amt.get(row["subsidy_type"], {"median": 1, "std": 1})
    median_val = stype_stats["median"] or 1
    std_val = stype_stats["std"] or median_val
    amount_zscore = abs(row["amount"] - median_val) / std_val if std_val > 0 else 0
    amount_adequacy = max(0, 1 - min(amount_zscore, 3) / 3)

    # группа 2: сроки и бюджет

    # deadline_compliance
    submit_date = row.get("submit_date", None)
    deadline_compliance = check_deadline_compliance(submit_date)
    temporal_features = _single_temporal_features(
        submit_date,
        submit_month=row.get("submit_month", None),
    )

    # budget_pressure — для single scoring используем 0.5 (нет контекста очереди)
    budget_pressure = 0.5

    # queue_position — позиция относительно существующей очереди
    queue_group_sizes = tables.get("queue_group_sizes", {})
    qp_key = (row["region"], row["direction"], row["subsidy_type"])
    group_size = queue_group_sizes.get(qp_key, 0)
    # новая заявка встаёт в конец очереди
    queue_position = max(0.0, 1 - group_size / (group_size + 1)) if group_size > 0 else 0.5

    # группа 3: региональная специфика

    # region_specialization
    region_spec = spec.get((row["region"], row["direction"]), 0.1)

    # region_direction_approval_rate
    rd_key = (row["region"], row["direction"])
    rd_data = ar.get("region_direction", {}).get(rd_key, {})
    region_direction_rate = rd_data.get("approval_rate", 0.5)

    # группа 4: Акимат

    # akimat_approval_rate
    akimat_data = ar["akimat"].get(row.get("akimat", ""), {})
    akimat_rate = akimat_data.get("approval_rate", 0.5)

    # старые сохранённые

    direction_rate = ar["direction"].get(row["direction"], {}).get("approval_rate", 0.5)
    subsidy_rate = ar["subsidy_type"].get(row["subsidy_type"], {}).get("approval_rate", 0.5)
    region_rate = ar["region"].get(row["region"], {}).get("approval_rate", 0.5)

    # unit_count (перцентильный ранг относительно batch)
    if effective_norm and effective_norm > 0:
        unit_count_raw = row["amount"] / effective_norm
    else:
        unit_count_raw = 0
    unit_count_norm = 0.5  # для single scoring без batch-контекста
    if original_norm and original_norm > 0:
        unit_count_original_raw = row["amount"] / original_norm
    else:
        unit_count_original_raw = 0
    amount_to_type_median_ratio = row["amount"] / median_val if median_val > 0 else 1.0
    amount_log = float(np.log1p(max(row["amount"], 0)))
    direction_history_count_log = float(
        np.log1p(ar["direction"].get(row["direction"], {}).get("total_apps", 0))
    )
    subsidy_type_history_count_log = float(
        np.log1p(ar["subsidy_type"].get(row["subsidy_type"], {}).get("total_apps", 0))
    )
    region_direction_history_count_log = float(
        np.log1p(rd_data.get("total_apps", 0))
    )
    akimat_history_count_log = float(
        np.log1p(akimat_data.get("total_apps", 0))
    )

    result = {
        "normative_match": normative_match,
        "normative_original_match": float(normative_original_match),
        "normative_original_log": float(np.log1p(max(original_norm, 0.0))),
        "normative_reference_gap": normative_reference_gap,
        "normative_reference_typicality": normative_reference_typicality,
        "amount_normative_integrity": amount_integrity,
        "amount_adequacy": amount_adequacy,
        "budget_pressure": budget_pressure,
        "queue_position": queue_position,
        "region_specialization": region_spec,
        "region_direction_approval_rate": region_direction_rate,
        "akimat_approval_rate": akimat_rate,
        "unit_count": unit_count_norm,
        "direction_approval_rate": direction_rate,
        "subsidy_type_approval_rate": subsidy_rate,
        "region_approval_rate": region_rate,
        "direction_history_count_log": direction_history_count_log,
        "subsidy_type_history_count_log": subsidy_type_history_count_log,
        "region_direction_history_count_log": region_direction_history_count_log,
        "akimat_history_count_log": akimat_history_count_log,
        "amount_log": amount_log,
        "unit_count_original_log": float(
            np.log1p(np.clip(unit_count_original_raw, 0, 500))
        ),
        "amount_to_type_median_ratio": float(np.clip(amount_to_type_median_ratio, 0, 5)),
        "submit_month_sin": temporal_features["submit_month_sin"],
        "submit_month_cos": temporal_features["submit_month_cos"],
        # condition-признаки: для single scoring используем defaults (0.5),
        # реальные значения подставляются через batch или API
        "pasture_compliance": 0.5,
        "mortality_compliance": 0.5,
        "grazing_utilization": 0.5,
        "criteria_complexity": 0.5,
        "direction_risk": 0.5,
        "regional_pasture_capacity": 0.5,
    }
    condition_context = build_condition_context_features(pd.DataFrame([row])).iloc[0].to_dict()
    for column_name in CONDITION_REFERENCE_COLUMNS + SAFE_CONDITION_FEATURE_COLUMNS:
        if column_name in condition_context and not pd.isna(condition_context[column_name]):
            result[column_name] = float(condition_context[column_name])
    return result


def extract_features_single_with_history(
    row: pd.Series,
    history_df: pd.DataFrame,
    normative_lookup: dict[str, int],
) -> dict:
    row_series = row.copy()
    submit_date = pd.to_datetime(row_series.get("submit_date"), errors="coerce")
    submit_month = row_series.get("submit_month")
    if pd.isna(submit_month):
        submit_month = submit_date.month if not pd.isna(submit_date) else 6

    request_row = pd.DataFrame(
        [
            {
                "app_number": "\uffff__request__",
                "region": str(row_series.get("region", "")).strip(),
                "district": str(row_series.get("district", "")).strip(),
                "direction": str(row_series.get("direction", "")).strip(),
                "subsidy_type": str(row_series.get("subsidy_type", "")).strip(),
                "akimat": str(row_series.get("akimat", "")).strip(),
                "status": "Черновик",
                "normative": pd.to_numeric(
                    pd.Series([row_series.get("normative")]), errors="coerce"
                ).fillna(0).iloc[0],
                "normative_original": pd.to_numeric(
                    pd.Series(
                        [row_series.get("normative_original", row_series.get("normative"))]
                    ),
                    errors="coerce",
                ).fillna(0).iloc[0],
                "amount": pd.to_numeric(
                    pd.Series([row_series.get("amount")]), errors="coerce"
                ).fillna(0).iloc[0],
                "submit_date": submit_date,
                "submit_month": submit_month,
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
    if not pd.isna(submit_date):
        history = history[history["submit_date"].isna() | (history["submit_date"] <= submit_date)]

    combined = pd.concat([history[required_columns], request_row], ignore_index=True)
    features = extract_features_batch(
        combined,
        {"normative_lookup": normative_lookup},
    )
    return features.iloc[-1].to_dict()


def extract_features_batch(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    # векторизованное извлечение признаков для всего датафрейма
    norm_lookup = tables["normative_lookup"]

    features = pd.DataFrame(index=df.index)

    # normative_match
    ref_norms = df["subsidy_type"].map(
        lambda st: get_normative_for_type(st, norm_lookup)
    )
    # сравниваем normative из заявки (уже заменённый в pipeline) с эталоном
    has_ref = ref_norms.notna()
    ref_filled = ref_norms.fillna(0)
    row_norms = df["normative"]

    # точное совпадение = 1.0
    exact_match = (row_norms == ref_filled) & has_ref
    # отклонение < 5% = 0.8
    deviation = np.where(
        ref_filled > 0,
        np.abs(row_norms - ref_filled) / ref_filled,
        1.0,
    )
    close_match = (deviation < 0.05) & ~exact_match & has_ref

    features["normative_match"] = np.where(
        ~has_ref, 0.5,
        np.where(exact_match, 1.0,
                 np.where(close_match, 0.8, 0.0))
    )

    original_norms = (
        pd.to_numeric(df.get("normative_original", df["normative"]), errors="coerce")
        .fillna(0.0)
    )
    original_deviation = pd.Series(
        np.where(
            ref_filled > 0,
            np.abs(original_norms - ref_filled) / ref_filled,
            np.nan,
        ),
        index=df.index,
    ).replace([np.inf, -np.inf], np.nan)
    features["normative_original_log"] = np.log1p(original_norms.clip(lower=0.0))
    features["normative_reference_gap"] = (
        original_deviation.fillna(1.0).clip(0, 3).astype(float)
    )
    features["normative_reference_typicality"] = _ratio_to_typicality(
        (
            (original_norms / ref_filled.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
        ),
        neutral_value=1.0,
    ).reindex(df.index, fill_value=1.0)
    features["normative_original_match"] = np.where(
        ~has_ref,
        0.5,
        np.where(
            original_deviation.fillna(1.0).eq(0),
            1.0,
            np.where(original_deviation.fillna(1.0) < 0.05, 0.8, 0.0),
        ),
    )

    # amount_normative_integrity
    effective_norm = np.where(ref_filled > 0, ref_filled, row_norms)
    effective_norm_safe = pd.Series(effective_norm, index=df.index).replace(0, np.nan)

    unit_counts = df["amount"] / effective_norm_safe
    remainders = unit_counts - unit_counts.round()
    features["amount_normative_integrity"] = (
        (1 - remainders.abs() * 4).clip(0, 1).fillna(0.5)
    )

    # amount_adequacy — считаем по историческому распределению,
    # чтобы не использовать будущие суммы
    features["amount_adequacy"] = compute_historical_amount_adequacy(df)

    # unit_count — относительный размер заявки к исторической медиане,
    # без использования будущих заявок того же типа
    features["_unit_count_raw"] = df["amount"] / effective_norm_safe.fillna(1)
    df_sorted = _sort_by_submit_order(df)
    unit_raw_sorted = features.loc[df_sorted["_orig_index"], "_unit_count_raw"]
    unit_hist_median = unit_raw_sorted.groupby(
        df_sorted["subsidy_type"], sort=False
    ).transform(lambda s: s.expanding().median().shift(1))
    global_unit_hist_median = unit_raw_sorted.expanding().median().shift(1)
    unit_hist_median = unit_hist_median.fillna(global_unit_hist_median).replace(0, np.nan)
    unit_hist_median = unit_hist_median.fillna(unit_raw_sorted).replace(0, np.nan)
    unit_hist_median = unit_hist_median.fillna(1.0)
    unit_scaled = (unit_raw_sorted / unit_hist_median).clip(0, 2) / 2
    features.loc[df_sorted["_orig_index"], "unit_count"] = unit_scaled.to_numpy()
    features["unit_count"] = features["unit_count"].fillna(0.5)
    unit_count_original = (
        df["amount"] / original_norms.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["unit_count_original_log"] = np.log1p(unit_count_original.clip(0, 500))
    features.drop(columns=["_unit_count_raw"], inplace=True)

    # budget_pressure
    features["budget_pressure"] = compute_budget_pressure(df)

    # queue_position
    features["queue_position"] = compute_queue_position(df)

    # региональные и approval-rate признаки считаем только по прошлой истории
    features["region_specialization"] = compute_historical_region_specialization(df)
    features["region_approval_rate"] = compute_historical_approval_rate(df, "region")
    features["region_direction_approval_rate"] = compute_historical_approval_rate(
        df, ["region", "direction"]
    )
    features["akimat_approval_rate"] = compute_historical_approval_rate(df, "akimat")
    features["direction_approval_rate"] = compute_historical_approval_rate(
        df, "direction"
    )
    features["subsidy_type_approval_rate"] = compute_historical_approval_rate(
        df, "subsidy_type"
    )
    features["direction_history_count_log"] = np.log1p(
        compute_historical_group_count(df, "direction")
    )
    features["subsidy_type_history_count_log"] = np.log1p(
        compute_historical_group_count(df, "subsidy_type")
    )
    features["region_direction_history_count_log"] = np.log1p(
        compute_historical_group_count(df, ["region", "direction"])
    )
    features["akimat_history_count_log"] = np.log1p(
        compute_historical_group_count(df, "akimat")
    )
    features["amount_log"] = np.log1p(df["amount"].clip(lower=0))
    features["amount_to_type_median_ratio"] = compute_historical_amount_to_median_ratio(df)
    submit_month = pd.to_datetime(df["submit_date"], errors="coerce").dt.month.fillna(
        df.get("submit_month", 6)
    )
    angle = 2 * np.pi * (submit_month - 1) / 12
    features["submit_month_sin"] = np.sin(angle)
    features["submit_month_cos"] = np.cos(angle)

    context_features = build_condition_context_features(df)
    for column_name in CONDITION_REFERENCE_COLUMNS + SAFE_CONDITION_FEATURE_COLUMNS:
        if column_name in context_features.columns:
            features[column_name] = pd.to_numeric(
                context_features[column_name],
                errors="coerce",
            ).fillna(0.5 if column_name in SAFE_CONDITION_FEATURE_COLUMNS else np.nan)

    cond = load_condition_features(columns=LEAKY_CONDITION_FEATURE_COLUMNS)
    if cond is not None and "app_number" in df.columns:
        cond["app_number"] = cond["app_number"].astype(str).str.lstrip("0")
        app_key = df["app_number"].astype(str).str.lstrip("0")
        cond_indexed = cond.set_index("app_number")
        for col in LEAKY_CONDITION_FEATURE_COLUMNS:
            if col in cond_indexed.columns:
                features[col] = app_key.map(cond_indexed[col]).fillna(0.5)
            else:
                features[col] = 0.5
    else:
        for col in LEAKY_CONDITION_FEATURE_COLUMNS:
            features[col] = 0.5

    for col in SAFE_CONDITION_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.5
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.5).clip(0, 1)

    return features
