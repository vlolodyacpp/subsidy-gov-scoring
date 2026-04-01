import pandas as pd
import numpy as np

from src.normatives import (
    build_normative_lookup,
    get_normative_for_type,
)


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


def compute_budget_pressure(df: pd.DataFrame) -> pd.Series:
    # бюджетное давление
    df_sorted = df.sort_values("submit_date").copy()

    # кумулятивная сумма запрошенных средств внутри (region, direction)
    df_sorted["cum_requested"] = df_sorted.groupby(
        ["region", "direction"]
    )["amount"].cumsum()

    # общая одобренная сумма по (region, direction)
    approved_totals = (
        df[df["is_approved"] == 1]
        .groupby(["region", "direction"])["amount"]
        .sum()
    )

    # маппинг общей одобренной суммы
    df_sorted["total_approved"] = df_sorted.apply(
        lambda row: approved_totals.get((row["region"], row["direction"]), 0),
        axis=1,
    )

    # budget_pressure: чем меньше запрошено относительно одобренного — тем ближе к 1.0
    pressure = np.where(
        df_sorted["total_approved"] > 0,
        np.clip(1 - df_sorted["cum_requested"] / df_sorted["total_approved"], 0, 1),
        0.5,  # неизвестно
    )

    # Вернуть в исходном порядке индексов
    result = pd.Series(pressure, index=df_sorted.index)
    return result.reindex(df.index)


def compute_queue_position(df: pd.DataFrame) -> pd.Series:
    # позиция в очереди подачи
    df_sorted = df.sort_values("submit_date").copy()

    # ранг внутри группы
    df_sorted["rank"] = df_sorted.groupby(
        ["region", "direction", "subsidy_type"]
    ).cumcount() + 1

    # общее количество в группе
    group_sizes = df.groupby(
        ["region", "direction", "subsidy_type"]
    ).size()

    df_sorted["group_total"] = df_sorted.apply(
        lambda row: group_sizes.get(
            (row["region"], row["direction"], row["subsidy_type"]), 1
        ),
        axis=1,
    )

    # нормализация: ранние = ближе к 1.0
    position = np.where(
        df_sorted["group_total"] > 1,
        1 - (df_sorted["rank"] - 1) / (df_sorted["group_total"] - 1),
        0.5,  # единственная заявка в группе
    )

    result = pd.Series(position, index=df_sorted.index)
    return result.reindex(df.index)


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
    # извлечение 12 числовых признаков из одной заявки
    ar = tables["approval_rates"]
    amt = tables["amount_stats"]
    spec = tables["region_specialization"]
    norm_lookup = tables["normative_lookup"]

    # группа 1: нормативное соответствие

    # normative_match
    ref_norm = get_normative_for_type(row["subsidy_type"], norm_lookup)
    row_norm = row.get("normative", 0)
    if ref_norm is None:
        normative_match = 0.5
    elif row_norm == ref_norm:
        normative_match = 1.0
    elif ref_norm > 0 and abs(row_norm - ref_norm) / ref_norm < 0.05:
        normative_match = 0.8
    else:
        normative_match = 0.0

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

    # budget_pressure — оценка на основе batch-контекста
    approved_totals = tables.get("approved_totals", {})
    total_requested = tables.get("total_requested", {})
    bp_key = (row["region"], row["direction"])
    bp_approved = approved_totals.get(bp_key, 0)
    bp_requested = total_requested.get(bp_key, 0)
    if bp_approved > 0:
        # новая заявка добавляется к уже запрошенному объёму
        budget_pressure = max(0.0, min(1.0, 1 - (bp_requested + row["amount"]) / bp_approved))
    else:
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

    # unit_count (перцентильный ранг относительно batch)
    if effective_norm and effective_norm > 0:
        unit_count_raw = row["amount"] / effective_norm
    else:
        unit_count_raw = 0

    unit_count_by_type = tables.get("unit_count_by_type", {})
    type_values = unit_count_by_type.get(row["subsidy_type"], [])
    if type_values and unit_count_raw > 0:
        # бинарный поиск для перцентильного ранга
        import bisect
        pos = bisect.bisect_left(type_values, unit_count_raw)
        unit_count_norm = pos / len(type_values)
    else:
        unit_count_norm = 0.5

    return {
        "normative_match": normative_match,
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
    }


def extract_features_batch(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    # векторизованное извлечение всех 12 признаков для всего датафрейма
    ar = tables["approval_rates"]
    amt = tables["amount_stats"]
    spec = tables["region_specialization"]
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

    # amount_normative_integrity
    effective_norm = np.where(ref_filled > 0, ref_filled, row_norms)
    effective_norm_safe = pd.Series(effective_norm, index=df.index).replace(0, np.nan)

    unit_counts = df["amount"] / effective_norm_safe
    remainders = unit_counts - unit_counts.round()
    features["amount_normative_integrity"] = (
        (1 - remainders.abs() * 4).clip(0, 1).fillna(0.5)
    )

    # amount_adequacy 
    medians = df["subsidy_type"].map(
        {k: v["median"] for k, v in amt.items()}
    ).fillna(1)
    stds = df["subsidy_type"].map(
        {k: v["std"] for k, v in amt.items()}
    ).fillna(1).replace(0, 1)

    z_scores = (df["amount"] - medians).abs() / stds
    features["amount_adequacy"] = (1 - z_scores.clip(0, 3) / 3).clip(0, 1)

    # unit_count (перцентильный ранг)
    features["_unit_count_raw"] = df["amount"] / effective_norm_safe.fillna(1)

    # перцентильный ранг внутри subsidy_type
    features["unit_count"] = features.groupby(df["subsidy_type"])[
        "_unit_count_raw"
    ].rank(pct=True).fillna(0.5)
    features.drop(columns=["_unit_count_raw"], inplace=True)

    # budget_pressure
    features["budget_pressure"] = compute_budget_pressure(df)

    # queue_position
    features["queue_position"] = compute_queue_position(df)

    # region_specialization
    features["region_specialization"] = df.apply(
        lambda row: spec.get((row["region"], row["direction"]), 0.1),
        axis=1,
    )

    # region_direction_approval_rate
    rd_rates = ar.get("region_direction", {})
    features["region_direction_approval_rate"] = df.apply(
        lambda row: rd_rates.get(
            (row["region"], row["direction"]), {}
        ).get("approval_rate", 0.5),
        axis=1,
    )

    # akimat_approval_rate
    akimat_rates = {k: v["approval_rate"] for k, v in ar["akimat"].items()}
    features["akimat_approval_rate"] = df["akimat"].map(akimat_rates).fillna(0.5)

    features["direction_approval_rate"] = df["direction"].map(
        {k: v["approval_rate"] for k, v in ar["direction"].items()}
    ).fillna(0.5)

    features["subsidy_type_approval_rate"] = df["subsidy_type"].map(
        {k: v["approval_rate"] for k, v in ar["subsidy_type"].items()}
    ).fillna(0.5)

    return features
