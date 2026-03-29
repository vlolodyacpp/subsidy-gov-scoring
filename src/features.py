# вычисление признаков для скоринга
import pandas as pd
import numpy as np


def compute_approval_rates(df: pd.DataFrame) -> dict:
    # вычисление процентов одобрений по регионам, направлениям, типам субсидий
    rates = {}

    for col in ["region", "direction", "subsidy_type", "district"]:
        group = df.groupby(col)["is_approved"].agg(["mean", "count"])
        group.columns = ["approval_rate", "total_apps"]
        rates[col] = group.to_dict("index")

    return rates


def compute_amount_stats(df: pd.DataFrame) -> dict:
    # оценки адекватности - медиана и стандартное отклонение суммы по типу субсидии
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


def compute_seasonality_scores(df: pd.DataFrame) -> dict:
    # ранние заявки одобряются чаще
    monthly = df.groupby("submit_month")["is_approved"].mean()
    return monthly.to_dict()


def build_feature_tables(df: pd.DataFrame) -> dict:
    # сборка всех справочных таблиц для скоринга
    return {
        "approval_rates": compute_approval_rates(df),
        "amount_stats": compute_amount_stats(df),
        "seasonality": compute_seasonality_scores(df),
    }


def extract_features(row: pd.Series, tables: dict) -> dict:
    # извлечение числовых признаков из одной заявки
    ar = tables["approval_rates"]
    amt = tables["amount_stats"]
    seas = tables["seasonality"]

  
    region_rate = ar["region"].get(row["region"], {}).get("approval_rate", 0.5)

    direction_rate = ar["direction"].get(row["direction"], {}).get("approval_rate", 0.5)

    subsidy_rate = ar["subsidy_type"].get(row["subsidy_type"], {}).get("approval_rate", 0.5)

    stype_stats = amt.get(row["subsidy_type"], {"median": 1, "std": 1})
    median_val = stype_stats["median"] or 1
    std_val = stype_stats["std"] or median_val
    amount_zscore = abs(row["amount"] - median_val) / std_val if std_val > 0 else 0

    amount_adequacy = max(0, 1 - min(amount_zscore, 3) / 3)

    month = row.get("submit_month", 6)
    season_rate = seas.get(month, 0.5)

    # количество заявок в этом районе (активность района)
    district_apps = ar["district"].get(row["district"], {}).get("total_apps", 1)

    return {
        "region_approval_rate": region_rate,
        "direction_approval_rate": direction_rate,
        "subsidy_type_approval_rate": subsidy_rate,
        "amount_adequacy": amount_adequacy,
        "season_approval_rate": season_rate,
        "district_activity": min(district_apps / 100, 1),  # нормализация
        "amount": row["amount"],
        "normative": row["normative"],
    }


def extract_features_batch(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    #векторизованное извлечение признаков для всего датафрейма 
    ar = tables["approval_rates"]
    amt = tables["amount_stats"]
    seas = tables["seasonality"]

    features = pd.DataFrame(index=df.index)

    # маппинг через словари
    features["region_approval_rate"] = df["region"].map(
        {k: v["approval_rate"] for k, v in ar["region"].items()}
    ).fillna(0.5)

    features["direction_approval_rate"] = df["direction"].map(
        {k: v["approval_rate"] for k, v in ar["direction"].items()}
    ).fillna(0.5)

    features["subsidy_type_approval_rate"] = df["subsidy_type"].map(
        {k: v["approval_rate"] for k, v in ar["subsidy_type"].items()}
    ).fillna(0.5)

    features["season_approval_rate"] = df["submit_month"].map(seas).fillna(0.5)

    features["district_activity"] = df["district"].map(
        {k: min(v["total_apps"] / 100, 1) for k, v in ar["district"].items()}
    ).fillna(0.1)

    # векторизированная адекватность суммы
    medians = df["subsidy_type"].map(
        {k: v["median"] for k, v in amt.items()}
    ).fillna(1)
    stds = df["subsidy_type"].map(
        {k: v["std"] for k, v in amt.items()}
    ).fillna(1).replace(0, 1)

    z_scores = (df["amount"] - medians).abs() / stds
    features["amount_adequacy"] = (1 - z_scores.clip(0, 3) / 3).clip(0, 1)

    features["amount"] = df["amount"]
    features["normative"] = df["normative"]
    features["amount_to_norm_ratio"] = (
        df["amount"] / df["normative"].replace(0, 1)
    ).clip(0, 1000)

    return features
