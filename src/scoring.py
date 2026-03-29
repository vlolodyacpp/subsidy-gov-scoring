# экспертные веса на основе правил субсидирования 

import pandas as pd
import numpy as np
from dataclasses import dataclass


# веса факторов (сумма = 1.0)
WEIGHTS = {
    "region_approval_rate": 0.25,       # региональная одобряемость
    "direction_approval_rate": 0.20,    # одобряемость направления
    "subsidy_type_approval_rate": 0.20, # одобряемость типа субсидии
    "amount_adequacy": 0.20,            # адекватность суммы
    "season_approval_rate": 0.15,       # сезонность подачи
}

FACTOR_LABELS = {
    "region_approval_rate": "Региональная одобряемость",
    "direction_approval_rate": "Одобряемость направления",
    "subsidy_type_approval_rate": "Одобряемость типа субсидии",
    "amount_adequacy": "Адекватность суммы заявки",
    "season_approval_rate": "Сезонность подачи",
}


@dataclass
class ScoringResult:
    score: float                    # итоговый балл 0–100
    factors: dict[str, float]       # вклад каждого фактора
    explanation: list[str]          # текстовые объяснения
    risk_level: str                 # низкий / средний / высокий


def score_single(features: dict) -> ScoringResult:
    # скоринг одной заявки с объяснением 
    factors = {}
    raw_score = 0.0

    for factor, weight in WEIGHTS.items():
        value = features.get(factor, 0.5)
        contribution = value * weight * 100
        factors[factor] = round(contribution, 2)
        raw_score += contribution

    score = round(min(max(raw_score, 0), 100), 1)

    # генерация объяснений
    explanation = []
    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    # сортируем факторы по убыванию вклада для лучшей наглядности пользователю
    for name, contrib in sorted_factors:
        label = FACTOR_LABELS.get(name, name)
        value = features.get(name, 0)
        if contrib >= 15:
            explanation.append(f"✓ {label}: высокий ({value:.0%}) — +{contrib:.1f} баллов")
        elif contrib >= 8:
            explanation.append(f"● {label}: средний ({value:.0%}) — +{contrib:.1f} баллов")
        else:
            explanation.append(f"✗ {label}: низкий ({value:.0%}) — +{contrib:.1f} баллов")

    # определяем уровень риска
    risk_level = "низкий" if score >= 70 else "средний" if score >= 45 else "высокий"

    return ScoringResult(
        score=score,
        factors=factors,
        explanation=explanation,
        risk_level=risk_level,
    )


def score_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    # векторизированный скоринг каждой заявки.
    scores = pd.Series(0.0, index=features_df.index)
    factor_contributions = {}

    for factor, weight in WEIGHTS.items():
        if factor in features_df.columns:
            contrib = features_df[factor] * weight * 100
            factor_contributions[f"contrib_{factor}"] = contrib
            scores += contrib

    result = pd.DataFrame(factor_contributions, index=features_df.index)
    result["score"] = scores.clip(0, 100).round(1)
    result["risk_level"] = pd.cut(
        result["score"],
        bins=[-1, 45, 70, 101],
        labels=["Высокий", "Средний", "Низкий"],
    )

    # главный фактор
    contrib_cols = [c for c in result.columns if c.startswith("contrib_")]
    result["top_factor"] = result[contrib_cols].idxmax(axis=1).str.replace("contrib_", "")
    result["top_factor_label"] = result["top_factor"].map(FACTOR_LABELS)

    return result


def generate_shortlist(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    top_n: int = 50,
    output_dir: str = "output",
) -> pd.DataFrame:
    # формирование shortlist — топ-N заявок для комиссии. Сохраняет CSV 
    from pathlib import Path

    combined = pd.concat([df, scores], axis=1)
    shortlist = (
        combined
        .sort_values("score", ascending=False)
        .head(top_n)
        [["app_number", "region", "district", "direction", "subsidy_type",
          "amount", "score", "risk_level", "top_factor_label"]]
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "shortlist.csv"
    shortlist.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Shortlist сохранён → {csv_path}")

    return shortlist


def get_score_distribution(scores: pd.DataFrame) -> dict:
    # cтатистика распределения скоров. 
    s = scores["score"]
    return {
        "mean": round(s.mean(), 1),
        "median": round(s.median(), 1),
        "std": round(s.std(), 1),
        "min": round(s.min(), 1),
        "max": round(s.max(), 1),
        "risk_distribution": scores["risk_level"].value_counts().to_dict(),
    }
