import pandas as pd
from dataclasses import dataclass


# веса факторов v3 (сумма = 1.0)
# 17 факторов, 6 групп
WEIGHTS = {
    # группа 1: нормативное соответствие (20%)
    "normative_match":              0.08,
    "amount_normative_integrity":   0.06,
    "amount_adequacy":              0.06,

    # группа 2: бюджет и очередь (22%)
    "budget_pressure":              0.13,
    "queue_position":               0.09,

    # группа 3: региональная специфика (20%)
    "region_specialization":        0.07,
    "region_direction_approval_rate": 0.08,
    "akimat_approval_rate":         0.05,

    # группа 4: характеристики заявки (10%)
    "unit_count":                   0.04,
    "direction_approval_rate":      0.03,
    "subsidy_type_approval_rate":   0.03,

    # группа 5: условия содержания (20%)
    "pasture_compliance":           0.08,
    "mortality_compliance":         0.07,
    "grazing_utilization":          0.05,

    # группа 6: регуляторная сложность (8%)
    "criteria_complexity":          0.03,
    "direction_risk":               0.03,
    "regional_pasture_capacity":    0.02,
}

FACTOR_LABELS = {
    "normative_match":              "Соответствие норматива эталону",
    "amount_normative_integrity":   "Арифметическая корректность суммы",
    "amount_adequacy":              "Адекватность суммы заявки",
    "budget_pressure":              "Бюджетное давление",
    "queue_position":               "Позиция в очереди подачи",
    "region_specialization":        "Профильность направления для региона",
    "region_direction_approval_rate": "Одобряемость направления в регионе",
    "akimat_approval_rate":         "Уровень одобрения акимата",
    "unit_count":                   "Количество заявленных единиц",
    "direction_approval_rate":      "Одобряемость направления",
    "subsidy_type_approval_rate":   "Одобряемость типа субсидии",
    "pasture_compliance":           "Соответствие нагрузки на пастбища нормам",
    "mortality_compliance":         "Соответствие падежа допустимым нормам",
    "grazing_utilization":          "Использование пастбищного сезона",
    "criteria_complexity":          "Простота регуляторной проверки",
    "direction_risk":               "Биологическая безопасность направления",
    "regional_pasture_capacity":    "Ресурсная ёмкость пастбищ региона",
}

# группировка факторов для объяснений
FACTOR_GROUPS = {
    "Нормативное соответствие": ["normative_match", "amount_normative_integrity", "amount_adequacy"],
    "Бюджет и очередь": ["budget_pressure", "queue_position"],
    "Региональная специфика": ["region_specialization", "region_direction_approval_rate", "akimat_approval_rate"],
    "Характеристики заявки": ["unit_count", "direction_approval_rate", "subsidy_type_approval_rate"],
    "Условия содержания": ["pasture_compliance", "mortality_compliance", "grazing_utilization"],
    "Регуляторная сложность": ["criteria_complexity", "direction_risk", "regional_pasture_capacity"],
}

DEADLINE_DISQUALIFICATION_REASON = "Просрочен срок подачи заявки"


@dataclass
class ScoringResult:
    score: float                    # итоговый балл 0–100
    factors: dict[str, float]       # вклад каждого фактора
    explanation: list[str]          # текстовые объяснения
    risk_level: str                 # низкий / средний / высокий


def get_disqualification_reason(features: dict | pd.Series) -> str | None:
    deadline_value = features.get("deadline_compliance", 0.5)
    if pd.isna(deadline_value):
        return None
    if float(deadline_value) <= 0:
        return DEADLINE_DISQUALIFICATION_REASON
    return None


def score_single(features: dict) -> ScoringResult:
    # скоринг одной допустимой заявки с explainability по rule-факторам
    factors = {}
    raw_score = 0.0

    for factor, weight in WEIGHTS.items():
        value = features.get(factor, 0.5)
        contribution = value * weight * 100
        factors[factor] = round(contribution, 2)
        raw_score += contribution

    score = round(min(max(raw_score, 0), 100), 1)

    # генерация объяснений по группам — человечным языком
    explanation = []

    GROUP_DESCRIPTIONS = {
        "Нормативное соответствие": {
            "high": "документы полностью соответствуют установленным правилам",
            "medium": "есть незначительные расхождения с нормативами",
            "low": "выявлены существенные расхождения с эталонными нормативами",
        },
        "Бюджет и очередь": {
            "high": "заявка подана в благоприятный период, конкуренция невысокая",
            "medium": "средняя конкуренция за бюджетные средства",
            "low": "высокая конкуренция или поздняя подача заявки",
        },
        "Региональная специфика": {
            "high": "направление хорошо развито в регионе, высокая одобряемость",
            "medium": "направление умеренно представлено в регионе",
            "low": "направление нетипично для данного региона",
        },
        "Характеристики заявки": {
            "high": "параметры заявки характерны для успешных обращений",
            "medium": "параметры заявки в рамках нормы",
            "low": "параметры заявки нетипичны для одобряемых обращений",
        },
        "Условия содержания": {
            "high": "нагрузка на пастбища и падёж в пределах допустимых норм",
            "medium": "частичное соответствие нормам содержания животных",
            "low": "показатели содержания существенно отклоняются от допустимых норм",
        },
        "Регуляторная сложность": {
            "high": "направление и регион благоприятны с точки зрения регуляторных требований",
            "medium": "умеренная регуляторная нагрузка на данный тип заявки",
            "low": "высокая регуляторная сложность, повышенный биологический риск направления",
        },
    }

    for group_name, group_factors in FACTOR_GROUPS.items():
        group_total = sum(factors.get(f, 0) for f in group_factors)
        group_max = sum(WEIGHTS.get(f, 0) for f in group_factors) * 100

        if group_max > 0:
            group_pct = group_total / group_max
        else:
            group_pct = 0

        if group_pct >= 0.7:
            level_key = "high"
        elif group_pct >= 0.4:
            level_key = "medium"
        else:
            level_key = "low"

        desc = GROUP_DESCRIPTIONS.get(group_name, {}).get(level_key, "")
        explanation.append(
            f"{group_name}: {desc} "
            f"(оценка {group_total:.1f} из {group_max:.0f})."
        )

        # детализация по факторам внутри группы
        for factor in group_factors:
            value = features.get(factor, 0)
            contrib = factors.get(factor, 0)
            label = FACTOR_LABELS.get(factor, factor)
            max_contrib = WEIGHTS.get(factor, 0) * 100
            explanation.append(f"  • {label}: {contrib:.1f} из {max_contrib:.0f}")

    risk_level = "низкий" if score >= 70 else "средний" if score >= 45 else "высокий"

    return ScoringResult(
        score=score,
        factors=factors,
        explanation=explanation,
        risk_level=risk_level,
    )


def score_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    # векторизированный скоринг допустимых заявок + eligibility-флаг по дедлайну
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
    result["disqualified"] = False
    result["disqualification_reason"] = None

    deadline_mask = features_df.get("deadline_compliance", pd.Series(0.5, index=features_df.index)).fillna(0.5) <= 0
    if deadline_mask.any():
        result.loc[deadline_mask, "disqualified"] = True
        result.loc[deadline_mask, "disqualification_reason"] = DEADLINE_DISQUALIFICATION_REASON

    # главный фактор и главная группа
    contrib_cols = [c for c in result.columns if c.startswith("contrib_")]
    result["top_factor"] = result[contrib_cols].idxmax(axis=1).str.replace("contrib_", "")
    result["top_factor_label"] = result["top_factor"].map(FACTOR_LABELS)

    # группа с наибольшим суммарным вкладом
    for group_name, group_factors in FACTOR_GROUPS.items():
        group_cols = [f"contrib_{f}" for f in group_factors if f"contrib_{f}" in result.columns]
        if group_cols:
            result[f"group_{group_name}"] = result[group_cols].sum(axis=1).round(1)

    return result


def generate_shortlist(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    top_n: int = 50,
    output_dir: str = "output",
) -> pd.DataFrame:
    # формирование shortlist — топ-N заявок для комиссии
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
    # cтатистика распределения скоров 
    s = scores["score"]
    return {
        "mean": round(s.mean(), 1),
        "median": round(s.median(), 1),
        "std": round(s.std(), 1),
        "min": round(s.min(), 1),
        "max": round(s.max(), 1),
        "risk_distribution": scores["risk_level"].value_counts().to_dict(),
    }
