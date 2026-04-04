import logging
import shutil
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.advisory import (
    build_history_advisory_batch,
    build_history_advisory_single_from_tables,
    build_history_advisory_tables,
)
from src.eligibility import (
    DECISION_SUPPORT_NOTE,
    evaluate_batch_eligibility,
    evaluate_single_eligibility,
)
from src.pipeline import run_pipeline
from src.features import (
    build_feature_tables,
    extract_features_batch,
    extract_features_single_with_history,
)
from src.modeling import (
    DEFAULT_BLEND_WEIGHTS,
    build_primary_model_frame,
    explain_prediction_with_model,
    load_bundle,
    score_features_with_model,
)
from src.normatives import get_normative_for_type
from src.scoring import (
    score_single,
    score_batch,
    get_score_distribution,
    FACTOR_LABELS,
)
from src.schemas import (
    ScoreRequest,
    ScoreResponse,
    RankRequest,
    RankResponse,
    ExplainResponse,
    StatsResponse,
    HealthResponse,
    FactorDetail,
    ApplicationBrief,
    RegionStat,
    PaginatedApplications,
)

logger = logging.getLogger("subsidy_api")
logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/subsidies.xlsx"
MODEL_PATH = "models/artifacts/subsidy_model.joblib"
API_VERSION = "4.0.0"
SCORING_ENGINE_NAME = "merit-ml-advisory-v4.0"
RULE_ENGINE_NAME = "eligibility-rules-v4.0"


def _load_model_bundle_if_available(model_path: str) -> dict | None:
    path = Path(model_path)
    if not path.exists():
        logger.error("ML-модель не найдена: %s.", path)
        return None

    try:
        bundle = load_bundle(path)
        logger.info(
            "ML-модель загружена: %s (%s)",
            bundle.get("model_name", "unknown"),
            path,
        )
        return bundle
    except Exception:
        logger.exception("Не удалось загрузить ML-модель из %s. Фолбэк на rule-based.", path)
        return None


def _get_scoring_engine() -> str:
    return SCORING_ENGINE_NAME if getattr(app.state, "model_bundle", None) else RULE_ENGINE_NAME


def _get_model_name() -> str | None:
    bundle = getattr(app.state, "model_bundle", None)
    if not bundle:
        return None
    return bundle.get("model_name")


def _init_runtime_monitor() -> dict[str, object]:
    return {
        "score_requests_total": 0,
        "rank_requests_total": 0,
        "explain_requests_total": 0,
        "score_latency_total_ms": 0.0,
        "rank_latency_total_ms": 0.0,
        "explain_latency_total_ms": 0.0,
    }


def _record_runtime_event(event_name: str, duration_ms: float) -> None:
    monitor = getattr(app.state, "runtime_monitor", None)
    if monitor is None:
        return

    key_map = {
        "score": ("score_requests_total", "score_latency_total_ms"),
        "rank": ("rank_requests_total", "rank_latency_total_ms"),
        "explain": ("explain_requests_total", "explain_latency_total_ms"),
    }
    if event_name not in key_map:
        return

    count_key, latency_key = key_map[event_name]
    monitor[count_key] = int(monitor.get(count_key, 0)) + 1
    monitor[latency_key] = float(monitor.get(latency_key, 0.0)) + float(duration_ms)


def _average_latency(event_name: str) -> float | None:
    monitor = getattr(app.state, "runtime_monitor", None)
    if monitor is None:
        return None
    count_key = f"{event_name}_requests_total"
    latency_key = f"{event_name}_latency_total_ms"
    count = int(monitor.get(count_key, 0))
    if count <= 0:
        return None
    return round(float(monitor.get(latency_key, 0.0)) / count, 3)


def _prepare_api_scores(
    model_input: pd.DataFrame,
    rule_scores: pd.DataFrame,
    advisory: pd.DataFrame,
    model_bundle: dict | None,
) -> pd.DataFrame:
    scores = pd.concat([rule_scores.copy(), advisory.copy()], axis=1)
    scores["rule_score"] = scores["score"].astype(float)
    scores["rule_risk_level"] = scores["risk_level"].astype(str)
    scores["ml_probability"] = pd.NA
    scores["ml_score"] = pd.NA
    scores["ml_decision_threshold"] = pd.NA
    scores["ml_predicted_positive"] = pd.NA

    if not model_bundle:
        return scores

    blended_scores = score_features_with_model(
        features_input=model_input,
        model=model_bundle["model"],
        rule_scores=rule_scores,
        blend_weights=model_bundle.get("blend_weights", DEFAULT_BLEND_WEIGHTS),
        decision_threshold=model_bundle.get("decision_threshold", 0.5),
        probability_calibrator=model_bundle.get("probability_calibrator"),
        probability_temperature=model_bundle.get("probability_temperature", 1.0),
        disqualified_mask=rule_scores["disqualified"],
    )
    scores["ml_probability"] = blended_scores["ml_probability"]
    scores["ml_score"] = blended_scores["ml_score"]
    scores["ml_decision_threshold"] = blended_scores["ml_decision_threshold"]
    scores["ml_predicted_positive"] = blended_scores["ml_predicted_positive"]
    scores["score"] = blended_scores["score"]
    scores["risk_level"] = blended_scores["risk_level"]
    return scores


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _single_risk_label(value: str) -> str:
    return str(value).lower()


def _safe_feature_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return round(float(value), 4)
    return str(value)


def _format_ml_effect_label(name: str, value) -> str:
    if value is None or pd.isna(value):
        return name
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{name} = {float(value):.4f}"
    return f"{name} = {value}"


# human-readable names for ML features used in explanations
_ML_FACTOR_HUMAN_NAMES = {
    "normative_match": "соответствие нормативу",
    "amount_normative_integrity": "корректность суммы",
    "amount_adequacy": "адекватность суммы",
    "budget_pressure": "бюджетное давление",
    "queue_position": "позиция в очереди",
    "region_specialization": "профильность региона",
    "region_direction_approval_rate": "одобряемость направления в регионе",
    "akimat_approval_rate": "одобрение акимата",
    "unit_count": "кол-во единиц",
    "direction_approval_rate": "одобряемость направления",
    "subsidy_type_approval_rate": "одобряемость типа субсидии",
    "rule_score": "оценка по правилам",
    "contrib_budget_pressure": "вклад бюджетного давления",
    "contrib_region_direction_approval_rate": "вклад одобряемости региона",
    "contrib_normative_match": "вклад соответствия нормативу",
    "contrib_queue_position": "вклад позиции в очереди",
    "contrib_akimat_approval_rate": "вклад одобрения акимата",
    "contrib_amount_adequacy": "вклад адекватности суммы",
    "contrib_region_specialization": "вклад специализации региона",
    "contrib_amount_normative_integrity": "вклад корректности расчёта",
    "contrib_direction_approval_rate": "вклад одобряемости направления",
    "contrib_subsidy_type_approval_rate": "вклад одобряемости типа субсидии",
    "contrib_unit_count": "вклад количества единиц",
    "contrib_normative_match": "вклад соответствия нормативу",
    "contrib_queue_position": "вклад позиции в очереди",
    "contrib_budget_pressure": "вклад бюджетного давления",
    "region_encoded": "региональный код",
    "direction_encoded": "код направления",
    "submit_month": "месяц подачи",
    "log_amount": "размер суммы (лог.)",
    "amount_per_unit": "сумма на единицу",
    # расширенные фичи модели
    "normative_log": "норматив (лог.)",
    "normative_original_match": "совпадение исходного норматива",
    "normative_reference_gap": "отклонение от эталонного норматива",
    "normative_reference_typicality": "типичность норматива",
    "unit_count_log": "кол-во единиц (лог.)",
    "unit_count_original_log": "исходное кол-во единиц (лог.)",
    "submit_month_sin": "сезонность подачи",
    "region_approval_rate": "одобряемость в регионе",
    "region_direction_lift": "эффект направления в регионе",
    "akimat_lift": "эффект акимата",
    "direction_history_count_log": "история заявок по направлению",
    "subsidy_type_history_count_log": "история заявок по типу субсидии",
    "region_direction_history_count_log": "история заявок направления в регионе",
    "akimat_history_count_log": "история заявок акимата",
    "amount_to_normative_ratio": "отношение суммы к нормативу",
    "amount_to_type_median_ratio": "отношение суммы к медиане типа",
    "adequacy_x_direction_rate": "адекватность × одобряемость направления",
    "adequacy_x_budget_pressure": "адекватность × бюджетное давление",
    "rule_score_feature": "оценка по правилам (фича)",
    "subsidy_type": "тип субсидии",
}


def _format_ml_factor_human(name: str) -> str:
    """Convert internal ML feature name to human-readable Russian."""
    return _ML_FACTOR_HUMAN_NAMES.get(name, name)


def _build_model_factor_details(model_explanation: dict[str, object] | None) -> list[FactorDetail]:
    if not model_explanation:
        return []

    factor_details: list[FactorDetail] = []
    for item in model_explanation.get("feature_effects", [])[:10]:
        contribution = float(item.get("score_impact", 0.0))
        abs_contribution = abs(contribution)
        if abs_contribution >= 8:
            level = "высокий"
        elif abs_contribution >= 3:
            level = "средний"
        else:
            level = "низкий"

        raw_value = _safe_feature_value(item.get("value"))
        feature_name = str(item.get("name"))
        human_label = _format_ml_factor_human(feature_name)
        factor_details.append(
            FactorDetail(
                name=feature_name,
                label=human_label,
                value=raw_value,
                contribution=round(contribution, 2),
                level=level,
            )
        )
    return factor_details


def _build_ml_explanation_lines(
    rule_score: float,
    ml_probability: float | None,
    ml_score: float | None,
    final_score: float,
    model_explanation: dict[str, object] | None = None,
) -> list[str]:
    if ml_probability is None or ml_score is None:
        return []

    weights = getattr(app.state, "blend_weights", DEFAULT_BLEND_WEIGHTS)

    lines = []

    # Probability description
    if ml_probability >= 0.8:
        prob_desc = "высоко оценивает шансы заявки"
    elif ml_probability >= 0.6:
        prob_desc = "оценивает шансы заявки как хорошие"
    elif ml_probability >= 0.4:
        prob_desc = "оценивает шансы заявки как средние"
    else:
        prob_desc = "оценивает шансы заявки как невысокие"

    lines.append(
        f"Нейросетевая модель {prob_desc} на одобрение — "
        f"вероятность составляет {ml_probability:.0%}, "
        f"что соответствует {ml_score:.1f} баллам из 100."
    )

    # Final score composition
    if weights["rule_score"] > 0 and weights["ml_score"] > 0:
        lines.append(
            f"Итоговый балл ({final_score:.1f}) складывается из оценки по правилам "
            f"({rule_score:.1f}) и оценки модели ({ml_score:.1f}) "
            f"в пропорции {weights['rule_score']:.0%} / {weights['ml_score']:.0%}."
        )
    elif weights["rule_score"] <= 0:
        lines.append(
            f"Итоговый балл ({final_score:.1f}) полностью определяется ML-моделью."
        )

    # ML feature effects — human-readable
    if model_explanation:
        top_positive = [
            item
            for item in model_explanation.get("feature_effects", [])
            if float(item.get("score_impact", 0.0)) > 0
        ][:3]
        top_negative = [
            item
            for item in model_explanation.get("feature_effects", [])
            if float(item.get("score_impact", 0.0)) < 0
        ][:3]
        if top_positive:
            positive_items = ", ".join(
                f"{_format_ml_factor_human(str(item['name']))} "
                f"(+{float(item['score_impact']):.1f} б.)"
                for item in top_positive
            )
            lines.append(
                f"✅ Модель положительно отметила: {positive_items}."
            )
        if top_negative:
            negative_items = ", ".join(
                f"{_format_ml_factor_human(str(item['name']))} "
                f"({float(item['score_impact']):.1f} б.)"
                for item in top_negative
            )
            lines.append(
                f"⚠ Модель обратила внимание на слабые стороны: {negative_items}."
            )
    return lines


def _build_history_explanation_lines(history_payload: dict[str, object]) -> list[str]:
    history_score = history_payload.get("history_advisory_score")
    history_recommendation = history_payload.get("history_recommendation")
    if history_score is None or history_recommendation is None:
        return []

    score_val = float(history_score)
    if score_val >= 70:
        assessment = "положительный"
    elif score_val >= 40:
        assessment = "умеренный"
    else:
        assessment = "настораживающий"

    lines = [
        f"По историческим данным похожих заявок — {assessment} сигнал "
        f"({history_recommendation}, {score_val:.0f}/100)."
    ]
    history_note = history_payload.get("history_note")
    if history_note:
        lines.append(str(history_note))
    return lines


def _build_single_rule_scores_frame(
    rule_result,
    eligibility_payload: dict[str, object],
) -> pd.DataFrame:
    payload = {
        "score": float(rule_result.score),
        "risk_level": str(rule_result.risk_level).capitalize(),
        "disqualified": bool(eligibility_payload["disqualified"]),
        "disqualification_reason": eligibility_payload["disqualification_reason"],
        "eligibility_status": str(eligibility_payload["eligibility_status"]),
        "manual_review_required": bool(eligibility_payload["manual_review_required"]),
        "eligibility_note": eligibility_payload["eligibility_note"],
        "normative_reference_found": bool(eligibility_payload["normative_reference_found"]),
    }
    for factor_name, contribution in rule_result.factors.items():
        payload[f"contrib_{factor_name}"] = float(contribution)
    return pd.DataFrame([payload])


def _score_single_payload(
    model_input: dict | pd.Series | pd.DataFrame,
    rule_result,
    eligibility_payload: dict[str, object],
    history_payload: dict[str, object],
) -> dict[str, object]:
    is_disqualified = bool(eligibility_payload["disqualified"])
    payload = {
        "score": 0.0 if is_disqualified else float(rule_result.score),
        "risk_level": "высокий" if is_disqualified else _single_risk_label(rule_result.risk_level),
        "rule_score": float(rule_result.score),
        "ml_score": None,
        "ml_probability": None,
        "disqualified": is_disqualified,
        "disqualification_reason": eligibility_payload["disqualification_reason"],
        "eligibility_status": str(eligibility_payload["eligibility_status"]),
        "manual_review_required": bool(eligibility_payload["manual_review_required"]),
        "eligibility_note": eligibility_payload["eligibility_note"],
        "normative_reference_found": bool(eligibility_payload["normative_reference_found"]),
        "history_match_source": history_payload["history_match_source"],
        "history_match_count": int(history_payload["history_match_count"]),
        "history_approval_rate": float(history_payload["history_approval_rate"]),
        "history_advisory_score": float(history_payload["history_advisory_score"]),
        "history_recommendation": history_payload["history_recommendation"],
        "history_note": history_payload["history_note"],
        "scoring_engine": _get_scoring_engine(),
        "model_name": _get_model_name(),
    }
    if payload["disqualified"]:
        return payload

    model_bundle = getattr(app.state, "model_bundle", None)
    if not model_bundle:
        return payload

    rule_scores_frame = _build_single_rule_scores_frame(
        rule_result=rule_result,
        eligibility_payload=eligibility_payload,
    )
    blended_row = score_features_with_model(
        features_input=model_input,
        model=model_bundle["model"],
        rule_scores=rule_scores_frame,
        blend_weights=model_bundle.get("blend_weights", DEFAULT_BLEND_WEIGHTS),
        decision_threshold=model_bundle.get("decision_threshold", 0.5),
        probability_calibrator=model_bundle.get("probability_calibrator"),
        probability_temperature=model_bundle.get("probability_temperature", 1.0),
    ).iloc[0]
    payload["score"] = float(blended_row["score"])
    payload["risk_level"] = _single_risk_label(blended_row["risk_level"])
    payload["rule_score"] = float(blended_row["rule_score"])
    payload["ml_score"] = float(blended_row["ml_score"])
    payload["ml_probability"] = float(blended_row["ml_probability"])
    payload["disqualified"] = bool(blended_row.get("disqualified", payload["disqualified"]))
    payload["disqualification_reason"] = (
        None
        if pd.isna(blended_row.get("disqualification_reason"))
        else str(blended_row.get("disqualification_reason"))
    )
    payload["eligibility_status"] = str(
        blended_row.get("eligibility_status", payload["eligibility_status"])
    )
    payload["manual_review_required"] = bool(
        blended_row.get("manual_review_required", payload["manual_review_required"])
    )
    payload["eligibility_note"] = (
        None
        if pd.isna(blended_row.get("eligibility_note"))
        else str(blended_row.get("eligibility_note"))
    )
    payload["normative_reference_found"] = bool(
        blended_row.get("normative_reference_found", payload["normative_reference_found"])
    )
    return payload


def _score_payload_from_index(idx: int, fallback_rule_result) -> dict[str, object]:
    scores = app.state.scores
    if idx not in scores.index:
        return {
            "score": float(fallback_rule_result.score),
            "risk_level": _single_risk_label(fallback_rule_result.risk_level),
            "rule_score": float(fallback_rule_result.score),
            "ml_score": None,
            "ml_probability": None,
            "disqualified": False,
            "disqualification_reason": None,
            "eligibility_status": "preliminarily_eligible",
            "manual_review_required": True,
            "eligibility_note": None,
            "normative_reference_found": True,
            "history_match_source": "global",
            "history_match_count": 0,
            "history_approval_rate": None,
            "history_advisory_score": None,
            "history_recommendation": None,
            "history_note": None,
            "scoring_engine": _get_scoring_engine(),
            "model_name": _get_model_name(),
        }

    row = scores.loc[idx]
    return {
        "score": float(row.get("score", fallback_rule_result.score)),
        "risk_level": _single_risk_label(
            row.get("risk_level", fallback_rule_result.risk_level)
        ),
        "rule_score": float(row.get("rule_score", fallback_rule_result.score)),
        "ml_score": _optional_float(row.get("ml_score")),
        "ml_probability": _optional_float(row.get("ml_probability")),
        "disqualified": bool(row.get("disqualified", False)),
        "disqualification_reason": (
            None if pd.isna(row.get("disqualification_reason")) else str(row.get("disqualification_reason"))
        ),
        "eligibility_status": str(row.get("eligibility_status", "preliminarily_eligible")),
        "manual_review_required": bool(row.get("manual_review_required", True)),
        "eligibility_note": (
            None if pd.isna(row.get("eligibility_note")) else str(row.get("eligibility_note"))
        ),
        "normative_reference_found": bool(row.get("normative_reference_found", True)),
        "history_match_source": str(row.get("history_match_source", "global")),
        "history_match_count": int(row.get("history_match_count", 0)),
        "history_approval_rate": _optional_float(row.get("history_approval_rate")),
        "history_advisory_score": _optional_float(row.get("history_advisory_score")),
        "history_recommendation": (
            None if pd.isna(row.get("history_recommendation")) else str(row.get("history_recommendation"))
        ),
        "history_note": (
            None if pd.isna(row.get("history_note")) else str(row.get("history_note"))
        ),
        "scoring_engine": _get_scoring_engine(),
        "model_name": _get_model_name(),
    }


def _load_dataset_into_state(app_instance: FastAPI, data_path: str) -> int:
    """Load dataset, run full pipeline, compute scores, update app.state."""
    logger.info("Загрузка данных из %s ...", data_path)
    df = run_pipeline(data_path)

    logger.info("Feature engineering v4.0 ...")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    advisory = build_history_advisory_batch(df)
    advisory_tables = build_history_advisory_tables(df)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])

    logger.info("Batch merit scoring v4.0 ...")
    rule_scores = score_batch(features)
    rule_scores[eligibility.columns] = eligibility
    model_input = build_primary_model_frame(
        raw_input=df,
        extracted_features=features,
        rule_scores=rule_scores,
    )
    model_bundle = getattr(app_instance.state, "model_bundle", None)
    if model_bundle is None:
        model_bundle = _load_model_bundle_if_available(MODEL_PATH)
    if model_bundle is None:
        raise RuntimeError(
            "ML bundle is required for API startup. "
            "Сначала обучите модель через `python train.py`."
        )

    scores = _prepare_api_scores(model_input, rule_scores, advisory, model_bundle)

    app_instance.state.df = df
    app_instance.state.tables = tables
    app_instance.state.features = features
    app_instance.state.model_input = model_input
    app_instance.state.history_df = df.reindex(
        columns=[
            "app_number",
            "region",
            "district",
            "direction",
            "subsidy_type",
            "akimat",
            "status",
            "normative",
            "normative_original",
            "amount",
            "submit_date",
            "submit_month",
            "is_approved",
        ]
    ).copy()
    app_instance.state.advisory = advisory
    app_instance.state.advisory_tables = advisory_tables
    app_instance.state.scores = scores
    app_instance.state.model_bundle = model_bundle
    app_instance.state.model_path = MODEL_PATH
    latest_submit_date = pd.to_datetime(df["submit_date"], errors="coerce").dropna()
    app_instance.state.default_request_year = (
        int(latest_submit_date.max().year)
        if not latest_submit_date.empty
        else datetime.now().year
    )
    app_instance.state.blend_weights = (
        model_bundle.get("blend_weights", DEFAULT_BLEND_WEIGHTS)
        if model_bundle
        else DEFAULT_BLEND_WEIGHTS
    )
    app_instance.state.dataset_name = Path(data_path).name

    logger.info(
        "Данные загружены: %d записей, средний балл %.1f, движок %s",
        len(df),
        scores["score"].mean(),
        _get_scoring_engine(),
    )
    return len(df)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_dataset_into_state(app, DATA_PATH)
    app.state.runtime_monitor = _init_runtime_monitor()
    app.state.started_at = datetime.now().isoformat()
    yield
    logger.info("Сервер остановлен.")


app = FastAPI(
    title="Subsidy Merit Scoring API v4.0",
    description=(
        "AI-система merit-скоринга заявок на субсидирование "
        "сельхозпроизводителей Казахстана. "
        "Primary ML использует нейросетевой merit scoring с time-aware feature engineering, "
        "сжатыми региональными priors, нормативными сигналами и ограниченным набором "
        "rule-derived признаков. Итоговый score строится как blend линейного rule-based "
        "score и neural-network score. Вероятности калибруются через validation-based "
        "calibrator без temperature-smoothing. Исторический реестр используется как "
        "advisory-слой и как источник time-causal признаков для новых заявок. "
        "Просроченные дедлайны отсекаются eligibility-фильтром до ML. "
        "Система является инструментом поддержки решения и не заменяет установленный "
        "Правилами порядок рассмотрения заявки."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_combined() -> pd.DataFrame:
    return pd.concat([app.state.df, app.state.scores], axis=1)


def _apply_filters(
    combined: pd.DataFrame,
    region: str | None = None,
    direction: str | None = None,
    subsidy_type: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    risk_level: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=combined.index)
    if region:
        mask &= combined["region"] == region
    if direction:
        mask &= combined["direction"] == direction
    if subsidy_type:
        mask &= combined["subsidy_type"] == subsidy_type
    if min_score is not None:
        mask &= combined["score"] >= min_score
    if max_score is not None:
        mask &= combined["score"] <= max_score
    if risk_level:
        mask &= combined["risk_level"].astype(str).str.lower() == risk_level.lower()
    return combined[mask]


def _make_app_brief(row: pd.Series) -> ApplicationBrief:
    return ApplicationBrief(
        app_number=str(row.get("app_number", "")),
        region=str(row.get("region", "")),
        district=str(row.get("district", "")),
        direction=str(row.get("direction", "")),
        subsidy_type=str(row.get("subsidy_type", "")),
        amount=float(row.get("amount", 0)),
        status=str(row.get("status", "")),
        score=float(row.get("score", 0)),
        risk_level=str(row.get("risk_level", "")),
        rule_score=_optional_float(row.get("rule_score")),
        ml_score=_optional_float(row.get("ml_score")),
        ml_probability=_optional_float(row.get("ml_probability")),
        history_match_source=str(row.get("history_match_source", "global")),
        history_match_count=int(row.get("history_match_count", 0)),
        history_approval_rate=_optional_float(row.get("history_approval_rate")),
        history_advisory_score=_optional_float(row.get("history_advisory_score")),
        history_recommendation=(
            None if pd.isna(row.get("history_recommendation")) else str(row.get("history_recommendation"))
        ),
        history_note=(
            None if pd.isna(row.get("history_note")) else str(row.get("history_note"))
        ),
        disqualified=bool(row.get("disqualified", False)),
        disqualification_reason=(
            None if pd.isna(row.get("disqualification_reason")) else str(row.get("disqualification_reason"))
        ),
        eligibility_status=str(row.get("eligibility_status", "preliminarily_eligible")),
        manual_review_required=bool(row.get("manual_review_required", True)),
        eligibility_note=(
            None if pd.isna(row.get("eligibility_note")) else str(row.get("eligibility_note"))
        ),
        normative_reference_found=bool(row.get("normative_reference_found", True)),
        scoring_engine=_get_scoring_engine(),
        model_name=_get_model_name(),
        top_factor=str(row.get("top_factor_label", "")),
    )


def _build_rule_factor_details(features_dict: dict, scoring_result) -> list[FactorDetail]:
    details = []
    sorted_factors = sorted(
        scoring_result.factors.items(), key=lambda x: x[1], reverse=True
    )
    for name, contribution in sorted_factors:
        value = features_dict.get(name, 0)
        if contribution >= 15:
            level = "высокий"
        elif contribution >= 8:
            level = "средний"
        else:
            level = "низкий"
        details.append(
            FactorDetail(
                name=name,
                label=FACTOR_LABELS.get(name, name),
                value=round(float(value), 4),
                contribution=contribution,
                level=level,
            )
        )
    return details


def _group_stats(column: str) -> list[dict]:
    combined = _get_combined()
    stats = (
        combined.groupby(column)["score"]
        .agg(count="size", avg_score="mean")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return [
        {
            column: row[column],
            "count": int(row["count"]),
            "avg_score": round(float(row["avg_score"]), 1),
        }
        for _, row in stats.iterrows()
    ]


# endpoints 


@app.get("/health", response_model=HealthResponse, tags=["Системные"])
async def health_check():
    model_bundle = getattr(app.state, "model_bundle", {}) or {}
    model_report = model_bundle.get("report", {})
    validation_metrics = model_report.get("best_validation_metrics") or (
        (model_report.get("model_validation_candidates") or [{}])[0]
        .get("validation_metrics", {})
    )
    blend_weights = model_bundle.get("blend_weights", DEFAULT_BLEND_WEIGHTS)
    return HealthResponse(
        status="ok",
        version=app.version,
        records_loaded=len(app.state.df),
        scoring_engine=_get_scoring_engine(),
        model_loaded=getattr(app.state, "model_bundle", None) is not None,
        model_name=_get_model_name(),
        model_path=app.state.model_path,
        started_at=getattr(app.state, "started_at", None),
        model_created_at=model_bundle.get("created_at"),
        calibration_method=model_bundle.get("calibration_method"),
        decision_threshold=_optional_float(model_bundle.get("decision_threshold")),
        blend_rule_weight=_optional_float(blend_weights.get("rule_score")),
        blend_ml_weight=_optional_float(blend_weights.get("ml_score")),
        test_roc_auc=_optional_float(model_report.get("test_metrics", {}).get("roc_auc")),
        validation_roc_auc=_optional_float(validation_metrics.get("roc_auc")),
        region_sensitivity_mean_delta=_optional_float(
            model_report.get("test_region_sensitivity", {}).get("mean_abs_delta")
        ),
        score_requests_total=int(getattr(app.state, "runtime_monitor", {}).get("score_requests_total", 0)),
        rank_requests_total=int(getattr(app.state, "runtime_monitor", {}).get("rank_requests_total", 0)),
        explain_requests_total=int(getattr(app.state, "runtime_monitor", {}).get("explain_requests_total", 0)),
        avg_score_latency_ms=_average_latency("score"),
        avg_rank_latency_ms=_average_latency("rank"),
        avg_explain_latency_ms=_average_latency("explain"),
        dataset_name=getattr(app.state, "dataset_name", None),
    )


UPLOAD_DATA_DIR = Path("data")

_retrain_status: dict[str, object] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _run_retrain(data_path: str):
    """Run train.py in background, then reload model + re-score."""
    global _retrain_status
    try:
        _retrain_status["status"] = "training"
        _retrain_status["started_at"] = datetime.now().isoformat()
        _retrain_status["finished_at"] = None
        _retrain_status["error"] = None

        logger.info("Запуск переобучения модели на %s ...", data_path)
        result = subprocess.run(
            [sys.executable, "train.py", "--data-path", data_path],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            logger.error("Ошибка переобучения: %s", error_msg)
            _retrain_status["status"] = "error"
            _retrain_status["error"] = error_msg
            _retrain_status["finished_at"] = datetime.now().isoformat()
            return

        logger.info("Переобучение завершено, перезагрузка модели...")
        new_bundle = _load_model_bundle_if_available(MODEL_PATH)
        if new_bundle is None:
            _retrain_status["status"] = "error"
            _retrain_status["error"] = "Модель не найдена после обучения"
            _retrain_status["finished_at"] = datetime.now().isoformat()
            return

        app.state.model_bundle = new_bundle
        _load_dataset_into_state(app, data_path)

        _retrain_status["status"] = "done"
        _retrain_status["finished_at"] = datetime.now().isoformat()
        logger.info("Модель успешно переобучена и загружена.")

    except subprocess.TimeoutExpired:
        _retrain_status["status"] = "error"
        _retrain_status["error"] = "Таймаут обучения (10 мин)"
        _retrain_status["finished_at"] = datetime.now().isoformat()
    except Exception as e:
        logger.exception("Ошибка при переобучении: %s", e)
        _retrain_status["status"] = "error"
        _retrain_status["error"] = str(e)
        _retrain_status["finished_at"] = datetime.now().isoformat()


@app.post("/upload-dataset", tags=["Системные"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload .xlsx dataset, score with current model, auto-trigger retraining."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате .xlsx")

    UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DATA_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        count = _load_dataset_into_state(app, str(save_path))
        app.state.current_data_path = str(save_path)

        if _retrain_status["status"] != "training":
            thread = threading.Thread(target=_run_retrain, args=(str(save_path),), daemon=True)
            thread.start()

        return {
            "status": "ok",
            "records_loaded": count,
            "dataset_name": file.filename,
            "retrain_started": True,
        }
    except Exception as e:
        logger.exception("Ошибка при загрузке датасета: %s", e)
        raise HTTPException(status_code=400, detail=f"Ошибка обработки файла: {e}")


@app.post("/retrain", tags=["Системные"])
async def retrain_model():
    """Trigger model retraining on current dataset in background."""
    if _retrain_status["status"] == "training":
        raise HTTPException(status_code=409, detail="Обучение уже запущено")
    data_path = getattr(app.state, "current_data_path", DATA_PATH)
    thread = threading.Thread(target=_run_retrain, args=(data_path,), daemon=True)
    thread.start()
    return {"status": "started", "data_path": data_path}


@app.get("/retrain-status", tags=["Системные"])
async def retrain_status():
    """Check retraining status."""
    return dict(_retrain_status)


@app.post("/score", response_model=ScoreResponse, tags=["Скоринг"])
async def score_application(request: ScoreRequest):
    start_time = perf_counter()
    # скоринг одной заявки, норматив подставляется из справочника автоматически
    norm_lookup = app.state.tables["normative_lookup"]
    ref_norm = get_normative_for_type(request.subsidy_type, norm_lookup)

    # строим submit_date из месяца и дня
    try:
        submit_date = datetime(
            int(getattr(app.state, "default_request_year", datetime.now().year)),
            request.submit_month,
            request.submit_day,
        )
    except ValueError:
        submit_date = datetime(
            int(getattr(app.state, "default_request_year", datetime.now().year)),
            request.submit_month,
            15,
        )

    row = pd.Series(
        {
            "region": request.region,
            "direction": request.direction,
            "subsidy_type": request.subsidy_type,
            "district": request.district,
            "akimat": request.akimat,
            "amount": request.amount,
            "normative": ref_norm or 0,
            "submit_date": submit_date,
            "submit_month": request.submit_month,
        }
    )

    features_dict = extract_features_single_with_history(
        row=row,
        history_df=app.state.history_df,
        normative_lookup=norm_lookup,
    )
    eligibility_payload = evaluate_single_eligibility(row, norm_lookup)
    history_payload = build_history_advisory_single_from_tables(
        row,
        app.state.advisory_tables,
    )
    rule_result = score_single(features_dict)
    single_rule_scores = _build_single_rule_scores_frame(
        rule_result=rule_result,
        eligibility_payload=eligibility_payload,
    )
    model_input = build_primary_model_frame(
        raw_input=row,
        extracted_features=features_dict,
        rule_scores=single_rule_scores,
    )
    score_payload = _score_single_payload(
        model_input=model_input,
        rule_result=rule_result,
        eligibility_payload=eligibility_payload,
        history_payload=history_payload,
    )
    model_bundle = getattr(app.state, "model_bundle", None)
    model_explanation = None
    if model_bundle and not score_payload["disqualified"]:
        model_explanation = explain_prediction_with_model(
            features_input=model_input,
            model=model_bundle["model"],
            neutral_values=model_bundle.get("explanation_neutral_values"),
            probability_calibrator=model_bundle.get("probability_calibrator"),
            probability_temperature=model_bundle.get("probability_temperature"),
        )
    explanation = [DECISION_SUPPORT_NOTE]
    if score_payload["disqualified"] and score_payload["disqualification_reason"]:
        explanation.insert(
            0,
            f"✗ Заявка автоматически отклонена до ML-скоринга: {score_payload['disqualification_reason']}.",
        )
    elif score_payload["eligibility_note"]:
        explanation.append(str(score_payload["eligibility_note"]))
    explanation.extend(
        _build_ml_explanation_lines(
            rule_score=score_payload["rule_score"],
            ml_probability=score_payload["ml_probability"],
            ml_score=score_payload["ml_score"],
            final_score=score_payload["score"],
            model_explanation=model_explanation,
        )
    )
    explanation.extend(_build_history_explanation_lines(score_payload))
    explanation.append("Детальная оценка по критериям:")
    explanation.extend(rule_result.explanation)
    factor_details = _build_rule_factor_details(features_dict, rule_result)

    response = ScoreResponse(
        score=score_payload["score"],
        risk_level=score_payload["risk_level"],
        rule_score=score_payload["rule_score"],
        ml_score=score_payload["ml_score"],
        ml_probability=score_payload["ml_probability"],
        history_match_source=score_payload["history_match_source"],
        history_match_count=score_payload["history_match_count"],
        history_approval_rate=score_payload["history_approval_rate"],
        history_advisory_score=score_payload["history_advisory_score"],
        history_recommendation=score_payload["history_recommendation"],
        history_note=score_payload["history_note"],
        disqualified=score_payload["disqualified"],
        disqualification_reason=score_payload["disqualification_reason"],
        eligibility_status=score_payload["eligibility_status"],
        manual_review_required=score_payload["manual_review_required"],
        eligibility_note=score_payload["eligibility_note"],
        normative_reference_found=score_payload["normative_reference_found"],
        scoring_engine=score_payload["scoring_engine"],
        model_name=score_payload["model_name"],
        factors=factor_details,
        explanation=explanation,
    )
    _record_runtime_event("score", (perf_counter() - start_time) * 1000)
    return response


@app.post("/rank", response_model=RankResponse, tags=["Скоринг"])
async def rank_applications(request: RankRequest):
    start_time = perf_counter()
    combined = _get_combined()
    filtered = _apply_filters(
        combined,
        region=request.region,
        direction=request.direction,
        subsidy_type=request.subsidy_type,
        min_score=request.min_score,
        max_score=request.max_score,
        risk_level=request.risk_level,
    ).sort_values("score", ascending=False)

    total_filtered = len(filtered)
    top = filtered.head(request.top_n)

    response = RankResponse(
        total_filtered=total_filtered,
        returned=len(top),
        scoring_engine=_get_scoring_engine(),
        model_name=_get_model_name(),
        applications=[_make_app_brief(row) for _, row in top.iterrows()],
    )
    _record_runtime_event("rank", (perf_counter() - start_time) * 1000)
    return response


@app.get("/explain/{app_id}", response_model=ExplainResponse, tags=["Скоринг"])
async def explain_score(app_id: str):
    start_time = perf_counter()
    df = app.state.df
    match = df[df["app_number"].astype(str) == app_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Заявка {app_id} не найдена")

    idx = match.index[0]
    row = match.iloc[0]
    if idx in app.state.features.index:
        features_dict = app.state.features.loc[idx].to_dict()
    else:
        features_dict = extract_features_single_with_history(
            row=row,
            history_df=app.state.history_df,
            normative_lookup=app.state.tables["normative_lookup"],
        )

    rule_result = score_single(features_dict)
    score_payload = _score_payload_from_index(idx, rule_result)
    model_bundle = getattr(app.state, "model_bundle", None)
    model_explanation = None
    if model_bundle and not score_payload["disqualified"]:
        if idx in app.state.model_input.index:
            explanation_features = app.state.model_input.loc[[idx]]
        else:
            explanation_features = build_primary_model_frame(
                raw_input=row,
                extracted_features=features_dict,
                rule_scores=_build_single_rule_scores_frame(
                    rule_result=rule_result,
                    eligibility_payload={
                        "disqualified": score_payload["disqualified"],
                        "disqualification_reason": score_payload["disqualification_reason"],
                        "eligibility_status": score_payload["eligibility_status"],
                        "manual_review_required": score_payload["manual_review_required"],
                        "eligibility_note": score_payload["eligibility_note"],
                        "normative_reference_found": score_payload["normative_reference_found"],
                    },
                ),
            )
        model_explanation = explain_prediction_with_model(
            features_input=explanation_features,
            model=model_bundle["model"],
            neutral_values=model_bundle.get("explanation_neutral_values"),
            probability_calibrator=model_bundle.get("probability_calibrator"),
            probability_temperature=model_bundle.get("probability_temperature"),
        )
    explanation = [DECISION_SUPPORT_NOTE]
    if score_payload["disqualified"] and score_payload["disqualification_reason"]:
        explanation.insert(
            0,
            f"✗ Заявка автоматически отклонена до ML-скоринга: {score_payload['disqualification_reason']}.",
        )
    elif score_payload["eligibility_note"]:
        explanation.append(str(score_payload["eligibility_note"]))
    explanation.extend(
        _build_ml_explanation_lines(
            rule_score=score_payload["rule_score"],
            ml_probability=score_payload["ml_probability"],
            ml_score=score_payload["ml_score"],
            final_score=score_payload["score"],
            model_explanation=model_explanation,
        )
    )
    explanation.extend(_build_history_explanation_lines(score_payload))
    explanation.append("Детальная оценка по критериям:")
    explanation.extend(rule_result.explanation)
    factor_details = _build_rule_factor_details(features_dict, rule_result)
    ml_factor_details = _build_model_factor_details(model_explanation)

    response = ExplainResponse(
        app_number=str(row["app_number"]),
        region=str(row["region"]),
        direction=str(row["direction"]),
        subsidy_type=str(row["subsidy_type"]),
        amount=float(row["amount"]),
        status=str(row["status"]),
        score=score_payload["score"],
        risk_level=score_payload["risk_level"],
        rule_score=score_payload["rule_score"],
        ml_score=score_payload["ml_score"],
        ml_probability=score_payload["ml_probability"],
        history_match_source=score_payload["history_match_source"],
        history_match_count=score_payload["history_match_count"],
        history_approval_rate=score_payload["history_approval_rate"],
        history_advisory_score=score_payload["history_advisory_score"],
        history_recommendation=score_payload["history_recommendation"],
        history_note=score_payload["history_note"],
        disqualified=score_payload["disqualified"],
        disqualification_reason=score_payload["disqualification_reason"],
        eligibility_status=score_payload["eligibility_status"],
        manual_review_required=score_payload["manual_review_required"],
        eligibility_note=score_payload["eligibility_note"],
        normative_reference_found=score_payload["normative_reference_found"],
        scoring_engine=score_payload["scoring_engine"],
        model_name=score_payload["model_name"],
        factors=factor_details,
        ml_factors=ml_factor_details,
        explanation=explanation,
    )
    _record_runtime_event("explain", (perf_counter() - start_time) * 1000)
    return response


@app.get("/stats", response_model=StatsResponse, tags=["Аналитика"])
async def get_stats(
    region: str | None = Query(None, description="Фильтр по региону"),
    direction: str | None = Query(None, description="Фильтр по направлению"),
    subsidy_type: str | None = Query(None, description="Фильтр по типу субсидии"),
    risk_level: str | None = Query(None, description="Фильтр по уровню риска"),
    min_score: float | None = Query(None, ge=0, le=100, description="Минимальный балл"),
    max_score: float | None = Query(None, ge=0, le=100, description="Максимальный балл"),
):
    combined = _get_combined()
    filtered = _apply_filters(
        combined, region=region, direction=direction,
        subsidy_type=subsidy_type, risk_level=risk_level,
        min_score=min_score, max_score=max_score,
    )
    filtered_scores = app.state.scores.loc[filtered.index]

    dist = get_score_distribution(filtered_scores)

    # статистика по регионам
    region_stats = (
        filtered.groupby("region")["score"]
        .agg(count="size", avg_score="mean")
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    ar = app.state.tables["approval_rates"]["region"]
    top_regions = [
        RegionStat(
            region=r["region"],
            count=int(r["count"]),
            avg_score=round(float(r["avg_score"]), 1),
            approval_rate=round(float(ar.get(r["region"], {}).get("approval_rate", 0)), 4),
        )
        for _, r in region_stats.head(10).iterrows()
    ]

    risk_dict = {str(k): int(v) for k, v in dist["risk_distribution"].items()}

    return StatsResponse(
        total_records=len(filtered),
        mean_score=dist["mean"],
        median_score=dist["median"],
        std_score=dist["std"],
        min_score=dist["min"],
        max_score=dist["max"],
        risk_distribution=risk_dict,
        scoring_engine=_get_scoring_engine(),
        model_name=_get_model_name(),
        top_regions=top_regions,
    )


@app.get("/applications", response_model=PaginatedApplications, tags=["Заявки"])
async def list_applications(
    page: int = Query(1, ge=1, description="Номер страницы"),
    per_page: int = Query(20, ge=1, le=100, description="Заявок на странице"),
    region: str | None = Query(None, description="Фильтр по региону"),
    direction: str | None = Query(None, description="Фильтр по направлению"),
    min_score: float | None = Query(None, ge=0, le=100, description="Минимальный балл"),
    max_score: float | None = Query(None, ge=0, le=100, description="Максимальный балл"),
    sort_by: str = Query("score", description="Поле сортировки: score, amount"),
    sort_order: str = Query("desc", description="Порядок: asc / desc"),
):
    combined = _get_combined()
    filtered = _apply_filters(
        combined, region=region, direction=direction,
        min_score=min_score, max_score=max_score,
    )

    ascending = sort_order.lower() == "asc"
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    total = len(filtered)
    start = (page - 1) * per_page
    page_data = filtered.iloc[start:start + per_page]

    return PaginatedApplications(
        total=total,
        page=page,
        per_page=per_page,
        scoring_engine=_get_scoring_engine(),
        model_name=_get_model_name(),
        applications=[_make_app_brief(row) for _, row in page_data.iterrows()],
    )


@app.get("/applications/{app_id}", response_model=ApplicationBrief, tags=["Заявки"])
async def get_application(app_id: str):
    df = app.state.df
    match = df[df["app_number"].astype(str) == app_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Заявка {app_id} не найдена")

    idx = match.index[0]
    combined = _get_combined()
    return _make_app_brief(combined.loc[idx])


@app.get("/factor-stats", tags=["Аналитика"])
async def get_factor_stats(
    region: str | None = Query(None),
    direction: str | None = Query(None),
    subsidy_type: str | None = Query(None),
    min_score: float | None = Query(None, ge=0, le=100),
    max_score: float | None = Query(None, ge=0, le=100),
):
    """Средние значения каждого фактора (для графиков распределения)."""
    combined = _get_combined()
    filtered = _apply_filters(
        combined, region=region, direction=direction,
        subsidy_type=subsidy_type,
        min_score=min_score, max_score=max_score,
    )
    features = app.state.features.loc[filtered.index]

    result = {}
    for factor in FACTOR_LABELS:
        if factor in features.columns:
            col = features[factor]
            std_val = col.std()
            result[factor] = {
                "label": FACTOR_LABELS[factor],
                "mean": round(float(col.mean()), 4),
                "median": round(float(col.median()), 4),
                "std": round(float(std_val), 4) if pd.notna(std_val) else 0.0,
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
                "values": col.round(3).tolist(),
            }
    return result


@app.get("/region-factors", tags=["Аналитика"])
async def get_region_factors(
    direction: str | None = Query(None),
    subsidy_type: str | None = Query(None),
):
    """Средние значения факторов в разрезе регионов."""
    combined = _get_combined()
    filtered = _apply_filters(combined, direction=direction, subsidy_type=subsidy_type)
    features = app.state.features.loc[filtered.index]

    # добавляем регион к features
    feat_with_region = features.copy()
    feat_with_region["region"] = filtered["region"]

    factor_cols = [c for c in FACTOR_LABELS if c in features.columns]
    grouped = feat_with_region.groupby("region")[factor_cols].mean()

    result = []
    for region, row in grouped.iterrows():
        factors = {}
        for col in factor_cols:
            factors[col] = {
                "label": FACTOR_LABELS[col],
                "mean": round(float(row[col]), 4),
            }
        result.append({
            "region": region,
            "count": int((filtered["region"] == region).sum()),
            "factors": factors,
        })

    return sorted(result, key=lambda x: x["count"], reverse=True)


@app.get("/timeline", tags=["Аналитика"])
async def get_timeline(
    region: str | None = Query(None),
    direction: str | None = Query(None),
):
    """Динамика бюджетного давления и очереди по месяцам."""
    combined = _get_combined()
    filtered = _apply_filters(combined, region=region, direction=direction)
    features = app.state.features.loc[filtered.index]

    df_timeline = pd.DataFrame({
        "submit_month": filtered["submit_month"],
        "budget_pressure": features["budget_pressure"] if "budget_pressure" in features.columns else 0,
        "queue_position": features["queue_position"] if "queue_position" in features.columns else 0,
        "score": filtered["score"],
    })

    monthly = df_timeline.groupby("submit_month").agg(
        count=("score", "size"),
        avg_score=("score", "mean"),
        avg_budget_pressure=("budget_pressure", "mean"),
        avg_queue_position=("queue_position", "mean"),
    ).reset_index()

    monthly = monthly.sort_values("submit_month")

    return [
        {
            "month": int(row["submit_month"]),
            "count": int(row["count"]),
            "avg_score": round(float(row["avg_score"]), 1),
            "avg_budget_pressure": round(float(row["avg_budget_pressure"]), 4),
            "avg_queue_position": round(float(row["avg_queue_position"]), 4),
        }
        for _, row in monthly.iterrows()
    ]


@app.get("/regions", tags=["Справочники"])
async def list_regions():
    return _group_stats("region")


@app.get("/directions", tags=["Справочники"])
async def list_directions():
    return _group_stats("direction")
