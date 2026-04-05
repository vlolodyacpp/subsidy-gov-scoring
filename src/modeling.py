from __future__ import annotations

import inspect
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    CatBoostClassifier = None
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.advisory import build_history_advisory_batch
from src.eligibility import evaluate_batch_eligibility
from src.features import build_feature_tables, extract_features_batch
from src.pipeline import run_pipeline
from src.scoring import (
    DEADLINE_DISQUALIFICATION_REASON,
    WEIGHTS,
    score_batch,
)


SYNTHETIC_FEATURES_PATH = "data/cleaned/synthetic_features.csv"
RULE_FEATURE_COLUMNS = list(WEIGHTS.keys())
RULE_CONTRIBUTION_COLUMNS = [f"contrib_{column}" for column in RULE_FEATURE_COLUMNS]
PRIMARY_MODEL_CATEGORICAL_COLUMNS = [
    "subsidy_type",
]
SAFE_SYNTHETIC_FEATURE_COLUMNS = [
    "criteria_complexity",
    "direction_risk",
    "regional_pasture_capacity",
]
LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS = [
    "pasture_compliance",
    "mortality_compliance",
    "grazing_utilization",
    "actual_pasture_load",
    "actual_mortality_pct",
    "actual_grazing_days",
]
LEAKY_RULE_SYNTHETIC_FEATURE_COLUMNS = [
    "pasture_compliance",
    "mortality_compliance",
    "grazing_utilization",
]
EXCLUDED_COLUMNS = [
    "app_number",
    "row_id",
    "date_str",
    "submit_date",
    "status",
    "is_approved",
    "training_target",
    "historical_is_approved",
    "pasture_norm",
    "grazing_days",
    "mortality_mean",
    "mortality_max",
    "avg_criteria_count",
] + LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS
ALL_SYNTHETIC_FEATURE_COLUMNS = [
    "pasture_norm",
    "grazing_days",
    "mortality_mean",
    "mortality_max",
    "avg_criteria_count",
] + SAFE_SYNTHETIC_FEATURE_COLUMNS + LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS
SYNTHETIC_RULE_FEATURE_COLUMNS = [
    "pasture_compliance",
    "mortality_compliance",
    "grazing_utilization",
    "criteria_complexity",
    "direction_risk",
    "regional_pasture_capacity",
]
EXISTING_ONLY_FEATURE_SET = "existing_only"
EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET = "existing_plus_synthetic_safe"
OFFLINE_EXPERIMENTAL_FEATURE_SET = EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET
FEATURE_SET_ORDER = [
    EXISTING_ONLY_FEATURE_SET,
    EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET,
]
FEATURE_SET_DEPLOYABILITY = {
    EXISTING_ONLY_FEATURE_SET: {
        "deployable": True,
        "reason": (
            "Признаки строятся из raw request, нормативного lookup и исторических "
            "time-causal tables, поэтому могут быть честно воспроизведены для новой заявки."
        ),
    },
    EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET: {
        "deployable": False,
        "reason": (
            "Feature set использует offline-only synthetic condition signals, которые "
            "подтягиваются через cleaned `synthetic_features.csv` по `app_number` и не "
            "могут быть честно восстановлены для новой заявки без дополнительных фактов."
        ),
    },
}
RAW_SIGNAL_COLUMNS = [
    "normative_log",
    "amount_log",
    "amount_adequacy",
    "normative_match",
    "normative_original_match",
    "normative_reference_gap",
    "normative_reference_typicality",
    "amount_normative_integrity",
    "unit_count_log",
    "unit_count_original_log",
    "submit_month_sin",
    "submit_month_cos",
    "region_specialization",
    "region_direction_approval_rate",
    "akimat_approval_rate",
    "direction_approval_rate",
    "subsidy_type_approval_rate",
    "region_approval_rate",
    "region_direction_lift",
    "akimat_lift",
    "direction_history_count_log",
    "subsidy_type_history_count_log",
    "region_direction_history_count_log",
    "akimat_history_count_log",
    "budget_pressure",
    "queue_position",
    "amount_to_normative_ratio",
    "amount_to_type_median_ratio",
]
RULE_SCORE_AUGMENTATION_COLUMNS = [
    "rule_score_feature",
    "rule_score",
]
INTERACTION_FEATURE_COLUMNS = [
    "adequacy_x_direction_rate",
    "adequacy_x_budget_pressure",
    "criteria_complexity_x_subsidy_type_rate",
    "direction_risk_x_mortality_compliance",
    "rule_score_x_budget_pressure",
    "amount_log_x_rule_score",
]
PRIMARY_MODEL_NUMERIC_COLUMNS = list(
    dict.fromkeys(
        RAW_SIGNAL_COLUMNS
        + RULE_FEATURE_COLUMNS
        + RULE_SCORE_AUGMENTATION_COLUMNS
        + RULE_CONTRIBUTION_COLUMNS
        + INTERACTION_FEATURE_COLUMNS
    )
)
FEATURE_COLUMNS = PRIMARY_MODEL_CATEGORICAL_COLUMNS + PRIMARY_MODEL_NUMERIC_COLUMNS
FEATURE_SET_COLUMNS = {
    EXISTING_ONLY_FEATURE_SET: FEATURE_COLUMNS,
    EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET: FEATURE_COLUMNS,
}
DEFAULT_FEATURE_SET_NAME = EXISTING_ONLY_FEATURE_SET
MERIT_PROXY_FEATURE_COLUMNS = [
    "amount_adequacy",
    "amount_to_normative_ratio",
    "unit_count_log",
]
PROCESS_BIASED_FEATURE_COLUMNS: list[str] = list(LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS)
MIN_RULE_BLEND_WEIGHT = 0.25
DEFAULT_BLEND_WEIGHTS = {"rule_score": 0.25, "ml_score": 0.75}
DEFAULT_DECISION_SCORE_NAME = "final_score"
DEFAULT_DECISION_THRESHOLD = 50.0
DECISION_SCORE_SCALE_MAX = 100.0
BLEND_WEIGHT_SEARCH_GRID = [
    round(float(rule_weight), 2)
    for rule_weight in np.linspace(
        MIN_RULE_BLEND_WEIGHT,
        1.0,
        int((1.0 - MIN_RULE_BLEND_WEIGHT) / 0.05) + 1,
    )
]
MERIT_PROXY_POSITIVE_THRESHOLD = 0.68
REGIONAL_SIGNAL_SHRINK = 0.35
RULE_AUGMENTATION_COLUMNS = list(RULE_CONTRIBUTION_COLUMNS)


def resolve_blend_weights(
    blend_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if blend_weights is None:
        resolved = dict(DEFAULT_BLEND_WEIGHTS)
    else:
        resolved = {
            "rule_score": float(blend_weights.get("rule_score", 0.0)),
            "ml_score": float(blend_weights.get("ml_score", 0.0)),
        }
        total = float(resolved["rule_score"] + resolved["ml_score"])
        if total <= 0:
            raise ValueError("Blend weights must sum to a positive value.")
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Blend weights must sum to 1.0, got {resolved['rule_score']} + "
                f"{resolved['ml_score']} = {total}"
            )

    return {
        "rule_score": round(float(resolved["rule_score"]), 4),
        "ml_score": round(float(resolved["ml_score"]), 4),
    }


def compute_blended_scores(
    rule_scores: pd.Series | np.ndarray,
    ml_probabilities: pd.Series | np.ndarray,
    blend_weights: dict[str, float] | None = None,
    disqualified_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> np.ndarray:
    resolved_weights = resolve_blend_weights(blend_weights)
    rule_arr = np.asarray(rule_scores).astype(float)
    ml_score_arr = np.asarray(ml_probabilities).astype(float) * 100.0
    blended = np.round(
        rule_arr * resolved_weights["rule_score"]
        + ml_score_arr * resolved_weights["ml_score"],
        1,
    )
    if disqualified_mask is not None:
        disqualified_arr = np.asarray(disqualified_mask).astype(bool)
        blended[disqualified_arr] = 0.0
    return blended


def resolve_score_scale_max(
    y_score: pd.Series | np.ndarray,
    score_scale_max: float | None = None,
) -> float:
    if score_scale_max is not None:
        return float(score_scale_max)

    score_arr = np.asarray(y_score).astype(float)
    finite = score_arr[np.isfinite(score_arr)]
    if finite.size == 0:
        return 1.0
    if float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0 + 1e-8:
        return 1.0
    return DECISION_SCORE_SCALE_MAX


def build_threshold_candidates(
    y_score: pd.Series | np.ndarray,
    score_scale_max: float | None = None,
) -> np.ndarray:
    resolved_scale = resolve_score_scale_max(y_score, score_scale_max)
    score_arr = np.asarray(y_score).astype(float)
    finite = score_arr[np.isfinite(score_arr)]
    if finite.size == 0:
        return np.array([resolved_scale / 2.0], dtype=float)

    if resolved_scale <= 1.0 + 1e-8:
        base = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        sweep = np.linspace(0.05, 0.95, 37)
    else:
        base = np.arange(5.0, resolved_scale, 5.0)
        sweep = np.linspace(5.0, resolved_scale - 5.0, 37)

    quantiles = np.quantile(finite, np.linspace(0.05, 0.95, 19))
    thresholds = np.unique(
        np.concatenate([base, sweep, quantiles]).round(4)
    )
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < resolved_scale)]
    if thresholds.size == 0:
        thresholds = np.array([resolved_scale / 2.0], dtype=float)
    return thresholds.astype(float)


def resolve_feature_set_name(feature_set_name: str | None = None) -> str:
    resolved = feature_set_name or DEFAULT_FEATURE_SET_NAME
    if resolved not in FEATURE_SET_COLUMNS:
        raise ValueError(f"Unknown feature_set_name: {resolved}")
    return resolved


def get_feature_columns(feature_set_name: str | None = None) -> list[str]:
    resolved = resolve_feature_set_name(feature_set_name)
    return list(FEATURE_SET_COLUMNS[resolved])


def get_categorical_feature_columns(feature_set_name: str | None = None) -> list[str]:
    resolve_feature_set_name(feature_set_name)
    return list(PRIMARY_MODEL_CATEGORICAL_COLUMNS)


def get_numeric_feature_columns(feature_set_name: str | None = None) -> list[str]:
    resolved = resolve_feature_set_name(feature_set_name)
    return [
        column
        for column in FEATURE_SET_COLUMNS[resolved]
        if column not in PRIMARY_MODEL_CATEGORICAL_COLUMNS
    ]


def get_feature_set_deployability(
    feature_set_name: str | None = None,
) -> dict[str, object]:
    resolved = resolve_feature_set_name(feature_set_name)
    deployability = FEATURE_SET_DEPLOYABILITY.get(
        resolved,
        {
            "deployable": False,
            "reason": "Deployability is not defined for this feature set.",
        },
    )
    return {
        "feature_set_name": resolved,
        "deployable": bool(deployability.get("deployable", False)),
        "reason": str(deployability.get("reason", "")),
    }


def _normalise_app_number(series_like: pd.Series | np.ndarray) -> pd.Series:
    normalised = pd.Series(series_like).fillna("").astype(str).str.strip()
    normalised = normalised.str.lstrip("0")
    return normalised.replace("", "0")


def load_synthetic_feature_table(
    path: str = SYNTHETIC_FEATURES_PATH,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    path_obj = Path(path)
    info: dict[str, object] = {
        "path": str(path_obj),
        "loaded": False,
        "available_columns": [],
        "safe_available_columns": [],
        "leaky_available_columns": [],
        "excluded_available_columns": [],
        "missing_recognized_columns": list(ALL_SYNTHETIC_FEATURE_COLUMNS),
        "row_count": 0,
    }
    if not path_obj.exists():
        return None, info

    synthetic = pd.read_csv(path_obj)
    if "app_number" not in synthetic.columns:
        raise ValueError("Synthetic feature table must contain 'app_number'")

    available_columns = [
        column for column in ALL_SYNTHETIC_FEATURE_COLUMNS
        if column in synthetic.columns
    ]
    keep_columns = ["app_number"] + available_columns
    synthetic = synthetic[keep_columns].copy()
    synthetic["app_number"] = _normalise_app_number(synthetic["app_number"])
    synthetic = synthetic.drop_duplicates(subset="app_number", keep="last")

    info["loaded"] = True
    info["available_columns"] = available_columns
    info["safe_available_columns"] = [
        column for column in SAFE_SYNTHETIC_FEATURE_COLUMNS
        if column in available_columns
    ]
    info["leaky_available_columns"] = [
        column for column in LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS
        if column in available_columns
    ]
    info["excluded_available_columns"] = [
        column for column in EXCLUDED_COLUMNS
        if column in synthetic.columns
    ]
    info["missing_recognized_columns"] = [
        column for column in ALL_SYNTHETIC_FEATURE_COLUMNS
        if column not in available_columns
    ]
    info["row_count"] = int(len(synthetic))
    return synthetic, info


def merge_synthetic_features(
    raw_input: dict | pd.Series | pd.DataFrame,
    extracted_features: pd.DataFrame,
    synthetic_table: pd.DataFrame | None = None,
    synthetic_info: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_frame = _coerce_dataframe(raw_input)
    feature_frame = extracted_features.copy()
    info = dict(synthetic_info or {})
    info.setdefault("merged_columns", [])
    info.setdefault("already_present_columns", [])

    if synthetic_table is None:
        synthetic_table, loaded_info = load_synthetic_feature_table()
        info = {**loaded_info, **info}

    if synthetic_table is None or "app_number" not in raw_frame.columns:
        info.setdefault("merge_skipped_reason", None)
        if synthetic_table is None:
            info["merge_skipped_reason"] = "synthetic_table_unavailable"
        else:
            info["merge_skipped_reason"] = "app_number_missing"
        return feature_frame, info

    app_key = _normalise_app_number(raw_frame["app_number"])
    indexed = synthetic_table.set_index("app_number")
    for column in [
        candidate
        for candidate in ALL_SYNTHETIC_FEATURE_COLUMNS
        if candidate in indexed.columns
    ]:
        mapped = pd.to_numeric(app_key.map(indexed[column]), errors="coerce")
        if column in feature_frame.columns:
            info["already_present_columns"].append(column)
            feature_frame[column] = (
                pd.to_numeric(feature_frame[column], errors="coerce")
                .fillna(mapped)
            )
        else:
            feature_frame[column] = mapped
            info["merged_columns"].append(column)

    return feature_frame, info


def _supports_sample_weight(model: object) -> bool:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
    try:
        signature = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False
    return "sample_weight" in signature.parameters


def _fit_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> object:
    fit_kwargs: dict[str, object] = {}
    if sample_weight is not None and _supports_sample_weight(model):
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            fit_kwargs["model__sample_weight"] = np.asarray(sample_weight)
        else:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight)

    fitted_model = deepcopy(model)
    fitted_model.fit(X, y, **fit_kwargs)
    return fitted_model


def _coerce_dataframe(data: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return pd.DataFrame([data.to_dict()])
    return pd.DataFrame([data])


def _coerce_feature_dataframe(
    extracted_features: dict | pd.Series | pd.DataFrame | None,
    index: pd.Index,
) -> pd.DataFrame:
    if extracted_features is None:
        return pd.DataFrame(index=index)

    feature_frame = _coerce_dataframe(extracted_features)
    if len(feature_frame) == len(index):
        feature_frame.index = index
        return feature_frame

    if len(feature_frame) == 1 and len(index) == 1:
        feature_frame.index = index
        return feature_frame

    return feature_frame.reindex(index=index)


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


def prepare_extracted_features_for_feature_set(
    extracted_features: dict | pd.Series | pd.DataFrame,
    feature_set_name: str = DEFAULT_FEATURE_SET_NAME,
) -> pd.DataFrame:
    resolved_feature_set = resolve_feature_set_name(feature_set_name)
    feature_frame = _coerce_dataframe(extracted_features).copy()

    for column_name in RULE_FEATURE_COLUMNS:
        if column_name not in feature_frame.columns:
            feature_frame[column_name] = 0.5

    for column_name in SAFE_SYNTHETIC_FEATURE_COLUMNS:
        feature_frame[column_name] = (
            pd.to_numeric(
                feature_frame[column_name]
                if column_name in feature_frame.columns
                else pd.Series(0.5, index=feature_frame.index),
                errors="coerce",
            )
            .fillna(0.5)
            .clip(lower=0.0, upper=1.0)
            .astype(float)
        )

    for column_name in LEAKY_RULE_SYNTHETIC_FEATURE_COLUMNS:
        if resolved_feature_set == EXISTING_ONLY_FEATURE_SET:
            feature_frame[column_name] = 0.5
        else:
            feature_frame[column_name] = (
                pd.to_numeric(
                    feature_frame[column_name]
                    if column_name in feature_frame.columns
                    else pd.Series(0.5, index=feature_frame.index),
                    errors="coerce",
                )
                .fillna(0.5)
                .clip(lower=0.0, upper=1.0)
                .astype(float)
            )

    return feature_frame


def build_rule_scores_for_feature_set(
    extracted_features: dict | pd.Series | pd.DataFrame,
    feature_set_name: str = DEFAULT_FEATURE_SET_NAME,
) -> pd.DataFrame:
    scoring_frame = prepare_extracted_features_for_feature_set(
        extracted_features,
        feature_set_name=feature_set_name,
    )
    return score_batch(scoring_frame)


def build_primary_model_frame(
    raw_input: dict | pd.Series | pd.DataFrame,
    extracted_features: dict | pd.Series | pd.DataFrame | None = None,
    rule_scores: dict | pd.Series | pd.DataFrame | None = None,
    feature_set_name: str = DEFAULT_FEATURE_SET_NAME,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    raw_frame = _coerce_dataframe(raw_input)
    raw_features_frame = _coerce_feature_dataframe(extracted_features, raw_frame.index)
    features_frame = prepare_extracted_features_for_feature_set(
        raw_features_frame,
        feature_set_name=feature_set_name,
    ).reindex(raw_frame.index)
    rule_frame = _coerce_feature_dataframe(rule_scores, raw_frame.index)
    resolved_feature_columns = feature_columns or FEATURE_COLUMNS

    amount_series = (
        raw_frame["amount"]
        if "amount" in raw_frame.columns
        else pd.Series(0.0, index=raw_frame.index)
    )
    normative_series = (
        raw_frame["normative"]
        if "normative" in raw_frame.columns
        else pd.Series(0.0, index=raw_frame.index)
    )
    amount = pd.to_numeric(amount_series, errors="coerce").fillna(0.0).clip(lower=0.0)
    normative = pd.to_numeric(normative_series, errors="coerce").fillna(0.0)
    normative_safe = normative.replace(0, np.nan)
    unit_count_raw = (amount / normative_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    primary_frame = pd.DataFrame(index=raw_frame.index)
    for column in PRIMARY_MODEL_CATEGORICAL_COLUMNS:
        column_series = (
            raw_frame[column]
            if column in raw_frame.columns
            else pd.Series("", index=raw_frame.index)
        )
        primary_frame[column] = (
            column_series
            .fillna("")
            .astype(str)
            .str.strip()
        )

    primary_frame["normative_log"] = np.log1p(normative.clip(lower=0.0)).astype(float)
    primary_frame["amount_log"] = np.log1p(amount.clip(lower=0.0)).astype(float)
    primary_frame["amount_adequacy"] = (
        pd.to_numeric(
            features_frame["amount_adequacy"]
            if "amount_adequacy" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .astype(float)
    )
    primary_frame["normative_match"] = (
        pd.to_numeric(
            features_frame["normative_match"]
            if "normative_match" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["normative_original_match"] = (
        pd.to_numeric(
            features_frame["normative_original_match"]
            if "normative_original_match" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["normative_reference_gap"] = (
        pd.to_numeric(
            features_frame["normative_reference_gap"]
            if "normative_reference_gap" in features_frame.columns
            else pd.Series(1.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(1.0)
        .clip(lower=0.0, upper=3.0)
        .astype(float)
    )
    primary_frame["normative_reference_typicality"] = (
        pd.to_numeric(
            features_frame["normative_reference_typicality"]
            if "normative_reference_typicality" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["amount_normative_integrity"] = (
        pd.to_numeric(
            features_frame["amount_normative_integrity"]
            if "amount_normative_integrity" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["unit_count_log"] = np.log1p(unit_count_raw.clip(lower=0.0, upper=500.0)).astype(float)
    primary_frame["unit_count_original_log"] = (
        pd.to_numeric(
            features_frame["unit_count_original_log"]
            if "unit_count_original_log" in features_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=8.0)
        .astype(float)
    )
    primary_frame["submit_month_sin"] = (
        pd.to_numeric(
            features_frame["submit_month_sin"]
            if "submit_month_sin" in features_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )
    primary_frame["submit_month_cos"] = (
        pd.to_numeric(
            features_frame["submit_month_cos"]
            if "submit_month_cos" in features_frame.columns
            else pd.Series(1.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(1.0)
        .astype(float)
    )
    primary_frame["region_specialization"] = (
        pd.to_numeric(
            features_frame["region_specialization"]
            if "region_specialization" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    region_direction_approval_rate = (
        pd.to_numeric(
            features_frame["region_direction_approval_rate"]
            if "region_direction_approval_rate" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .astype(float)
    )
    primary_frame["region_direction_approval_rate"] = (
        region_direction_approval_rate
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["direction_approval_rate"] = (
        pd.to_numeric(
            features_frame["direction_approval_rate"]
            if "direction_approval_rate" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .astype(float)
    )
    primary_frame["subsidy_type_approval_rate"] = (
        pd.to_numeric(
            features_frame["subsidy_type_approval_rate"]
            if "subsidy_type_approval_rate" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .astype(float)
    )
    akimat_approval_rate = (
        pd.to_numeric(
            features_frame["akimat_approval_rate"]
            if "akimat_approval_rate" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .astype(float)
    )
    primary_frame["akimat_approval_rate"] = (
        akimat_approval_rate
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["region_approval_rate"] = (
        pd.to_numeric(
            features_frame["region_approval_rate"]
            if "region_approval_rate" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .sub(0.5)
        .mul(REGIONAL_SIGNAL_SHRINK)
        .add(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["region_direction_lift"] = (
        region_direction_approval_rate - primary_frame["direction_approval_rate"]
    ).mul(REGIONAL_SIGNAL_SHRINK).clip(-0.5, 0.5).astype(float)
    primary_frame["akimat_lift"] = (
        akimat_approval_rate - primary_frame["region_approval_rate"]
    ).mul(REGIONAL_SIGNAL_SHRINK).clip(-0.5, 0.5).astype(float)
    primary_frame["direction_history_count_log"] = (
        pd.to_numeric(
            features_frame["direction_history_count_log"]
            if "direction_history_count_log" in features_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=12.0)
        .astype(float)
    )
    primary_frame["subsidy_type_history_count_log"] = (
        pd.to_numeric(
            features_frame["subsidy_type_history_count_log"]
            if "subsidy_type_history_count_log" in features_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=12.0)
        .astype(float)
    )
    primary_frame["region_direction_history_count_log"] = (
        pd.to_numeric(
            features_frame["region_direction_history_count_log"]
            if "region_direction_history_count_log" in features_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=12.0)
        .astype(float)
    )
    primary_frame["akimat_history_count_log"] = (
        pd.to_numeric(
            features_frame["akimat_history_count_log"]
            if "akimat_history_count_log" in features_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=12.0)
        .astype(float)
    )
    primary_frame["budget_pressure"] = (
        pd.to_numeric(
            features_frame["budget_pressure"]
            if "budget_pressure" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["queue_position"] = (
        pd.to_numeric(
            features_frame["queue_position"]
            if "queue_position" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["amount_to_normative_ratio"] = (
        (amount / normative_safe)
        .replace([np.inf, -np.inf], np.nan)
        .clip(0.0, 50.0)
        .fillna(1.0)
        .astype(float)
    )
    primary_frame["amount_to_type_median_ratio"] = (
        pd.to_numeric(
            features_frame["amount_to_type_median_ratio"]
            if "amount_to_type_median_ratio" in features_frame.columns
            else pd.Series(1.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(1.0)
        .clip(lower=0.0, upper=5.0)
        .astype(float)
    )
    primary_frame["adequacy_x_direction_rate"] = (
        primary_frame["amount_adequacy"]
        * primary_frame["direction_approval_rate"]
    ).astype(float)
    primary_frame["adequacy_x_budget_pressure"] = (
        primary_frame["amount_adequacy"]
        * primary_frame["budget_pressure"]
    ).astype(float)
    primary_frame["rule_score_feature"] = (
        pd.to_numeric(
            rule_frame["score"]
            if "score" in rule_frame.columns
            else pd.Series(0.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .div(100.0)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    primary_frame["rule_score"] = (
        pd.to_numeric(
            rule_frame["score"]
            if "score" in rule_frame.columns
            else pd.Series(50.0, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(50.0)
        .clip(lower=0.0, upper=100.0)
        .astype(float)
    )
    for column_name in RULE_FEATURE_COLUMNS:
        primary_frame[column_name] = (
            pd.to_numeric(
                features_frame[column_name]
                if column_name in features_frame.columns
                else pd.Series(0.5, index=raw_frame.index),
                errors="coerce",
            )
            .fillna(0.5)
            .clip(lower=0.0, upper=1.0)
            .astype(float)
        )
    for column_name in RULE_AUGMENTATION_COLUMNS:
        contribution_column = column_name
        source_column = column_name
        if source_column not in rule_frame.columns and source_column.replace("contrib_", "") in rule_frame.columns:
            source_column = source_column.replace("contrib_", "")
        primary_frame[column_name] = (
            pd.to_numeric(
                rule_frame[source_column]
                if source_column in rule_frame.columns
                else pd.Series(0.0, index=raw_frame.index),
                errors="coerce",
            )
            .fillna(0.0)
            .div(100.0)
            .clip(lower=0.0, upper=1.0)
            .astype(float)
        )

    primary_frame["criteria_complexity_x_subsidy_type_rate"] = (
        primary_frame["criteria_complexity"]
        * primary_frame["subsidy_type_approval_rate"]
    ).astype(float)
    primary_frame["direction_risk_x_mortality_compliance"] = (
        primary_frame["direction_risk"]
        * primary_frame["mortality_compliance"]
    ).astype(float)
    primary_frame["rule_score_x_budget_pressure"] = (
        primary_frame["rule_score_feature"]
        * primary_frame["budget_pressure"]
    ).astype(float)
    primary_frame["amount_log_x_rule_score"] = (
        primary_frame["amount_log"]
        * primary_frame["rule_score_feature"]
    ).astype(float)

    return primary_frame.reindex(columns=resolved_feature_columns)


def build_merit_sample_weight(
    y: pd.Series | np.ndarray,
    merit_signal: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    return pd.Series(
        compute_sample_weight(class_weight="balanced", y=np.asarray(y).astype(int))
    )


def build_merit_target(primary_frame: pd.DataFrame) -> pd.DataFrame:
    amount_ratio_typicality = _ratio_to_typicality(
        primary_frame["amount_to_normative_ratio"]
        if "amount_to_normative_ratio" in primary_frame.columns
        else pd.Series(1.0, index=primary_frame.index),
        neutral_value=0.5,
    ).reindex(primary_frame.index, fill_value=0.5)
    unit_count_score = (
        pd.to_numeric(
            primary_frame["unit_count_log"]
            if "unit_count_log" in primary_frame.columns
            else pd.Series(0.0, index=primary_frame.index),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(0.0, 6.0)
        / 6.0
    )
    merit_score = (
        pd.to_numeric(
            primary_frame["amount_adequacy"]
            if "amount_adequacy" in primary_frame.columns
            else pd.Series(0.5, index=primary_frame.index),
            errors="coerce",
        ).fillna(0.5).astype(float) * 0.50
        + amount_ratio_typicality.astype(float) * 0.30
        + unit_count_score.astype(float) * 0.20
    ).clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "merit_proxy_score": merit_score.round(6),
            "merit_proxy_positive": (merit_score >= MERIT_PROXY_POSITIVE_THRESHOLD).astype(int),
        },
        index=primary_frame.index,
    )


def _candidate_models(
    random_state: int,
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    categorical_columns = categorical_columns or PRIMARY_MODEL_CATEGORICAL_COLUMNS
    numeric_columns = numeric_columns or PRIMARY_MODEL_NUMERIC_COLUMNS
    logistic_preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                categorical_columns,
            ),
            ("num", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )
    neural_preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                categorical_columns,
            ),
            ("num", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )
    forest_preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                categorical_columns,
            ),
            ("num", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", logistic_preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=0.7,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "neural_network": Pipeline(
            steps=[
                ("preprocess", neural_preprocess),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=512,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.1,
                        max_iter=300,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", forest_preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=10,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight=None,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", forest_preprocess),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        max_depth=12,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight=None,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    skipped_models: dict[str, str] = {}
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            iterations=900,
            depth=4,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights=None,
            l2_leaf_reg=12.0,
            min_data_in_leaf=80,
            random_seed=random_state,
            verbose=0,
            allow_writing_files=False,
            cat_features=categorical_columns,
        )
    else:
        skipped_models["catboost"] = "catboost dependency is not installed"
    return models, skipped_models


def get_available_model_candidates(
    random_state: int = 42,
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> dict[str, object]:
    models, skipped_models = _candidate_models(
        random_state,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    return {
        "available_model_names": list(models.keys()),
        "skipped_models": skipped_models,
    }


def model_supports_sample_weight(
    model_name: str,
    random_state: int = 42,
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> bool:
    models, _ = _candidate_models(
        random_state,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    if model_name not in models:
        return False
    return _supports_sample_weight(models[model_name])


def rolling_time_cv_leaderboard(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    random_state: int = 42,
    fold_bounds: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    if fold_bounds is None:
        fold_bounds = [(0.5, 0.65), (0.6, 0.75), (0.7, 0.85)]

    candidate_models, skipped_models = _candidate_models(random_state)
    submit_dates = pd.to_datetime(metadata["submit_date"], errors="coerce").fillna(
        pd.Timestamp.max
    )
    ordered_index = (
        pd.DataFrame(
            {
                "idx": metadata.index,
                "submit_date": submit_dates,
                "app_number": metadata["app_number"].astype(str),
            }
        )
        .sort_values(["submit_date", "app_number", "idx"])
        ["idx"]
        .tolist()
    )

    n_rows = len(ordered_index)
    leaderboard: list[dict[str, object]] = []
    for model_name, model in candidate_models.items():
        fold_scores: list[float] = []
        for train_frac, valid_frac in fold_bounds:
            train_end = int(n_rows * train_frac)
            valid_end = int(n_rows * valid_frac)
            train_idx = ordered_index[:train_end]
            valid_idx = ordered_index[train_end:valid_end]
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue

            train_sample_weight = None
            if sample_weight is not None:
                train_sample_weight = sample_weight.loc[train_idx]
            fitted_model = _fit_model(
                model,
                X.loc[train_idx],
                y.loc[train_idx],
                sample_weight=train_sample_weight,
            )
            y_valid_prob = fitted_model.predict_proba(X.loc[valid_idx])[:, 1]
            fold_scores.append(float(roc_auc_score(y.loc[valid_idx], y_valid_prob)))

        if not fold_scores:
            continue

        leaderboard.append(
            {
                "model_name": model_name,
                "cv_roc_auc_mean": round(float(np.mean(fold_scores)), 4),
                "cv_roc_auc_std": round(float(np.std(fold_scores)), 4),
                "fold_scores": [round(float(score), 4) for score in fold_scores],
            }
        )

    leaderboard.sort(
        key=lambda item: (
            item["cv_roc_auc_mean"],
            -item["cv_roc_auc_std"],
        ),
        reverse=True,
    )
    return {
        "best_model_name": leaderboard[0]["model_name"],
        "leaderboard": leaderboard,
        "skipped_models": skipped_models,
    }


def get_blend_rule_score_column(feature_set_name: str) -> str:
    resolved = resolve_feature_set_name(feature_set_name)
    return f"blend_rule_score__{resolved}"


def build_training_dataset(data_path: str) -> dict[str, object]:
    df = run_pipeline(data_path)
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    synthetic_table, synthetic_feature_info = load_synthetic_feature_table()
    merged_features, synthetic_feature_info = merge_synthetic_features(
        raw_input=df,
        extracted_features=features,
        synthetic_table=synthetic_table,
        synthetic_info=synthetic_feature_info,
    )
    rule_scores = score_batch(merged_features)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])
    rule_scores[eligibility.columns] = eligibility
    deadline_disqualified = eligibility["disqualified"].fillna(False).astype(bool)
    eligible_mask = ~deadline_disqualified

    X_by_feature_set: dict[str, pd.DataFrame] = {}
    feature_set_rule_scores: dict[str, pd.DataFrame] = {}
    feature_set_columns: dict[str, list[str]] = {}
    feature_set_numeric_columns: dict[str, list[str]] = {}
    feature_set_categorical_columns: dict[str, list[str]] = {}

    eligible_df = df.loc[eligible_mask]
    eligible_features = merged_features.loc[eligible_mask]
    for feature_set_name in FEATURE_SET_ORDER:
        honest_rule_scores = build_rule_scores_for_feature_set(
            eligible_features,
            feature_set_name=feature_set_name,
        )
        honest_rule_scores[eligibility.columns] = eligibility.loc[eligible_mask]
        feature_set_rule_scores[feature_set_name] = honest_rule_scores
        feature_set_columns[feature_set_name] = get_feature_columns(feature_set_name)
        feature_set_numeric_columns[feature_set_name] = get_numeric_feature_columns(feature_set_name)
        feature_set_categorical_columns[feature_set_name] = get_categorical_feature_columns(feature_set_name)
        X_by_feature_set[feature_set_name] = build_primary_model_frame(
            raw_input=eligible_df,
            extracted_features=eligible_features,
            rule_scores=honest_rule_scores,
            feature_set_name=feature_set_name,
            feature_columns=feature_set_columns[feature_set_name],
        )

    X = X_by_feature_set[DEFAULT_FEATURE_SET_NAME]
    merit_target_frame = build_merit_target(X)
    y = df.loc[eligible_mask, "is_approved"].astype(int).copy()

    metadata_columns = [
        "app_number",
        "submit_date",
        "region",
        "district",
        "direction",
        "subsidy_type",
        "amount",
        "status",
        "is_approved",
    ]
    metadata = df.loc[eligible_mask].reindex(columns=metadata_columns).copy()
    metadata["rule_score"] = rule_scores.loc[eligible_mask, "score"].astype(float)
    metadata["rule_risk_level"] = rule_scores.loc[eligible_mask, "risk_level"].astype(str)
    metadata["disqualified"] = rule_scores.loc[eligible_mask, "disqualified"].fillna(False).astype(bool)
    metadata["disqualification_reason"] = rule_scores.loc[
        eligible_mask, "disqualification_reason"
    ]
    metadata["eligibility_status"] = rule_scores.loc[
        eligible_mask, "eligibility_status"
    ].astype(str)
    metadata["manual_review_required"] = rule_scores.loc[
        eligible_mask, "manual_review_required"
    ].fillna(False).astype(bool)
    metadata["eligibility_note"] = rule_scores.loc[eligible_mask, "eligibility_note"]
    metadata["normative_reference_found"] = rule_scores.loc[
        eligible_mask, "normative_reference_found"
    ].fillna(False).astype(bool)
    metadata["historical_is_approved"] = y.astype(int)
    metadata["merit_proxy_score"] = merit_target_frame["merit_proxy_score"].astype(float)
    metadata["merit_proxy_positive"] = merit_target_frame["merit_proxy_positive"].astype(int)
    metadata["training_target"] = y.astype(int)
    for feature_set_name, honest_rule_scores in feature_set_rule_scores.items():
        metadata[get_blend_rule_score_column(feature_set_name)] = honest_rule_scores["score"].astype(float)

    metadata["sample_weight"] = build_merit_sample_weight(
        y=y,
        merit_signal=metadata["merit_proxy_score"],
    ).to_numpy()

    return {
        "df": df,
        "tables": tables,
        "features": merged_features,
        "base_features": features,
        "rule_scores": rule_scores,
        "X": X,
        "X_by_feature_set": X_by_feature_set,
        "feature_set_columns": feature_set_columns,
        "feature_set_numeric_columns": feature_set_numeric_columns,
        "feature_set_categorical_columns": feature_set_categorical_columns,
        "feature_set_deployability": {
            feature_set_name: get_feature_set_deployability(feature_set_name)
            for feature_set_name in FEATURE_SET_ORDER
        },
        "feature_set_rule_scores": feature_set_rule_scores,
        "synthetic_feature_info": synthetic_feature_info,
        "y": y,
        "metadata": metadata,
        "deadline_disqualified_count": int(deadline_disqualified.sum()),
        "eligible_rows": int(eligible_mask.sum()),
    }


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    mode: str = "time",
    random_state: int = 42,
) -> dict[str, dict[str, object]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")
    if len(X) < 10:
        raise ValueError("Need at least 10 records to build train/valid/test splits")

    if mode == "time":
        submit_dates = pd.to_datetime(metadata["submit_date"], errors="coerce")
        order_frame = pd.DataFrame(
            {
                "submit_date": submit_dates.fillna(pd.Timestamp.max),
                "app_number": metadata["app_number"].astype(str),
                "row_id": metadata.index,
            }
        )
        ordered_index = order_frame.sort_values(
            ["submit_date", "app_number", "row_id"]
        )["row_id"].tolist()
    elif mode == "random":
        ordered_index = metadata.index.to_list()
        rng = np.random.default_rng(random_state)
        rng.shuffle(ordered_index)
    else:
        raise ValueError("mode must be 'time' or 'random'")

    n_rows = len(ordered_index)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))

    train_end = max(train_end, 1)
    valid_end = max(valid_end, train_end + 1)
    valid_end = min(valid_end, n_rows - 1)

    split_index = {
        "train": ordered_index[:train_end],
        "valid": ordered_index[train_end:valid_end],
        "test": ordered_index[valid_end:],
    }

    result: dict[str, dict[str, object]] = {}
    for split_name, split_rows in split_index.items():
        result[split_name] = {
            "X": X.loc[split_rows].copy(),
            "y": y.loc[split_rows].copy(),
            "metadata": metadata.loc[split_rows].copy(),
        }
    return result


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    score_scale_max: float | None = None,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    resolved_scale_max = resolve_score_scale_max(y_prob_arr, score_scale_max)
    normalized_scores = (
        np.clip(y_prob_arr / resolved_scale_max, 0.0, 1.0)
        if resolved_scale_max > 1.0 + 1e-8
        else np.clip(y_prob_arr, 0.0, 1.0)
    )

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_accuracy = (specificity + sensitivity) / 2

    metrics: dict[str, object] = {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "positive_rate_pred": round(float(y_pred.mean()), 4),
        "positive_rate_true": round(float(y_true_arr.mean()), 4),
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if len(np.unique(y_true_arr)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
        metrics["average_precision"] = round(
            float(average_precision_score(y_true_arr, y_prob_arr)), 4
        )
        metrics["brier_score"] = round(float(brier_score_loss(y_true_arr, normalized_scores)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["brier_score"] = None

    return metrics


class ProbabilityCalibrator:
    def __init__(
        self,
        method: str = "identity",
        estimator: object | None = None,
    ) -> None:
        self.method = method
        self.estimator = estimator

    def transform(
        self,
        y_prob: pd.Series | np.ndarray,
    ) -> np.ndarray:
        prob = np.asarray(y_prob).astype(float)
        clipped = np.clip(prob, 1e-6, 1 - 1e-6)

        if self.method == "identity" or self.estimator is None:
            return np.clip(prob, 0.0, 1.0)
        if self.method == "sigmoid":
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            calibrated = self.estimator.predict_proba(logits)[:, 1]
            return np.clip(calibrated, 0.0, 1.0)
        if self.method == "isotonic":
            calibrated = self.estimator.predict(clipped)
            return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
        raise ValueError(f"Unknown calibrator method: {self.method}")


def apply_probability_calibrator(
    y_prob: pd.Series | np.ndarray,
    calibrator: ProbabilityCalibrator | None = None,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    return calibrator.transform(y_prob)


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    if len(y_true_arr) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        prob_mean = float(np.mean(y_prob_arr[mask]))
        true_mean = float(np.mean(y_true_arr[mask]))
        ece += abs(prob_mean - true_mean) * (float(np.sum(mask)) / float(len(y_true_arr)))
    return round(float(ece), 6)


def _fit_probability_calibrator(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    method: str,
) -> ProbabilityCalibrator:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    clipped = np.clip(y_prob_arr, 1e-6, 1 - 1e-6)

    if method == "identity":
        return ProbabilityCalibrator(method="identity")
    if method == "sigmoid":
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        estimator.fit(logits, y_true_arr)
        return ProbabilityCalibrator(method="sigmoid", estimator=estimator)
    if method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(clipped, y_true_arr)
        return ProbabilityCalibrator(method="isotonic", estimator=estimator)
    raise ValueError(f"Unknown calibrator method: {method}")


def choose_probability_calibrator(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
) -> dict[str, object]:
    candidate_methods = ["identity", "sigmoid", "isotonic"]
    candidates: list[dict[str, object]] = []
    y_true_arr = np.asarray(y_true).astype(int)
    for method in candidate_methods:
        calibrator = _fit_probability_calibrator(y_true_arr, y_prob, method)
        calibrated_prob = apply_probability_calibrator(y_prob, calibrator)
        metrics = evaluate_predictions(y_true_arr, calibrated_prob)
        class_means = (
            pd.DataFrame({"target": y_true_arr, "probability": calibrated_prob})
            .groupby("target")["probability"]
            .mean()
            .to_dict()
        )
        class_gap = float(class_means.get(1, 0.0) - class_means.get(0, 0.0))
        candidates.append(
            {
                "method": method,
                "calibrator": calibrator,
                "validation_metrics": metrics,
                "expected_calibration_error": expected_calibration_error(
                    y_true_arr,
                    calibrated_prob,
                ),
                "probability_std": round(float(np.std(calibrated_prob)), 6),
                "class_gap": round(class_gap, 6),
            }
        )

    best_balanced_accuracy = max(
        float(item["validation_metrics"]["balanced_accuracy"])
        for item in candidates
    )
    discriminative_candidates = [
        item
        for item in candidates
        if float(item["validation_metrics"]["balanced_accuracy"])
        >= best_balanced_accuracy - 0.03
    ]
    if not discriminative_candidates:
        discriminative_candidates = candidates.copy()
    best_auc = max(
        item["validation_metrics"]["roc_auc"]
        if item["validation_metrics"]["roc_auc"] is not None
        else -1
        for item in discriminative_candidates
    )
    shortlisted = [
        item
        for item in discriminative_candidates
        if (item["validation_metrics"]["roc_auc"] or -1) >= best_auc - 0.01
    ]
    if not shortlisted:
        shortlisted = discriminative_candidates.copy()
    shortlisted.sort(
        key=lambda item: (
            item["validation_metrics"]["brier_score"] or 1e9,
            -item["class_gap"],
            -item["probability_std"],
            item["expected_calibration_error"],
        ),
    )
    best = shortlisted[0]
    return {
        "best_method": best["method"],
        "best_calibrator": best["calibrator"],
        "best_validation_metrics": best["validation_metrics"],
        "candidates": [
            {
                key: value
                for key, value in item.items()
                if key != "calibrator"
            }
            for item in candidates
        ],
    }


def apply_probability_temperature(
    y_prob: pd.Series | np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    prob = np.asarray(y_prob).astype(float)
    if temperature <= 1.0:
        return np.clip(prob, 0.0, 1.0)

    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    scaled = 1 / (1 + np.exp(-(logits / float(temperature))))
    return np.clip(scaled, 0.0, 1.0)


def tune_decision_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 3.0,
    fp_cost: float = 1.0,
    score_scale_max: float | None = None,
) -> dict[str, object]:
    resolved_scale_max = resolve_score_scale_max(y_prob, score_scale_max)
    decision_anchor = 0.5 if resolved_scale_max <= 1.0 + 1e-8 else resolved_scale_max / 2.0
    candidate_thresholds = build_threshold_candidates(
        y_prob,
        score_scale_max=resolved_scale_max,
    )

    candidates: list[dict[str, object]] = []
    for threshold in candidate_thresholds:
        metrics = evaluate_predictions(
            y_true,
            y_prob,
            threshold=float(threshold),
            score_scale_max=resolved_scale_max,
        )
        confusion = metrics["confusion_matrix"]
        weighted_cost = (
            float(confusion["fn"]) * float(fn_cost)
            + float(confusion["fp"]) * float(fp_cost)
        )
        metrics["selection_cost"] = round(weighted_cost, 4)
        candidates.append(metrics)

    best_cost = min(float(item["selection_cost"]) for item in candidates)
    cost_tolerance = max(best_cost * 6.0, best_cost + 50.0)
    relaxed_candidates = [
        item
        for item in candidates
        if float(item["selection_cost"]) <= cost_tolerance
    ]
    relaxed_candidates.sort(
        key=lambda item: (
            item["balanced_accuracy"],
            item["recall"],
            item["f1"],
            -float(item["selection_cost"]),
            -abs(float(item["threshold"]) - decision_anchor),
        ),
        reverse=True,
    )
    best = relaxed_candidates[0]
    return {
        "best_threshold": float(best["threshold"]),
        "best_metrics": best,
        "candidates": relaxed_candidates[:15],
        "best_selection_cost": round(float(best_cost), 4),
    }


def tune_blend_weights(
    y_true: pd.Series | np.ndarray,
    rule_scores: pd.Series | np.ndarray,
    ml_probabilities: pd.Series | np.ndarray,
    disqualified_mask: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true).astype(int)
    if disqualified_mask is None:
        disqualified_arr = np.zeros_like(y_true_arr, dtype=bool)
    else:
        disqualified_arr = np.asarray(disqualified_mask).astype(bool)

    candidates: list[dict[str, object]] = []
    for rule_weight in BLEND_WEIGHT_SEARCH_GRID:
        ml_weight = 1.0 - rule_weight
        blended = compute_blended_scores(
            rule_scores=rule_scores,
            ml_probabilities=ml_probabilities,
            blend_weights={
                "rule_score": rule_weight,
                "ml_score": ml_weight,
            },
            disqualified_mask=disqualified_arr,
        )

        metrics = {
            "roc_auc": round(float(roc_auc_score(y_true_arr, blended)), 4),
            "average_precision": round(float(average_precision_score(y_true_arr, blended)), 4),
            "mean_score": round(float(np.mean(blended)), 4),
        }
        candidates.append(
            {
                "rule_score": round(float(rule_weight), 2),
                "ml_score": round(float(ml_weight), 2),
                "metrics": metrics,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["metrics"]["roc_auc"],
            item["metrics"]["average_precision"],
            item["rule_score"],
        ),
        reverse=True,
    )
    best = candidates[0]
    return {
        "best_weights": {
            "rule_score": best["rule_score"],
            "ml_score": best["ml_score"],
        },
        "best_metrics": best["metrics"],
        "candidates": candidates[:10],
    }


def compute_permutation_feature_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    max_rows: int = 3000,
) -> list[dict[str, float]]:
    if len(X) > max_rows:
        sample_idx = X.sample(n=max_rows, random_state=random_state).index
        X_eval = X.loc[sample_idx]
        y_eval = y.loc[sample_idx]
    else:
        X_eval = X
        y_eval = y

    result = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring="roc_auc",
        n_repeats=8,
        random_state=random_state,
        n_jobs=-1,
    )
    ranking = [
        {
            "feature": feature_name,
            "importance_mean": round(float(mean), 6),
            "importance_std": round(float(std), 6),
        }
        for feature_name, mean, std in zip(
            X_eval.columns,
            result.importances_mean,
            result.importances_std,
        )
    ]
    ranking.sort(key=lambda item: item["importance_mean"], reverse=True)
    return ranking


def _extract_feature_importance(model: object, feature_columns: list[str]) -> list[dict[str, float]]:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    resolved_feature_columns = feature_columns

    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        preprocessor = model.named_steps["preprocess"]
        if hasattr(preprocessor, "get_feature_names_out"):
            resolved_feature_columns = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return []

    ranking = [
        {"feature": feature_name, "importance": round(float(importance), 6)}
        for feature_name, importance in zip(resolved_feature_columns, values)
    ]
    ranking.sort(key=lambda item: item["importance"], reverse=True)
    return ranking


def train_final_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    random_state: int = 42,
) -> object:
    candidates, _ = _candidate_models(
        random_state,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    if model_name not in candidates:
        raise ValueError(f"Unknown model_name: {model_name}")

    return _fit_model(candidates[model_name], X, y, sample_weight=sample_weight)


def train_calibrated_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    calibration_method: str = "sigmoid",
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    random_state: int = 42,
) -> object:
    candidates, _ = _candidate_models(
        random_state,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    if model_name not in candidates:
        raise ValueError(f"Unknown model_name: {model_name}")
    if calibration_method == "raw":
        return train_final_model(
            model_name=model_name,
            X=X,
            y=y,
            sample_weight=sample_weight,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            random_state=random_state,
        )

    calibrated_model = CalibratedClassifierCV(
        estimator=deepcopy(candidates[model_name]),
        method=calibration_method,
        cv=5,
        ensemble=False,
    )
    fit_kwargs: dict[str, object] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight)
    calibrated_model.fit(X, y, **fit_kwargs)
    return calibrated_model


def _score_to_risk_label(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-1, 45, 70, 101],
        labels=["Высокий", "Средний", "Низкий"],
    ).astype(str)


def score_to_risk_label(score: float) -> str:
    if score >= 70:
        return "Низкий"
    if score >= 45:
        return "Средний"
    return "Высокий"


def build_explanation_neutral_values(
    feature_frame: pd.DataFrame,
) -> dict[str, object]:
    neutral_values: dict[str, object] = {}
    for column_name in feature_frame.columns:
        column = feature_frame[column_name]
        if pd.api.types.is_numeric_dtype(column):
            median_value = pd.to_numeric(column, errors="coerce").median()
            neutral_values[column_name] = (
                float(median_value)
                if pd.notna(median_value)
                else 0.0
            )
        else:
            neutral_values[column_name] = ""
    return neutral_values


def prepare_feature_frame(
    features_input: dict | pd.Series | pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_frame = _coerce_dataframe(features_input)
    resolved_feature_columns = feature_columns or FEATURE_COLUMNS
    return feature_frame.reindex(columns=resolved_feature_columns).copy()


def explain_prediction_with_model(
    features_input: dict | pd.Series | pd.DataFrame,
    model: object,
    neutral_values: dict[str, object] | None = None,
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    feature_frame = prepare_feature_frame(
        features_input,
        feature_columns=feature_columns,
    )
    if len(feature_frame) != 1:
        raise ValueError("explain_prediction_with_model supports exactly one row")

    if neutral_values is None:
        neutral_values = build_explanation_neutral_values(feature_frame)

    base_prob_raw = float(model.predict_proba(feature_frame)[:, 1][0])
    if probability_calibrator is not None:
        base_prob = float(apply_probability_calibrator([base_prob_raw], probability_calibrator)[0])
    elif probability_temperature is not None:
        base_prob = float(apply_probability_temperature([base_prob_raw], probability_temperature)[0])
    else:
        base_prob = base_prob_raw

    perturbed_frames: list[pd.DataFrame] = []
    column_names = feature_frame.columns.tolist()
    for column_name in column_names:
        perturbed = feature_frame.copy()
        perturbed.loc[perturbed.index[0], column_name] = neutral_values.get(column_name, "")
        perturbed_frames.append(perturbed)

    stacked = pd.concat(perturbed_frames, axis=0, ignore_index=True)
    perturbed_prob_raw = model.predict_proba(stacked)[:, 1]
    if probability_calibrator is not None:
        perturbed_prob = apply_probability_calibrator(perturbed_prob_raw, probability_calibrator)
    elif probability_temperature is not None:
        perturbed_prob = apply_probability_temperature(
            perturbed_prob_raw,
            temperature=probability_temperature,
        )
    else:
        perturbed_prob = np.asarray(perturbed_prob_raw, dtype=float)

    feature_effects: list[dict[str, object]] = []
    for idx, column_name in enumerate(column_names):
        raw_value = feature_frame.iloc[0][column_name]
        contribution = round(float((base_prob - perturbed_prob[idx]) * 100), 4)
        feature_effects.append(
            {
                "name": column_name,
                "value": raw_value,
                "neutral_value": neutral_values.get(column_name),
                "score_impact": contribution,
            }
        )

    feature_effects.sort(
        key=lambda item: abs(float(item["score_impact"])),
        reverse=True,
    )
    return {
        "ml_probability": round(base_prob, 6),
        "feature_effects": feature_effects,
    }


def resolve_disqualification_mask(
    features_input: dict | pd.Series | pd.DataFrame,
    rule_scores: pd.DataFrame | None = None,
    disqualified_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.Series:
    if disqualified_mask is not None:
        mask = pd.Series(disqualified_mask)
        return mask.fillna(False).astype(bool)

    if rule_scores is not None and "disqualified" in rule_scores.columns:
        return rule_scores["disqualified"].fillna(False).astype(bool)

    if isinstance(features_input, pd.DataFrame):
        feature_frame = features_input.copy()
    elif isinstance(features_input, pd.Series):
        feature_frame = pd.DataFrame([features_input.to_dict()])
    else:
        feature_frame = pd.DataFrame([features_input])

    if "deadline_compliance" not in feature_frame.columns:
        return pd.Series(False, index=feature_frame.index)

    deadline_values = pd.to_numeric(
        feature_frame["deadline_compliance"], errors="coerce"
    ).fillna(0.5)
    return (deadline_values <= 0).astype(bool)


def score_features_with_model(
    features_input: dict | pd.Series | pd.DataFrame,
    model: object,
    rule_scores: pd.DataFrame | None = None,
    blend_rule_scores: pd.DataFrame | None = None,
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float = 1.0,
    disqualified_mask: pd.Series | np.ndarray | list[bool] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    blend_weights = resolve_blend_weights(blend_weights)

    feature_frame = prepare_feature_frame(
        features_input,
        feature_columns=feature_columns,
    )
    resolved_disqualified_mask = resolve_disqualification_mask(
        features_input=features_input,
        rule_scores=rule_scores,
        disqualified_mask=disqualified_mask,
    ).reindex(feature_frame.index, fill_value=False)

    if rule_scores is None:
        full_feature_frame = _coerce_dataframe(features_input)
        if set(RULE_FEATURE_COLUMNS).issubset(full_feature_frame.columns):
            rule_scores = score_batch(full_feature_frame)
        else:
            rule_scores = pd.DataFrame(index=feature_frame.index)
            rule_scores["score"] = 0.0
            rule_scores["risk_level"] = "Высокий"
    else:
        rule_scores = rule_scores.copy()

    if blend_rule_scores is None:
        blend_rule_scores = rule_scores.copy()
    else:
        blend_rule_scores = blend_rule_scores.copy()

    result = pd.DataFrame(index=feature_frame.index)
    result["rule_score"] = rule_scores["score"].astype(float)
    result["rule_risk_level"] = rule_scores["risk_level"].astype(str)
    result["blend_rule_score"] = blend_rule_scores["score"].astype(float)
    result["ml_probability"] = np.nan
    result["ml_score"] = np.nan
    result["decision_score_name"] = DEFAULT_DECISION_SCORE_NAME
    result["decision_threshold"] = float(decision_threshold)
    result["decision_predicted_positive"] = False
    result["ml_decision_threshold"] = float(decision_threshold)
    result["ml_predicted_positive"] = False
    result["final_score"] = np.round(
        blend_rule_scores["score"].astype(float) * blend_weights["rule_score"],
        1,
    )
    result["final_risk_level"] = result["final_score"].apply(score_to_risk_label)
    result["disqualified"] = resolved_disqualified_mask.astype(bool)
    if "disqualification_reason" in rule_scores.columns:
        result["disqualification_reason"] = rule_scores["disqualification_reason"]
    else:
        result["disqualification_reason"] = np.where(
            resolved_disqualified_mask,
            DEADLINE_DISQUALIFICATION_REASON,
            None,
        )
    for passthrough_column in [
        "eligibility_status",
        "manual_review_required",
        "eligibility_note",
        "normative_reference_found",
    ]:
        if passthrough_column in rule_scores.columns:
            result[passthrough_column] = rule_scores[passthrough_column]

    eligible_mask = ~resolved_disqualified_mask
    if eligible_mask.any():
        ml_prob_raw = model.predict_proba(feature_frame.loc[eligible_mask])[:, 1]
        if probability_calibrator is not None:
            ml_prob = apply_probability_calibrator(
                ml_prob_raw,
                probability_calibrator,
            )
        else:
            ml_prob = apply_probability_temperature(
                ml_prob_raw,
                temperature=probability_temperature,
            )
        ml_score = np.round(ml_prob * 100, 1)
        result.loc[eligible_mask, "ml_probability"] = np.round(ml_prob, 4)
        result.loc[eligible_mask, "ml_score"] = ml_score
        result.loc[eligible_mask, "final_score"] = compute_blended_scores(
            rule_scores=result.loc[eligible_mask, "blend_rule_score"].astype(float),
            ml_probabilities=result.loc[eligible_mask, "ml_probability"].astype(float),
            blend_weights=blend_weights,
        )
        result.loc[eligible_mask, "decision_predicted_positive"] = (
            result.loc[eligible_mask, "final_score"].astype(float) >= float(decision_threshold)
        )
        result.loc[eligible_mask, "ml_predicted_positive"] = (
            result.loc[eligible_mask, "decision_predicted_positive"].astype(bool)
        )
        result.loc[eligible_mask, "final_risk_level"] = result.loc[
            eligible_mask, "final_score"
        ].apply(score_to_risk_label)

    disqualified_mask = result["disqualified"].fillna(False)
    if disqualified_mask.any():
        result.loc[disqualified_mask, "final_score"] = 0.0
        result.loc[disqualified_mask, "final_risk_level"] = "Высокий"
        result.loc[disqualified_mask, "decision_predicted_positive"] = False
        result.loc[disqualified_mask, "ml_predicted_positive"] = False

    result["score"] = result["final_score"]
    result["risk_level"] = result["final_risk_level"]

    return result


def build_prediction_frame(
    df: pd.DataFrame,
    tables: dict,
    model: object,
    feature_set_name: str = DEFAULT_FEATURE_SET_NAME,
    feature_columns: list[str] | None = None,
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float = 1.0,
) -> pd.DataFrame:
    resolved_feature_set = resolve_feature_set_name(feature_set_name)
    resolved_feature_columns = feature_columns or get_feature_columns(resolved_feature_set)
    features = extract_features_batch(df, tables)
    merged_features, _ = merge_synthetic_features(df, features)
    advisory = build_history_advisory_batch(df)
    rule_scores = score_batch(merged_features)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])
    rule_scores[eligibility.columns] = eligibility
    blend_rule_scores = build_rule_scores_for_feature_set(
        merged_features,
        feature_set_name=resolved_feature_set,
    )
    blend_rule_scores[eligibility.columns] = eligibility
    model_input = build_primary_model_frame(
        raw_input=df,
        extracted_features=merged_features,
        rule_scores=blend_rule_scores,
        feature_set_name=resolved_feature_set,
        feature_columns=resolved_feature_columns,
    )
    blended_scores = score_features_with_model(
        model_input,
        model=model,
        rule_scores=rule_scores,
        blend_rule_scores=blend_rule_scores,
        blend_weights=blend_weights,
        decision_threshold=decision_threshold,
        probability_calibrator=probability_calibrator,
        probability_temperature=probability_temperature,
        disqualified_mask=rule_scores["disqualified"],
        feature_columns=resolved_feature_columns,
    )

    result = df[
        [
            "app_number",
            "region",
            "district",
            "direction",
            "subsidy_type",
            "amount",
            "status",
            "is_approved",
        ]
    ].copy()
    result = pd.concat([result, blended_scores, advisory], axis=1)

    return result.sort_values("final_score", ascending=False).reset_index(drop=True)


def save_bundle(
    model: object,
    tables: dict,
    model_name: str,
    output_path: str | Path,
    feature_columns: list[str] | None = None,
    feature_set_name: str = DEFAULT_FEATURE_SET_NAME,
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float = 1.0,
    calibration_method: str | None = None,
    explanation_neutral_values: dict[str, object] | None = None,
    report: dict[str, object] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_set_meta = get_feature_set_deployability(feature_set_name)

    payload = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns or FEATURE_COLUMNS,
        "feature_set_name": resolve_feature_set_name(feature_set_name),
        "feature_set_deployable": bool(feature_set_meta["deployable"]),
        "feature_set_deployability_reason": str(feature_set_meta["reason"]),
        "tables": tables,
        "blend_weights": resolve_blend_weights(blend_weights),
        "ranking_score_name": DEFAULT_DECISION_SCORE_NAME,
        "decision_score_name": DEFAULT_DECISION_SCORE_NAME,
        "decision_threshold_score_name": DEFAULT_DECISION_SCORE_NAME,
        "decision_threshold_scale_max": float(DECISION_SCORE_SCALE_MAX),
        "decision_threshold": float(decision_threshold),
        "probability_calibrator": probability_calibrator,
        "probability_temperature": float(probability_temperature),
        "calibration_method": calibration_method,
        "explanation_neutral_values": explanation_neutral_values or {},
        "report": report or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, output_path)
    return output_path


def load_bundle(model_path: str | Path) -> dict[str, object]:
    return joblib.load(model_path)


def save_json(data: dict[str, object], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
