from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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


RULE_FEATURE_COLUMNS = list(WEIGHTS.keys())
PRIMARY_MODEL_CATEGORICAL_COLUMNS = [
    "region",
    "direction",
    "subsidy_type",
]
PRIMARY_MODEL_NUMERIC_COLUMNS = [
    "amount_log",
    "normative_log",
    "unit_count_log",
    "normative_match",
    "amount_normative_integrity",
    "amount_adequacy",
    "unit_count",
    "amount_typicality",
    "has_normative_reference",
    "region_specialization",
    "submit_month_sin",
    "submit_month_cos",
]
FEATURE_COLUMNS = PRIMARY_MODEL_CATEGORICAL_COLUMNS + PRIMARY_MODEL_NUMERIC_COLUMNS
CORE_NEUTRAL_CATEGORY_VALUES = {
    "region": "",
}
CORE_NEUTRAL_NUMERIC_VALUES = {
    "region_specialization": 0.5,
}
MERIT_PROXY_FEATURE_COLUMNS = [
    "amount_normative_integrity",
    "amount_adequacy",
    "unit_count",
    "amount_typicality",
    "has_normative_reference",
    "normative_match",
]
CONTEXTUAL_HISTORICAL_FEATURE_COLUMNS = [
    "region_direction_approval_rate",
    "akimat_approval_rate",
    "direction_approval_rate",
    "subsidy_type_approval_rate",
    "region_approval_rate",
]
PROCESS_BIASED_FEATURE_COLUMNS = [
    "budget_pressure",
    "queue_position",
]
DEFAULT_BLEND_WEIGHTS = {"rule_score": 0.0, "ml_score": 1.0}
MIN_RULE_BLEND_WEIGHT = 0.0
MERIT_TARGET_THRESHOLD = 0.68
MERIT_PROXY_POSITIVE_THRESHOLD = 0.68
ENSEMBLE_CONTEXT_WEIGHT_DEFAULT = 0.7


def _fit_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> object:
    fit_kwargs: dict[str, object] = {}
    if sample_weight is not None:
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


class DualBranchEnsembleModel(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        context_model: object,
        core_model: object,
        context_weight: float = ENSEMBLE_CONTEXT_WEIGHT_DEFAULT,
        region_column: str = "region",
    ) -> None:
        self.context_model = context_model
        self.core_model = core_model
        self.context_weight = float(context_weight)
        self.region_column = region_column
        self.classes_ = np.array([0, 1])

    def _core_view(self, X: pd.DataFrame) -> pd.DataFrame:
        core_frame = X.copy()
        for column_name, neutral_value in CORE_NEUTRAL_CATEGORY_VALUES.items():
            if column_name in core_frame.columns:
                core_frame[column_name] = neutral_value
        for column_name, neutral_value in CORE_NEUTRAL_NUMERIC_VALUES.items():
            if column_name in core_frame.columns:
                core_frame[column_name] = neutral_value
        return core_frame

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X) -> np.ndarray:
        frame = _coerce_dataframe(X).reindex(columns=FEATURE_COLUMNS).copy()
        context_prob = self.context_model.predict_proba(frame)[:, 1]
        core_prob = self.core_model.predict_proba(self._core_view(frame))[:, 1]
        blended = np.clip(
            self.context_weight * context_prob
            + (1.0 - self.context_weight) * core_prob,
            0.0,
            1.0,
        )
        return np.column_stack([1.0 - blended, blended])


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


def build_primary_model_frame(
    raw_input: dict | pd.Series | pd.DataFrame,
    extracted_features: dict | pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw_frame = _coerce_dataframe(raw_input)
    features_frame = _coerce_feature_dataframe(extracted_features, raw_frame.index)

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

    primary_frame["amount_log"] = np.log1p(amount).astype(float)
    primary_frame["normative_log"] = np.log1p(normative.clip(lower=0.0)).astype(float)
    primary_frame["unit_count_log"] = np.log1p(unit_count_raw.clip(lower=0.0, upper=500.0)).astype(float)
    primary_frame["normative_match"] = (
        pd.to_numeric(
            features_frame["normative_match"]
            if "normative_match" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
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
        .astype(float)
    )
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
    primary_frame["unit_count"] = (
        pd.to_numeric(
            features_frame["unit_count"]
            if "unit_count" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
        .astype(float)
    )
    primary_frame["amount_typicality"] = _ratio_to_typicality(
        features_frame["amount_to_type_median_ratio"]
        if "amount_to_type_median_ratio" in features_frame.columns
        else pd.Series(1.0, index=raw_frame.index),
    ).reindex(raw_frame.index, fill_value=0.5)
    primary_frame["has_normative_reference"] = (
        primary_frame["normative_match"].ne(0.5).astype(float)
    )
    primary_frame["region_specialization"] = (
        pd.to_numeric(
            features_frame["region_specialization"]
            if "region_specialization" in features_frame.columns
            else pd.Series(0.5, index=raw_frame.index),
            errors="coerce",
        )
        .fillna(0.5)
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

    return primary_frame.reindex(columns=FEATURE_COLUMNS)


def build_merit_sample_weight(
    y: pd.Series | np.ndarray,
    merit_signal: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    y_arr = np.asarray(y).astype(int)
    base_weight = compute_sample_weight(class_weight="balanced", y=y_arr)

    if merit_signal is None:
        return pd.Series(base_weight)

    merit_arr = np.asarray(merit_signal).astype(float)

    consistent_positive = (y_arr == 1) & (merit_arr >= 0.60)
    consistent_negative = (y_arr == 0) & (merit_arr <= 0.50)
    contradictory_positive = (y_arr == 1) & (merit_arr <= 0.45)
    contradictory_negative = (y_arr == 0) & (merit_arr >= 0.70)

    multiplier = np.ones_like(merit_arr, dtype=float)
    multiplier[consistent_positive | consistent_negative] = 1.20
    multiplier[contradictory_positive | contradictory_negative] = 0.45

    distance = np.abs(merit_arr - 0.55)
    confidence = 0.85 + np.clip(distance / 0.30, 0.0, 1.0) * 0.30

    return pd.Series(base_weight * multiplier * confidence)


def build_merit_target(primary_frame: pd.DataFrame) -> pd.DataFrame:
    merit_score = (
        primary_frame["amount_normative_integrity"].astype(float) * 0.30
        + primary_frame["amount_adequacy"].astype(float) * 0.24
        + primary_frame["unit_count"].astype(float) * 0.18
        + primary_frame["amount_typicality"].astype(float) * 0.14
        + primary_frame["has_normative_reference"].astype(float) * 0.08
        + primary_frame["normative_match"].astype(float) * 0.06
    ).clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "merit_proxy_score": merit_score.round(6),
            "merit_proxy_positive": (merit_score >= MERIT_PROXY_POSITIVE_THRESHOLD).astype(int),
        },
        index=primary_frame.index,
    )


def _candidate_models(random_state: int) -> tuple[dict[str, object], dict[str, str]]:
    logistic_preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                PRIMARY_MODEL_CATEGORICAL_COLUMNS,
            ),
            ("num", StandardScaler(), PRIMARY_MODEL_NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )
    forest_preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                PRIMARY_MODEL_CATEGORICAL_COLUMNS,
            ),
            ("num", "passthrough", PRIMARY_MODEL_NUMERIC_COLUMNS),
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
                        n_estimators=350,
                        max_depth=14,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models, {}


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


def build_training_dataset(data_path: str) -> dict[str, object]:
    df = run_pipeline(data_path)
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    rule_scores = score_batch(features)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])
    rule_scores[eligibility.columns] = eligibility
    deadline_disqualified = eligibility["disqualified"].fillna(False).astype(bool)
    eligible_mask = ~deadline_disqualified

    X = build_primary_model_frame(
        raw_input=df.loc[eligible_mask],
        extracted_features=features.loc[eligible_mask],
    )
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

    metadata["sample_weight"] = build_merit_sample_weight(
        y=y,
        merit_signal=metadata["merit_proxy_score"],
    ).to_numpy()

    return {
        "df": df,
        "tables": tables,
        "features": features,
        "rule_scores": rule_scores,
        "X": X,
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
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= threshold).astype(int)

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
        metrics["brier_score"] = round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["brier_score"] = None

    return metrics


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


def choose_probability_temperature(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
) -> dict[str, object]:
    candidate_temperatures = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    candidates: list[dict[str, object]] = []
    for temperature in candidate_temperatures:
        adjusted_prob = apply_probability_temperature(y_prob, temperature)
        metrics = evaluate_predictions(y_true, adjusted_prob)
        candidates.append(
            {
                "temperature": float(temperature),
                "validation_metrics": metrics,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["validation_metrics"]["brier_score"]
            if item["validation_metrics"]["brier_score"] is not None
            else 1e9,
            -(
                item["validation_metrics"]["roc_auc"]
                if item["validation_metrics"]["roc_auc"] is not None
                else -1
            ),
            item["temperature"],
        )
    )
    best = candidates[0]
    return {
        "best_temperature": float(best["temperature"]),
        "best_validation_metrics": best["validation_metrics"],
        "candidates": candidates,
    }


def tune_decision_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, object]:
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
                np.linspace(0.05, 0.95, 37),
                np.quantile(y_prob, np.linspace(0.05, 0.95, 19)),
            ]
        ).round(4)
    )

    candidates: list[dict[str, object]] = []
    for threshold in candidate_thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold=float(threshold))
        candidates.append(metrics)

    best_balanced_accuracy = max(item["balanced_accuracy"] for item in candidates)
    relaxed_candidates = [
        item
        for item in candidates
        if item["balanced_accuracy"] >= best_balanced_accuracy - 0.01
    ]
    best_f1 = max(item["f1"] for item in relaxed_candidates)
    recall_friendly_candidates = [
        item
        for item in relaxed_candidates
        if item["balanced_accuracy"] >= best_balanced_accuracy - 0.01
        and item["f1"] >= best_f1 - 0.015
    ]
    if recall_friendly_candidates:
        recall_friendly_candidates.sort(
            key=lambda item: (
                item["recall"],
                item["balanced_accuracy"],
                item["f1"],
                item["precision"],
                -abs(item["threshold"] - 0.5),
            ),
            reverse=True,
        )
        best = recall_friendly_candidates[0]
    else:
        relaxed_candidates.sort(
            key=lambda item: (
                item["recall"],
                item["balanced_accuracy"],
                item["f1"],
                item["precision"],
                -abs(item["threshold"] - 0.5),
            ),
            reverse=True,
        )
        best = relaxed_candidates[0]
    return {
        "best_threshold": float(best["threshold"]),
        "best_metrics": best,
        "candidates": relaxed_candidates[:15],
        "best_balanced_accuracy": round(float(best_balanced_accuracy), 4),
    }


def _evaluate_calibration(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    method: str,
    sample_weight_train: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    if method == "raw":
        calibrated_model = _fit_model(
            model,
            X_train,
            y_train,
            sample_weight=sample_weight_train,
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=deepcopy(model),
            method=method,
            cv=5,
            ensemble=False,
        )
        fit_kwargs: dict[str, object] = {}
        if sample_weight_train is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight_train)
        calibrated_model.fit(X_train, y_train, **fit_kwargs)

    y_valid_prob = calibrated_model.predict_proba(X_valid)[:, 1]
    metrics = evaluate_predictions(y_valid, y_valid_prob)
    return {
        "method": method,
        "model": calibrated_model,
        "validation_metrics": metrics,
    }


def choose_calibration(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_weight_train: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    candidate_methods = ["raw"]
    if not hasattr(model, "named_steps"):
        candidate_methods.append("sigmoid")

    calibration_candidates = [
        _evaluate_calibration(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            method=method,
            sample_weight_train=sample_weight_train,
        )
        for method in candidate_methods
    ]
    best_auc = max(
        item["validation_metrics"]["roc_auc"] or -1 for item in calibration_candidates
    )
    epsilon = 0.002
    shortlisted = [
        item
        for item in calibration_candidates
        if (item["validation_metrics"]["roc_auc"] or -1) >= best_auc - epsilon
    ]
    best_brier = min(
        item["validation_metrics"]["brier_score"] or 1e9
        for item in shortlisted
    )
    raw_candidate = next(
        (item for item in shortlisted if item["method"] == "raw"),
        None,
    )
    if raw_candidate is not None:
        raw_brier = raw_candidate["validation_metrics"]["brier_score"] or 1e9
        if raw_brier <= best_brier + 0.01:
            best = raw_candidate
        else:
            shortlisted.sort(
                key=lambda item: (
                    item["validation_metrics"]["brier_score"] or 1e9,
                    -(item["validation_metrics"]["roc_auc"] or -1),
                ),
            )
            best = shortlisted[0]
    else:
        shortlisted.sort(
            key=lambda item: (
                item["validation_metrics"]["brier_score"] or 1e9,
                -(item["validation_metrics"]["roc_auc"] or -1),
            ),
        )
        best = shortlisted[0]
    return {
        "best_method": best["method"],
        "best_model": best["model"],
        "best_validation_metrics": best["validation_metrics"],
        "candidates": [
            {
                "method": item["method"],
                "validation_metrics": item["validation_metrics"],
            }
            for item in calibration_candidates
        ],
    }


def tune_blend_weights(
    y_true: pd.Series | np.ndarray,
    rule_scores: pd.Series | np.ndarray,
    ml_probabilities: pd.Series | np.ndarray,
    disqualified_mask: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true).astype(int)
    rule_arr = np.asarray(rule_scores).astype(float)
    ml_score_arr = np.asarray(ml_probabilities).astype(float) * 100
    if disqualified_mask is None:
        disqualified_arr = np.zeros_like(y_true_arr, dtype=bool)
    else:
        disqualified_arr = np.asarray(disqualified_mask).astype(bool)

    candidates: list[dict[str, object]] = []
    for rule_weight in np.linspace(MIN_RULE_BLEND_WEIGHT, 1.0, int((1.0 - MIN_RULE_BLEND_WEIGHT) / 0.05) + 1):
        ml_weight = 1.0 - rule_weight
        blended = np.round(rule_arr * rule_weight + ml_score_arr * ml_weight, 1)
        blended[disqualified_arr] = 0.0

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
    if isinstance(model, DualBranchEnsembleModel):
        return []
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


def select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    train_sample_weight: pd.Series | None = None,
    valid_rule_scores: pd.Series | None = None,
    valid_disqualified_mask: pd.Series | None = None,
    stability_scores: dict[str, dict[str, object]] | None = None,
    candidate_names: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    leaderboard: list[dict[str, object]] = []
    candidate_models, skipped_models = _candidate_models(random_state)
    if candidate_names is not None:
        candidate_models = {
            name: model
            for name, model in candidate_models.items()
            if name in set(candidate_names)
        }

    for model_name, model in candidate_models.items():
        fitted_base_model = _fit_model(
            model,
            X_train,
            y_train,
            sample_weight=train_sample_weight,
        )
        calibration = choose_calibration(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            sample_weight_train=train_sample_weight,
        )
        calibrated_model = calibration["best_model"]
        y_valid_prob_raw = calibrated_model.predict_proba(X_valid)[:, 1]
        temperature_tuning = choose_probability_temperature(y_valid, y_valid_prob_raw)
        y_valid_prob = apply_probability_temperature(
            y_valid_prob_raw,
            temperature_tuning["best_temperature"],
        )
        threshold_tuning = tune_decision_threshold(y_valid, y_valid_prob)
        stability = (stability_scores or {}).get(model_name, {})
        cv_auc = float(stability.get("cv_roc_auc_mean", 0.0))
        cv_std = float(stability.get("cv_roc_auc_std", 0.0))
        validation_auc = float(threshold_tuning["best_metrics"]["roc_auc"] or 0.0)
        validation_brier = float(
            threshold_tuning["best_metrics"]["brier_score"]
            if threshold_tuning["best_metrics"]["brier_score"] is not None
            else 1.0
        )
        composite_score = (
            validation_auc * 0.55
            + cv_auc * 0.25
            + (1.0 - validation_brier) * 0.20
            - cv_std * 0.10
        )

        blend_tuning = None
        if valid_rule_scores is not None:
            blend_tuning = tune_blend_weights(
                y_true=y_valid,
                rule_scores=valid_rule_scores,
                ml_probabilities=y_valid_prob,
                disqualified_mask=valid_disqualified_mask,
            )

        leaderboard.append(
            {
                "model_name": model_name,
                "calibration": {
                "best_method": calibration["best_method"],
                "candidates": calibration["candidates"],
            },
            "probability_temperature": temperature_tuning,
            "validation_metrics": threshold_tuning["best_metrics"],
            "threshold_tuning": threshold_tuning,
            "validation_blend": blend_tuning,
                "stability_metrics": stability,
                "selection_score": round(float(composite_score), 6),
                "feature_importance": _extract_feature_importance(
                    fitted_base_model, FEATURE_COLUMNS
                )[:10],
                "_model": calibrated_model,
                "_base_model": fitted_base_model,
            }
        )

    leaderboard.sort(
        key=lambda item: (
            item["selection_score"],
            item["validation_metrics"]["roc_auc"] is not None,
            item["validation_metrics"]["roc_auc"] or -1,
            item["validation_metrics"]["balanced_accuracy"],
            item["validation_metrics"]["f1"],
        ),
        reverse=True,
    )
    best_entry = leaderboard[0]
    best_name = best_entry["model_name"]

    return {
        "best_model_name": best_name,
        "best_validation_metrics": best_entry["validation_metrics"],
        "best_threshold": best_entry["threshold_tuning"]["best_threshold"],
        "best_calibration_method": best_entry["calibration"]["best_method"],
        "best_probability_temperature": best_entry["probability_temperature"]["best_temperature"],
        "best_blend_weights": (
            best_entry["validation_blend"]["best_weights"]
            if best_entry["validation_blend"] is not None
            else DEFAULT_BLEND_WEIGHTS
        ),
        "best_model": best_entry["_model"],
        "best_base_model": best_entry["_base_model"],
        "leaderboard": [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in leaderboard
        ],
        "skipped_models": skipped_models,
    }


def train_final_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    random_state: int = 42,
) -> object:
    candidates, _ = _candidate_models(random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown model_name: {model_name}")

    return _fit_model(candidates[model_name], X, y, sample_weight=sample_weight)


def train_calibrated_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    calibration_method: str = "sigmoid",
    random_state: int = 42,
) -> object:
    candidates, _ = _candidate_models(random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown model_name: {model_name}")
    if calibration_method == "raw":
        return train_final_model(
            model_name=model_name,
            X=X,
            y=y,
            sample_weight=sample_weight,
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


def prepare_feature_frame(features_input: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    feature_frame = _coerce_dataframe(features_input)
    return feature_frame.reindex(columns=FEATURE_COLUMNS).copy()


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
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    probability_temperature: float = 1.0,
    disqualified_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    if blend_weights is None:
        blend_weights = DEFAULT_BLEND_WEIGHTS

    feature_frame = prepare_feature_frame(features_input)
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

    result = pd.DataFrame(index=feature_frame.index)
    result["rule_score"] = rule_scores["score"].astype(float)
    result["rule_risk_level"] = rule_scores["risk_level"].astype(str)
    result["ml_probability"] = np.nan
    result["ml_score"] = np.nan
    result["ml_decision_threshold"] = float(decision_threshold)
    result["ml_predicted_positive"] = False
    result["final_score"] = np.round(
        rule_scores["score"].astype(float) * blend_weights["rule_score"],
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
        ml_prob = apply_probability_temperature(
            ml_prob_raw,
            temperature=probability_temperature,
        )
        ml_score = np.round(ml_prob * 100, 1)
        result.loc[eligible_mask, "ml_probability"] = np.round(ml_prob, 4)
        result.loc[eligible_mask, "ml_score"] = ml_score
        result.loc[eligible_mask, "ml_predicted_positive"] = (
            result.loc[eligible_mask, "ml_probability"] >= float(decision_threshold)
        )
        result.loc[eligible_mask, "final_score"] = np.round(
            result.loc[eligible_mask, "rule_score"].astype(float) * blend_weights["rule_score"]
            + result.loc[eligible_mask, "ml_score"].astype(float) * blend_weights["ml_score"],
            1,
        )
        result.loc[eligible_mask, "final_risk_level"] = result.loc[
            eligible_mask, "final_score"
        ].apply(score_to_risk_label)

    disqualified_mask = result["disqualified"].fillna(False)
    if disqualified_mask.any():
        result.loc[disqualified_mask, "final_score"] = 0.0
        result.loc[disqualified_mask, "final_risk_level"] = "Высокий"
        result.loc[disqualified_mask, "ml_predicted_positive"] = False

    result["score"] = result["final_score"]
    result["risk_level"] = result["final_risk_level"]

    return result


def build_prediction_frame(
    df: pd.DataFrame,
    tables: dict,
    model: object,
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    probability_temperature: float = 1.0,
) -> pd.DataFrame:
    features = extract_features_batch(df, tables)
    model_input = build_primary_model_frame(df, features)
    advisory = build_history_advisory_batch(df)
    rule_scores = score_batch(features)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])
    rule_scores[eligibility.columns] = eligibility
    blended_scores = score_features_with_model(
        model_input,
        model=model,
        rule_scores=rule_scores,
        blend_weights=blend_weights,
        decision_threshold=decision_threshold,
        probability_temperature=probability_temperature,
        disqualified_mask=rule_scores["disqualified"],
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
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    probability_temperature: float = 1.0,
    calibration_method: str | None = None,
    report: dict[str, object] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns or FEATURE_COLUMNS,
        "tables": tables,
        "blend_weights": blend_weights or DEFAULT_BLEND_WEIGHTS,
        "decision_threshold": float(decision_threshold),
        "probability_temperature": float(probability_temperature),
        "calibration_method": calibration_method,
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
