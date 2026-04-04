from __future__ import annotations

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


RULE_FEATURE_COLUMNS = list(WEIGHTS.keys())
PRIMARY_MODEL_CATEGORICAL_COLUMNS = [
    "subsidy_type",
]
PRIMARY_MODEL_NUMERIC_COLUMNS = [
    "normative_log",
    "amount_adequacy",
    "normative_match",
    "normative_original_match",
    "normative_reference_gap",
    "normative_reference_typicality",
    "amount_normative_integrity",
    "unit_count_log",
    "unit_count_original_log",
    "submit_month_sin",
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
    "adequacy_x_direction_rate",
    "adequacy_x_budget_pressure",
    "rule_score_feature",
    "contrib_normative_match",
    "contrib_amount_adequacy",
    "contrib_budget_pressure",
    "contrib_queue_position",
]
FEATURE_COLUMNS = PRIMARY_MODEL_CATEGORICAL_COLUMNS + PRIMARY_MODEL_NUMERIC_COLUMNS
MERIT_PROXY_FEATURE_COLUMNS = [
    "amount_adequacy",
    "amount_to_normative_ratio",
    "unit_count_log",
]
PROCESS_BIASED_FEATURE_COLUMNS: list[str] = []
DEFAULT_BLEND_WEIGHTS = {"rule_score": 0.25, "ml_score": 0.75}
MIN_RULE_BLEND_WEIGHT = 0.25
MERIT_PROXY_POSITIVE_THRESHOLD = 0.68
REGIONAL_SIGNAL_SHRINK = 0.35
RULE_AUGMENTATION_COLUMNS = [
    "contrib_normative_match",
    "contrib_amount_adequacy",
    "contrib_budget_pressure",
    "contrib_queue_position",
]


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
    rule_scores: dict | pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw_frame = _coerce_dataframe(raw_input)
    features_frame = _coerce_feature_dataframe(extracted_features, raw_frame.index)
    rule_frame = _coerce_feature_dataframe(rule_scores, raw_frame.index)

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

    return primary_frame.reindex(columns=FEATURE_COLUMNS)


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
        rule_scores=rule_scores.loc[eligible_mask],
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
            -abs(float(item["threshold"]) - 0.5),
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


def prepare_feature_frame(features_input: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    feature_frame = _coerce_dataframe(features_input)
    return feature_frame.reindex(columns=FEATURE_COLUMNS).copy()


def explain_prediction_with_model(
    features_input: dict | pd.Series | pd.DataFrame,
    model: object,
    neutral_values: dict[str, object] | None = None,
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float | None = None,
) -> dict[str, object]:
    feature_frame = prepare_feature_frame(features_input)
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
    blend_weights: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    probability_calibrator: ProbabilityCalibrator | None = None,
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
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float = 1.0,
) -> pd.DataFrame:
    features = extract_features_batch(df, tables)
    advisory = build_history_advisory_batch(df)
    rule_scores = score_batch(features)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])
    rule_scores[eligibility.columns] = eligibility
    model_input = build_primary_model_frame(
        raw_input=df,
        extracted_features=features,
        rule_scores=rule_scores,
    )
    blended_scores = score_features_with_model(
        model_input,
        model=model,
        rule_scores=rule_scores,
        blend_weights=blend_weights,
        decision_threshold=decision_threshold,
        probability_calibrator=probability_calibrator,
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
    probability_calibrator: ProbabilityCalibrator | None = None,
    probability_temperature: float = 1.0,
    calibration_method: str | None = None,
    explanation_neutral_values: dict[str, object] | None = None,
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
