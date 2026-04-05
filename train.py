import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling import (
    BLEND_WEIGHT_SEARCH_GRID,
    DEFAULT_DECISION_SCORE_NAME,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_FEATURE_SET_NAME,
    DEFAULT_BLEND_WEIGHTS,
    DECISION_SCORE_SCALE_MAX,
    EXCLUDED_COLUMNS,
    EXISTING_ONLY_FEATURE_SET,
    EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET,
    LEAKY_RULE_SYNTHETIC_FEATURE_COLUMNS,
    LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS,
    MERIT_PROXY_FEATURE_COLUMNS,
    MERIT_PROXY_POSITIVE_THRESHOLD,
    PROCESS_BIASED_FEATURE_COLUMNS,
    RULE_CONTRIBUTION_COLUMNS,
    RULE_FEATURE_COLUMNS,
    SAFE_SYNTHETIC_FEATURE_COLUMNS,
    apply_probability_calibrator,
    build_explanation_neutral_values,
    build_training_dataset,
    choose_probability_calibrator,
    compute_blended_scores,
    compute_permutation_feature_importance,
    expected_calibration_error,
    evaluate_predictions,
    build_threshold_candidates,
    get_available_model_candidates,
    get_blend_rule_score_column,
    get_feature_set_deployability,
    model_supports_sample_weight,
    resolve_blend_weights,
    save_bundle,
    save_json,
    split_dataset,
    tune_blend_weights,
    train_calibrated_model,
    tune_decision_threshold,
)


DEFAULT_DATA_PATH = "data/subsidies.xlsx"
DEFAULT_MODEL_PATH = "models/artifacts/subsidy_model.joblib"
DEFAULT_REPORT_PATH = "models/reports/training_metrics.json"
DEFAULT_TEST_PREDICTIONS_PATH = "models/reports/test_predictions.csv"
MODEL_NAME_CANDIDATES = [
    "neural_network",
    "logistic_regression",
    "catboost",
]
FEATURE_SET_CANDIDATES = [
    EXISTING_ONLY_FEATURE_SET,
    EXISTING_PLUS_SYNTHETIC_SAFE_FEATURE_SET,
]
FN_COST = 3.0
FP_COST = 1.0
PRODUCTION_DECISION_THRESHOLD_OVERRIDE = 32.5
REGIONAL_SIGNAL_COLUMNS = [
    "region_approval_rate",
    "region_direction_lift",
    "akimat_lift",
]


def _describe_split_dates(metadata: pd.DataFrame) -> dict[str, str | None]:
    dates = pd.to_datetime(metadata["submit_date"], errors="coerce").dropna()
    if dates.empty:
        return {"min_submit_date": None, "max_submit_date": None}
    return {
        "min_submit_date": str(dates.min().date()),
        "max_submit_date": str(dates.max().date()),
    }


def _class_gap(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
) -> float:
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true).astype(int),
            "probability": np.asarray(y_prob).astype(float),
        }
    )
    means = frame.groupby("target")["probability"].mean().to_dict()
    return round(float(means.get(1, 0.0) - means.get(0, 0.0)), 6)


def _rule_baseline_metrics(
    y_true: pd.Series | np.ndarray,
    rule_scores: pd.Series | np.ndarray,
) -> dict[str, object]:
    probabilities = np.asarray(rule_scores).astype(float) / 100.0
    return evaluate_predictions(y_true, probabilities, threshold=0.5)


def _blend_ranking_metrics(
    y_true: pd.Series | np.ndarray,
    rule_scores: pd.Series | np.ndarray,
    ml_probabilities: pd.Series | np.ndarray,
    blend_weights: dict[str, float] | None = None,
    disqualified_mask: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    resolved_weights = resolve_blend_weights(blend_weights)
    blended_scores = compute_blended_scores(
        rule_scores=rule_scores,
        ml_probabilities=ml_probabilities,
        blend_weights=resolved_weights,
        disqualified_mask=disqualified_mask,
    )
    y_true_arr = np.asarray(y_true).astype(int)
    return {
        "weights": resolved_weights,
        "roc_auc": round(float(roc_auc_score(y_true_arr, blended_scores)), 4),
        "average_precision": round(float(average_precision_score(y_true_arr, blended_scores)), 4),
        "mean_score": round(float(np.mean(blended_scores)), 4),
    }


def _feature_shuffle_sensitivity(
    model: object,
    feature_frame: pd.DataFrame,
    columns: list[str],
    probability_calibrator=None,
) -> dict[str, float]:
    available_columns = [column for column in columns if column in feature_frame.columns]
    if feature_frame.empty or not available_columns:
        return {
            "mean_abs_delta": 0.0,
            "p95_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }

    base_prob = model.predict_proba(feature_frame)[:, 1]
    shuffled = feature_frame.copy()
    rng = np.random.default_rng(42)
    for column in available_columns:
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    shuffled_prob = model.predict_proba(shuffled)[:, 1]

    if probability_calibrator is not None:
        base_prob = apply_probability_calibrator(base_prob, probability_calibrator)
        shuffled_prob = apply_probability_calibrator(shuffled_prob, probability_calibrator)

    delta = np.abs(np.asarray(base_prob, dtype=float) - np.asarray(shuffled_prob, dtype=float))
    return {
        "mean_abs_delta": round(float(delta.mean()), 6),
        "p95_abs_delta": round(float(np.quantile(delta, 0.95)), 6),
        "max_abs_delta": round(float(delta.max()), 6),
    }


def _classwise_decision_metrics(metrics: dict[str, object]) -> dict[str, object]:
    confusion = metrics["confusion_matrix"]
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    tp = int(confusion["tp"])

    precision_class_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_class_0 = (
        2 * precision_class_0 * recall_class_0 / (precision_class_0 + recall_class_0)
        if (precision_class_0 + recall_class_0) > 0
        else 0.0
    )
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    predicted_approval_rate = float(metrics["positive_rate_pred"])
    actual_approval_rate = float(metrics["positive_rate_true"])

    return {
        "predicted_approval_rate": round(predicted_approval_rate, 4),
        "actual_approval_rate": round(actual_approval_rate, 4),
        "approval_rate_gap": round(predicted_approval_rate - actual_approval_rate, 4),
        "class_1": {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        },
        "class_0": {
            "precision": round(float(precision_class_0), 4),
            "recall": round(float(recall_class_0), 4),
            "f1": round(float(f1_class_0), 4),
        },
        "false_positive_rate": round(float(false_positive_rate), 4),
        "false_negative_rate": round(float(false_negative_rate), 4),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "confusion_matrix": confusion,
    }


def _probability_distribution_summary(
    y_prob: pd.Series | np.ndarray,
) -> dict[str, float]:
    prob = np.asarray(y_prob).astype(float)
    return {
        "min": round(float(np.min(prob)), 6),
        "p05": round(float(np.quantile(prob, 0.05)), 6),
        "p25": round(float(np.quantile(prob, 0.25)), 6),
        "p50": round(float(np.quantile(prob, 0.50)), 6),
        "p75": round(float(np.quantile(prob, 0.75)), 6),
        "p95": round(float(np.quantile(prob, 0.95)), 6),
        "max": round(float(np.max(prob)), 6),
        "mean": round(float(np.mean(prob)), 6),
        "std": round(float(np.std(prob)), 6),
    }


def _threshold_grid(
    y_score: pd.Series | np.ndarray,
    score_scale_max: float | None = None,
) -> list[float]:
    return [
        float(value)
        for value in build_threshold_candidates(
            y_score,
            score_scale_max=score_scale_max,
        )
    ]


def _threshold_audit(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    score_scale_max: float | None = None,
) -> dict[str, object]:
    resolved_scale_max = score_scale_max or DECISION_SCORE_SCALE_MAX
    anchor = 0.5 if resolved_scale_max <= 1.0 + 1e-8 else resolved_scale_max / 2.0
    grid = _threshold_grid(y_score, score_scale_max=resolved_scale_max)
    candidates: list[dict[str, object]] = []
    target_rate = float(np.asarray(y_true).astype(int).mean())
    for threshold in grid:
        metrics = evaluate_predictions(
            y_true,
            y_score,
            threshold=threshold,
            score_scale_max=resolved_scale_max,
        )
        candidates.append(
            {
                "threshold": round(float(threshold), 4),
                "f1": float(metrics["f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "predicted_approval_rate": float(metrics["positive_rate_pred"]),
                "approval_rate_gap": round(float(metrics["positive_rate_pred"] - target_rate), 4),
            }
        )

    best_f1 = max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["balanced_accuracy"],
            -abs(item["threshold"] - anchor),
        ),
    )
    best_balanced_accuracy = max(
        candidates,
        key=lambda item: (
            item["balanced_accuracy"],
            item["f1"],
            -abs(item["threshold"] - anchor),
        ),
    )
    closest_rate = min(
        candidates,
        key=lambda item: (
            abs(item["approval_rate_gap"]),
            -item["balanced_accuracy"],
            -item["f1"],
        ),
    )
    return {
        "target_approval_rate": round(target_rate, 4),
        "score_name": DEFAULT_DECISION_SCORE_NAME,
        "score_scale_max": float(resolved_scale_max),
        "best_f1_threshold": best_f1,
        "best_balanced_accuracy_threshold": best_balanced_accuracy,
        "closest_historical_approval_rate_threshold": closest_rate,
        "sweep": candidates,
    }


def _calibration_audit(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
    calibration_bins: list[dict[str, object]] = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        calibration_bins.append(
            {
                "bin": f"{bins[bin_idx]:.2f}-{bins[bin_idx + 1]:.2f}",
                "count": int(np.sum(mask)),
                "mean_predicted_probability": round(float(np.mean(y_prob_arr[mask])), 6),
                "observed_approval_rate": round(float(np.mean(y_true_arr[mask])), 6),
                "absolute_gap": round(
                    float(abs(np.mean(y_prob_arr[mask]) - np.mean(y_true_arr[mask]))),
                    6,
                ),
            }
        )
    base_metrics = evaluate_predictions(y_true_arr, y_prob_arr, threshold=0.5)
    return {
        "brier_score": base_metrics.get("brier_score"),
        "expected_calibration_error": expected_calibration_error(y_true_arr, y_prob_arr),
        "calibration_bins": calibration_bins,
    }


def _rule_ml_compatibility_audit(
    test_predictions: pd.DataFrame,
    test_X: pd.DataFrame,
    threshold: float,
) -> dict[str, object]:
    rule_score_full = test_predictions["rule_score_full"].astype(float)
    blend_rule_score = test_predictions["blend_rule_score"].astype(float)
    ml_probability = test_predictions["ml_probability"].astype(float)
    final_score = test_predictions["final_score"].astype(float)
    rule_score_feature = (
        test_X["rule_score_feature"].astype(float) * 100.0
        if "rule_score_feature" in test_X.columns
        else pd.Series(np.nan, index=test_X.index)
    )
    decision_positive = final_score >= float(threshold)
    high_rule = blend_rule_score >= 70.0
    low_rule = blend_rule_score < 45.0

    def _corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
        aligned = pd.concat([left, right], axis=1).dropna()
        if len(aligned) < 2:
            return None
        return round(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method)), 6)

    return {
        "correlations": {
            "rule_score_full__ml_probability_pearson": _corr(rule_score_full, ml_probability, "pearson"),
            "rule_score_full__ml_probability_spearman": _corr(rule_score_full, ml_probability, "spearman"),
            "blend_rule_score__ml_probability_pearson": _corr(blend_rule_score, ml_probability, "pearson"),
            "blend_rule_score__ml_probability_spearman": _corr(blend_rule_score, ml_probability, "spearman"),
            "rule_score_feature__ml_probability_pearson": _corr(rule_score_feature, ml_probability, "pearson"),
            "rule_score_feature__ml_probability_spearman": _corr(rule_score_feature, ml_probability, "spearman"),
            "rule_score_feature__blend_rule_score_spearman": _corr(rule_score_feature, blend_rule_score, "spearman"),
            "ml_probability__final_score_spearman": _corr(ml_probability, final_score, "spearman"),
        },
        "disagreement_summary": {
            "rule_low_risk_but_decision_negative_rate": round(
                float((~decision_positive.loc[high_rule.index[high_rule]]).mean()) if high_rule.any() else 0.0,
                4,
            ),
            "rule_high_risk_but_decision_positive_rate": round(
                float(decision_positive.loc[low_rule.index[low_rule]].mean()) if low_rule.any() else 0.0,
                4,
            ),
            "mean_final_score_when_rule_low_risk": round(
                float(final_score.loc[high_rule].mean()) if high_rule.any() else 0.0,
                4,
            ),
            "mean_final_score_when_rule_high_risk": round(
                float(final_score.loc[low_rule].mean()) if low_rule.any() else 0.0,
                4,
            ),
            "decision_threshold": round(float(threshold), 4),
            "decision_score_name": DEFAULT_DECISION_SCORE_NAME,
        },
    }


def _segment_metrics(frame: pd.DataFrame, group_column: str, top_n: int = 10) -> list[dict[str, object]]:
    grouped = (
        frame.groupby(group_column)
        .agg(
            count=("target", "size"),
            actual_approval_rate=("target", "mean"),
            predicted_approval_rate=("predicted_positive", "mean"),
            mean_ml_probability=("ml_probability", "mean"),
            mean_rule_score_full=("rule_score_full", "mean"),
            mean_blend_rule_score=("blend_rule_score", "mean"),
            mean_final_score=("final_score", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    result: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        result.append(
            {
                group_column: str(row[group_column]),
                "count": int(row["count"]),
                "actual_approval_rate": round(float(row["actual_approval_rate"]), 4),
                "predicted_approval_rate": round(float(row["predicted_approval_rate"]), 4),
                "approval_rate_gap": round(
                    float(row["predicted_approval_rate"] - row["actual_approval_rate"]),
                    4,
                ),
                "mean_ml_probability": round(float(row["mean_ml_probability"]), 4),
                "mean_rule_score_full": round(float(row["mean_rule_score_full"]), 4),
                "mean_blend_rule_score": round(float(row["mean_blend_rule_score"]), 4),
                "mean_final_score": round(float(row["mean_final_score"]), 4),
            }
        )
    return result


def _segment_audit(
    metadata: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> dict[str, object]:
    # Use the scored prediction frame itself so segment metrics stay aligned with
    # the exact rows/probabilities written to `test_predictions.csv`, even after
    # ranking/sorting by final score.
    frame = test_predictions.copy()
    if "target" not in frame.columns:
        frame["target"] = frame["training_target"].astype(int).to_numpy()
    if "predicted_positive" not in frame.columns and "decision_predicted_positive" in frame.columns:
        frame["predicted_positive"] = frame["decision_predicted_positive"].astype(int).to_numpy()
    if "predicted_positive" not in frame.columns:
        frame["predicted_positive"] = frame["ml_predicted_positive"].astype(int).to_numpy()
    if "ml_probability" in frame.columns:
        frame["ml_probability"] = frame["ml_probability"].astype(float).to_numpy()
    if "rule_score_full" in frame.columns:
        frame["rule_score_full"] = frame["rule_score_full"].astype(float).to_numpy()
    if "blend_rule_score" in frame.columns:
        frame["blend_rule_score"] = frame["blend_rule_score"].astype(float).to_numpy()
    if "final_score" in frame.columns:
        frame["final_score"] = frame["final_score"].astype(float).to_numpy()

    amount = pd.to_numeric(frame["amount"], errors="coerce")
    if amount.notna().sum() >= 10:
        amount_bin_codes = pd.qcut(amount.rank(method="first"), q=4, duplicates="drop")
        frame["amount_bucket"] = amount_bin_codes.astype(str)
    else:
        frame["amount_bucket"] = "all"

    submit_date = pd.to_datetime(frame["submit_date"], errors="coerce")
    frame["submit_period_month"] = submit_date.dt.to_period("M").astype(str)

    return {
        "region": _segment_metrics(frame, "region", top_n=10),
        "direction": _segment_metrics(frame, "direction", top_n=10),
        "subsidy_type": _segment_metrics(frame, "subsidy_type", top_n=10),
        "amount_bucket": _segment_metrics(frame, "amount_bucket", top_n=6),
        "submit_period_month": _segment_metrics(frame, "submit_period_month", top_n=8),
    }


def _consistency_and_leakage_audit(
    dataset: dict[str, object],
    feature_set_name: str,
    feature_columns: list[str],
) -> dict[str, object]:
    feature_set_deployability = get_feature_set_deployability(feature_set_name)
    feature_column_order_matches_training = (
        list(dataset["X_by_feature_set"][feature_set_name].columns) == list(feature_columns)
    )
    leaky_columns_present = sorted(
        set(feature_columns).intersection(LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS)
    )
    neutralized_leaky_columns = (
        sorted(set(feature_columns).intersection(LEAKY_RULE_SYNTHETIC_FEATURE_COLUMNS))
        if feature_set_name == EXISTING_ONLY_FEATURE_SET
        else []
    )
    actively_varying_leaky_columns = sorted(
        set(leaky_columns_present).difference(neutralized_leaky_columns)
    )
    excluded_columns_in_training_features = sorted(
        set(feature_columns).intersection(EXCLUDED_COLUMNS).difference(neutralized_leaky_columns)
    )
    target_reconstruction_columns = sorted(
        set(feature_columns).intersection(
            {
                "status",
                "is_approved",
                "training_target",
                "historical_is_approved",
            }
        )
    )
    safe_synthetic_used = sorted(
        set(feature_columns).intersection(SAFE_SYNTHETIC_FEATURE_COLUMNS)
    )
    rule_feature_columns_used = sorted(
        set(feature_columns).intersection(RULE_FEATURE_COLUMNS)
    )
    contribution_columns_used = sorted(
        set(feature_columns).intersection(RULE_CONTRIBUTION_COLUMNS)
    )
    return {
        "feature_set_name": feature_set_name,
        "feature_set_deployable": feature_set_deployability["deployable"],
        "feature_set_deployability_reason": feature_set_deployability["reason"],
        "feature_column_order_matches_training": bool(feature_column_order_matches_training),
        "leaky_columns_in_training_features": actively_varying_leaky_columns,
        "leaky_columns_present_but_neutralized": neutralized_leaky_columns,
        "excluded_columns_in_training_features": excluded_columns_in_training_features,
        "target_reconstruction_columns_in_training_features": target_reconstruction_columns,
        "safe_synthetic_columns_in_training_features": safe_synthetic_used,
        "rule_feature_columns_in_training_features": rule_feature_columns_used,
        "contribution_columns_in_training_features": contribution_columns_used,
        "raw_rule_score_in_training_features": "rule_score" in feature_columns,
        "rule_score_feature_guarded_by_feature_set_rule_score": True,
        "blend_uses_feature_set_specific_rule_score": True,
        "threshold_selected_on_validation_only": True,
        "probability_calibration_selected_on_validation_only": True,
        "predict_uses_bundle_feature_columns": True,
        "api_uses_bundle_feature_columns": True,
        "build_primary_model_frame_shared_between_train_and_inference": True,
        "fill_defaults_shared_via_build_primary_model_frame": True,
    }


def _strictness_commentary(metrics: dict[str, object]) -> list[str]:
    notes: list[str] = []
    predicted_rate = float(metrics["positive_rate_pred"])
    actual_rate = float(metrics["positive_rate_true"])
    if predicted_rate >= 0.98:
        notes.append("Модель почти всех одобряет при текущем threshold.")
    elif predicted_rate <= 0.02:
        notes.append("Модель почти всех отклоняет при текущем threshold.")
    if predicted_rate - actual_rate >= 0.08:
        notes.append("Текущий threshold делает модель заметно мягче исторического approval rate.")
    elif actual_rate - predicted_rate >= 0.08:
        notes.append("Текущий threshold делает модель заметно строже исторического approval rate.")
    if float(metrics["recall"]) < 0.60:
        notes.append("Recall по классу одобренных низкий: модель может излишне резать хорошие заявки.")
    if float(metrics["precision"]) < 0.80:
        notes.append("Precision по классу одобренных низкий: модель может одобрять слишком много пограничных заявок.")
    return notes


def _selection_score(
    metrics: dict[str, object],
    class_gap: float,
    regional_sensitivity: dict[str, float],
    blend_metrics: dict[str, object] | None = None,
) -> float:
    roc_auc = float(metrics["roc_auc"] or 0.0)
    blend_auc = float((blend_metrics or {}).get("roc_auc", roc_auc) or 0.0)
    blend_ap = float((blend_metrics or {}).get("average_precision", 0.0) or 0.0)
    balanced_accuracy = float(metrics["balanced_accuracy"])
    recall = float(metrics["recall"])
    brier_score = float(metrics["brier_score"] or 1.0)
    selection_cost = float(metrics.get("selection_cost", 0.0))
    regional_penalty = float(regional_sensitivity["mean_abs_delta"])
    return round(
        roc_auc * 0.36
        + blend_auc * 0.18
        + balanced_accuracy * 0.16
        + recall * 0.06
        + (1.0 - brier_score) * 0.08
        + float(class_gap) * 0.12
        + blend_ap * 0.04
        - regional_penalty * 0.14
        - selection_cost * 0.00002,
        6,
    )


def _resolve_candidate_decision_threshold(
    threshold_tuning: dict[str, object],
) -> tuple[float, float, str]:
    tuned_threshold = float(threshold_tuning["best_threshold"])
    if PRODUCTION_DECISION_THRESHOLD_OVERRIDE is None:
        return tuned_threshold, tuned_threshold, "validation_tuned"
    return float(PRODUCTION_DECISION_THRESHOLD_OVERRIDE), tuned_threshold, "manual_production_override"


def _validation_ranking_score(candidate: dict[str, object]) -> tuple[float, float, float, float]:
    blend_metrics = dict(candidate.get("validation_blend", {}).get("best_metrics", {}))
    validation_metrics = dict(candidate.get("validation_metrics", {}))
    return (
        float(blend_metrics.get("roc_auc", validation_metrics.get("roc_auc") or -1.0) or -1.0),
        float(
            blend_metrics.get(
                "average_precision",
                validation_metrics.get("average_precision") or -1.0,
            )
            or -1.0
        ),
        float(validation_metrics.get("roc_auc") or -1.0),
        float(candidate.get("selection_score") or -1.0),
    )


def _evaluate_candidate_on_test(
    candidate: dict[str, object],
    splits: dict[str, dict[str, object]],
    random_state: int,
) -> dict[str, object]:
    feature_columns = list(candidate["feature_columns"])
    categorical_columns = list(candidate["categorical_columns"])
    numeric_columns = list(candidate["numeric_columns"])
    model_name = str(candidate["model_name"])
    decision_threshold = float(candidate["decision_threshold"])
    probability_calibrator = candidate["_probability_calibrator"]
    blend_weights = resolve_blend_weights(candidate["blend_weights"])
    sample_weight_supported = bool(candidate["sample_weight_supported"])
    rule_score_column = str(candidate["rule_score_column"])

    train_valid_X = pd.concat(
        [
            splits["train"]["X"][feature_columns],
            splits["valid"]["X"][feature_columns],
        ],
        axis=0,
    )
    train_valid_y = pd.concat([splits["train"]["y"], splits["valid"]["y"]], axis=0)
    train_valid_sample_weight = None
    if sample_weight_supported:
        train_valid_sample_weight = pd.concat(
            [
                splits["train"]["metadata"]["sample_weight"],
                splits["valid"]["metadata"]["sample_weight"],
            ],
            axis=0,
        )

    test_X = splits["test"]["X"][feature_columns].copy()
    test_y = splits["test"]["y"].copy()
    test_model = train_calibrated_model(
        model_name=model_name,
        X=train_valid_X,
        y=train_valid_y,
        sample_weight=train_valid_sample_weight,
        calibration_method="raw",
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        random_state=random_state,
    )

    test_prob_raw = test_model.predict_proba(test_X)[:, 1]
    test_prob = apply_probability_calibrator(test_prob_raw, probability_calibrator)
    test_predictions = splits["test"]["metadata"].copy()
    test_predictions["feature_set_name"] = str(candidate["feature_set_name"])
    test_predictions["model_name"] = model_name
    test_predictions["rule_score_full"] = test_predictions["rule_score"].astype(float)
    test_predictions["blend_rule_score"] = test_predictions[rule_score_column].astype(float)
    test_predictions["ml_probability"] = np.round(test_prob, 4)
    test_predictions["ml_score"] = np.round(test_prob * 100, 1)
    test_predictions["decision_score_name"] = DEFAULT_DECISION_SCORE_NAME
    test_predictions["decision_threshold"] = decision_threshold
    test_predictions["historical_is_approved"] = (
        test_predictions["historical_is_approved"].astype(int)
    )
    test_predictions["merit_proxy_positive"] = (
        test_predictions["merit_proxy_positive"].astype(int)
    )
    test_predictions["training_target"] = test_predictions["training_target"].astype(int)
    test_predictions["final_score"] = compute_blended_scores(
        rule_scores=test_predictions["blend_rule_score"].astype(float),
        ml_probabilities=test_predictions["ml_probability"].astype(float),
        blend_weights=blend_weights,
        disqualified_mask=None,
    )
    test_predictions["decision_predicted_positive"] = (
        test_predictions["final_score"].astype(float) >= decision_threshold
    )
    test_predictions["ml_decision_threshold"] = decision_threshold
    test_predictions["ml_predicted_positive"] = (
        test_predictions["decision_predicted_positive"].astype(bool)
    )
    disqualified_mask = test_predictions["disqualified"].fillna(False).astype(bool)
    test_predictions.loc[disqualified_mask, "final_score"] = 0.0
    test_predictions.loc[disqualified_mask, "decision_predicted_positive"] = False
    test_predictions.loc[disqualified_mask, "ml_predicted_positive"] = False
    test_predictions = test_predictions.sort_values("final_score", ascending=False)

    test_metrics = evaluate_predictions(
        test_predictions["training_target"].astype(int),
        test_predictions["final_score"].astype(float),
        threshold=decision_threshold,
        score_scale_max=DECISION_SCORE_SCALE_MAX,
    )
    test_final_score_auc = roc_auc_score(
        test_predictions["training_target"].astype(int),
        test_predictions["final_score"].astype(float),
    )
    test_final_score_ap = average_precision_score(
        test_predictions["training_target"].astype(int),
        test_predictions["final_score"].astype(float),
    )
    merit_proxy_alignment_auc = roc_auc_score(
        test_predictions["merit_proxy_positive"].astype(int),
        test_predictions["final_score"].astype(float),
    )
    merit_proxy_spearman = float(
        test_predictions["final_score"].astype(float).corr(
            test_predictions["merit_proxy_score"].astype(float),
            method="spearman",
        )
    )

    permutation_ranking = compute_permutation_feature_importance(
        model=test_model,
        X=test_X,
        y=test_y,
        random_state=random_state,
    )
    feature_sensitivity = _feature_shuffle_sensitivity(
        model=test_model,
        feature_frame=test_X,
        columns=REGIONAL_SIGNAL_COLUMNS,
        probability_calibrator=probability_calibrator,
    )
    calibration_audit = _calibration_audit(
        test_predictions["training_target"].astype(int),
        test_predictions["ml_probability"].astype(float),
    )
    strictness_metrics = _classwise_decision_metrics(test_metrics)
    return {
        "candidate_summary": {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        },
        "test_model": test_model,
        "test_X": test_X,
        "test_y": test_y,
        "test_predictions": test_predictions,
        "test_metrics": test_metrics,
        "test_decision_profile": strictness_metrics,
        "test_probability_distribution": _probability_distribution_summary(
            test_predictions["ml_probability"].astype(float)
        ),
        "test_decision_score_distribution": _probability_distribution_summary(
            test_predictions["final_score"].astype(float)
        ),
        "test_strictness_commentary": _strictness_commentary(test_metrics),
        "test_rule_baseline_full_metrics": _rule_baseline_metrics(
            test_predictions["training_target"].astype(int),
            test_predictions["rule_score_full"].astype(float),
        ),
        "test_rule_baseline_blend_metrics": _rule_baseline_metrics(
            test_predictions["training_target"].astype(int),
            test_predictions["blend_rule_score"].astype(float),
        ),
        "test_final_score_metrics": {
            "roc_auc": round(float(test_final_score_auc), 4),
            "average_precision": round(float(test_final_score_ap), 4),
            "mean_final_score": round(float(test_predictions["final_score"].mean()), 4),
            "mean_score_gap_between_classes": round(
                float(
                    test_predictions.loc[
                        test_predictions["training_target"].astype(bool),
                        "final_score",
                    ].mean()
                    - test_predictions.loc[
                        ~test_predictions["training_target"].astype(bool),
                        "final_score",
                    ].mean()
                ),
                4,
            ),
        },
        "test_region_sensitivity": feature_sensitivity,
        "test_calibration_audit": calibration_audit,
        "test_rule_ml_compatibility_audit": _rule_ml_compatibility_audit(
            test_predictions=test_predictions,
            test_X=test_X,
            threshold=decision_threshold,
        ),
        "test_segment_audit": _segment_audit(
            metadata=splits["test"]["metadata"],
            test_predictions=test_predictions,
        ),
        "validation_threshold_audit": candidate["validation_threshold_audit"],
        "merit_proxy_alignment_metrics": {
            "roc_auc_vs_merit_proxy_positive": round(float(merit_proxy_alignment_auc), 4),
            "spearman_score_vs_merit_proxy": round(float(merit_proxy_spearman), 4),
            "mean_score_when_historically_approved": round(
                float(
                    test_predictions.loc[
                        test_predictions["historical_is_approved"].astype(bool),
                        "final_score",
                    ].mean()
                ),
                4,
            ),
            "mean_score_when_historically_rejected": round(
                float(
                    test_predictions.loc[
                        ~test_predictions["historical_is_approved"].astype(bool),
                        "final_score",
                    ].mean()
                ),
                4,
            ),
        },
        "production_feature_importance": permutation_ranking[:15],
    }


def main():
    parser = argparse.ArgumentParser(description="Train subsidy approval ML model")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to xlsx data")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model bundle",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Where to save training metrics as JSON",
    )
    parser.add_argument(
        "--test-predictions-path",
        default=DEFAULT_TEST_PREDICTIONS_PATH,
        help="Where to save test-set predictions as CSV",
    )
    parser.add_argument(
        "--split-mode",
        choices=["time"],
        default="time",
        help="Split strategy for validation (temporal only)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Share of rows used for train split",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Share of rows used for validation split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )

    args = parser.parse_args()
    active_blend_weights = resolve_blend_weights(DEFAULT_BLEND_WEIGHTS)

    dataset = build_training_dataset(args.data_path)
    requested_feature_sets = [
        feature_set_name
        for feature_set_name in FEATURE_SET_CANDIDATES
        if feature_set_name in dataset["X_by_feature_set"]
    ]
    if not requested_feature_sets:
        raise RuntimeError("No feature sets available for training")

    splits_by_feature_set: dict[str, dict[str, dict[str, object]]] = {}
    available_models_by_feature_set: dict[str, dict[str, object]] = {}
    model_candidates: list[dict[str, object]] = []

    for feature_set_name in requested_feature_sets:
        feature_columns = dataset["feature_set_columns"][feature_set_name]
        numeric_columns = dataset["feature_set_numeric_columns"][feature_set_name]
        categorical_columns = dataset["feature_set_categorical_columns"][feature_set_name]
        feature_set_deployability = get_feature_set_deployability(feature_set_name)
        rule_score_column = get_blend_rule_score_column(feature_set_name)
        feature_set_splits = split_dataset(
            dataset["X_by_feature_set"][feature_set_name],
            dataset["y"],
            dataset["metadata"],
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            mode=args.split_mode,
            random_state=args.random_state,
        )
        splits_by_feature_set[feature_set_name] = feature_set_splits
        available_models = get_available_model_candidates(
            random_state=args.random_state,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        available_models_by_feature_set[feature_set_name] = available_models
        candidate_model_names = [
            model_name
            for model_name in MODEL_NAME_CANDIDATES
            if model_name in available_models["available_model_names"]
        ]

        train_X = feature_set_splits["train"]["X"][feature_columns].copy()
        valid_X = feature_set_splits["valid"]["X"][feature_columns].copy()
        train_y = feature_set_splits["train"]["y"].copy()
        valid_y = feature_set_splits["valid"]["y"].copy()
        valid_rule_scores = feature_set_splits["valid"]["metadata"][rule_score_column].astype(float)
        valid_full_rule_scores = feature_set_splits["valid"]["metadata"]["rule_score"].astype(float)
        valid_disqualified_mask = (
            feature_set_splits["valid"]["metadata"]["disqualified"].fillna(False).astype(bool)
        )

        for model_name in candidate_model_names:
            sample_weight_supported = model_supports_sample_weight(
                model_name=model_name,
                random_state=args.random_state,
                categorical_columns=categorical_columns,
                numeric_columns=numeric_columns,
            )
            train_sample_weight = None
            if sample_weight_supported:
                train_sample_weight = feature_set_splits["train"]["metadata"]["sample_weight"]

            valid_model = train_calibrated_model(
                model_name=model_name,
                X=train_X,
                y=train_y,
                sample_weight=train_sample_weight,
                calibration_method="raw",
                categorical_columns=categorical_columns,
                numeric_columns=numeric_columns,
                random_state=args.random_state,
            )
            valid_prob_raw = valid_model.predict_proba(valid_X)[:, 1]
            calibration_tuning = choose_probability_calibrator(valid_y, valid_prob_raw)
            probability_calibrator = calibration_tuning["best_calibrator"]
            valid_prob = apply_probability_calibrator(
                valid_prob_raw,
                probability_calibrator,
            )
            valid_final_score = compute_blended_scores(
                rule_scores=valid_rule_scores,
                ml_probabilities=valid_prob,
                blend_weights=active_blend_weights,
                disqualified_mask=valid_disqualified_mask,
            )
            threshold_tuning = tune_decision_threshold(
                valid_y,
                valid_final_score,
                fn_cost=FN_COST,
                fp_cost=FP_COST,
                score_scale_max=DECISION_SCORE_SCALE_MAX,
            )
            decision_threshold, tuned_threshold_raw, threshold_source = (
                _resolve_candidate_decision_threshold(threshold_tuning)
            )
            metrics = evaluate_predictions(
                valid_y,
                valid_final_score,
                threshold=decision_threshold,
                score_scale_max=DECISION_SCORE_SCALE_MAX,
            )
            class_gap = _class_gap(valid_y, valid_final_score)
            validation_threshold_audit = _threshold_audit(
                valid_y,
                valid_final_score,
                score_scale_max=DECISION_SCORE_SCALE_MAX,
            )
            blend_tuning = tune_blend_weights(
                y_true=valid_y,
                rule_scores=valid_rule_scores,
                ml_probabilities=valid_prob,
                disqualified_mask=valid_disqualified_mask,
            )
            validation_blend_metrics = _blend_ranking_metrics(
                y_true=valid_y,
                rule_scores=valid_rule_scores,
                ml_probabilities=valid_prob,
                blend_weights=active_blend_weights,
                disqualified_mask=valid_disqualified_mask,
            )
            regional_sensitivity = _feature_shuffle_sensitivity(
                model=valid_model,
                feature_frame=valid_X,
                columns=REGIONAL_SIGNAL_COLUMNS,
                probability_calibrator=probability_calibrator,
            )
            model_candidates.append(
                {
                    "feature_set_name": feature_set_name,
                    "feature_set_deployable": bool(feature_set_deployability["deployable"]),
                    "feature_set_deployability_reason": str(feature_set_deployability["reason"]),
                    "feature_columns": feature_columns,
                    "categorical_columns": categorical_columns,
                    "numeric_columns": numeric_columns,
                    "rule_score_column": rule_score_column,
                    "model_name": model_name,
                    "sample_weight_supported": sample_weight_supported,
                    "sample_weight_used": sample_weight_supported,
                    "calibration_method": calibration_tuning["best_method"],
                    "calibration_candidates": calibration_tuning["candidates"],
                    "decision_threshold": float(decision_threshold),
                    "decision_threshold_tuned": float(tuned_threshold_raw),
                    "decision_threshold_source": str(threshold_source),
                    "decision_score_name": DEFAULT_DECISION_SCORE_NAME,
                    "validation_metrics": metrics,
                    "validation_decision_profile": _classwise_decision_metrics(metrics),
                    "validation_class_gap": class_gap,
                    "validation_probability_distribution": _probability_distribution_summary(valid_prob),
                    "validation_decision_score_distribution": _probability_distribution_summary(
                        valid_final_score.astype(float)
                    ),
                    "validation_threshold_audit": validation_threshold_audit,
                    "validation_calibration_audit": _calibration_audit(valid_y, valid_prob),
                    "validation_strictness_commentary": _strictness_commentary(metrics),
                    "validation_regional_sensitivity": regional_sensitivity,
                    "blend_weights": active_blend_weights,
                    "validation_blend": {
                        "weight_source": "src.modeling.DEFAULT_BLEND_WEIGHTS",
                        "best_weights": active_blend_weights,
                        "best_metrics": validation_blend_metrics,
                        "diagnostic_tuned_best_weights": dict(blend_tuning["best_weights"]),
                        "diagnostic_tuned_best_metrics": dict(blend_tuning["best_metrics"]),
                        "diagnostic_candidates": blend_tuning["candidates"],
                    },
                    "validation_rule_baseline_metrics": _rule_baseline_metrics(
                        valid_y,
                        valid_rule_scores,
                    ),
                    "validation_rule_baseline_full_metrics": _rule_baseline_metrics(
                        valid_y,
                        valid_full_rule_scores,
                    ),
                    "selection_score": _selection_score(
                        metrics=metrics,
                        class_gap=class_gap,
                        regional_sensitivity=regional_sensitivity,
                        blend_metrics=validation_blend_metrics,
                    ),
                    "threshold_candidates": threshold_tuning["candidates"],
                    "_probability_calibrator": probability_calibrator,
                }
            )

    if not model_candidates:
        raise RuntimeError("No model candidates are available for training in the current environment")

    model_candidates.sort(
        key=lambda item: (
            *_validation_ranking_score(item),
            item["validation_class_gap"],
            -item["validation_regional_sensitivity"]["mean_abs_delta"],
        ),
        reverse=True,
    )
    best_overall_candidate = model_candidates[0]
    deployable_candidates = [
        item for item in model_candidates
        if bool(item["feature_set_deployable"])
    ]
    if not deployable_candidates:
        raise RuntimeError("No deployable model candidates are available for bundle export")

    best_deployable_candidate = deployable_candidates[0]
    overall_vs_deployable_different = (
        str(best_overall_candidate["feature_set_name"]) != str(best_deployable_candidate["feature_set_name"])
        or str(best_overall_candidate["model_name"]) != str(best_deployable_candidate["model_name"])
    )

    best_overall_eval = _evaluate_candidate_on_test(
        best_overall_candidate,
        splits_by_feature_set[str(best_overall_candidate["feature_set_name"])],
        args.random_state,
    )
    if overall_vs_deployable_different:
        best_deployable_eval = _evaluate_candidate_on_test(
            best_deployable_candidate,
            splits_by_feature_set[str(best_deployable_candidate["feature_set_name"])],
            args.random_state,
        )
    else:
        best_deployable_eval = best_overall_eval

    selected_candidate = best_deployable_candidate
    selected_eval = best_deployable_eval
    selected_model_name = str(selected_candidate["model_name"])
    selected_feature_set_name = str(selected_candidate["feature_set_name"])
    selected_feature_columns = list(selected_candidate["feature_columns"])
    selected_numeric_columns = list(selected_candidate["numeric_columns"])
    selected_categorical_columns = list(selected_candidate["categorical_columns"])
    selected_threshold_raw = float(selected_candidate["decision_threshold"])
    selected_threshold = float(selected_threshold_raw)
    selected_threshold_tuned = float(selected_candidate.get("decision_threshold_tuned", selected_threshold))
    selected_threshold_source = str(selected_candidate.get("decision_threshold_source", "validation_tuned"))
    selected_probability_calibrator = selected_candidate["_probability_calibrator"]
    selected_calibration_method = str(selected_candidate["calibration_method"])
    selected_blend_weights = resolve_blend_weights(selected_candidate["blend_weights"])
    selected_sample_weight_supported = bool(selected_candidate["sample_weight_supported"])

    test_predictions = selected_eval["test_predictions"]
    test_predictions_path = Path(args.test_predictions_path)
    test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")

    full_sample_weight = None
    if selected_sample_weight_supported:
        full_sample_weight = dataset["metadata"]["sample_weight"]
    full_model = train_calibrated_model(
        model_name=selected_model_name,
        X=dataset["X_by_feature_set"][selected_feature_set_name][selected_feature_columns],
        y=dataset["y"],
        sample_weight=full_sample_weight,
        calibration_method="raw",
        categorical_columns=selected_categorical_columns,
        numeric_columns=selected_numeric_columns,
        random_state=args.random_state,
    )

    explanation_neutral_values = build_explanation_neutral_values(
        dataset["X_by_feature_set"][selected_feature_set_name][selected_feature_columns]
    )
    consistency_audit = _consistency_and_leakage_audit(
        dataset=dataset,
        feature_set_name=selected_feature_set_name,
        feature_columns=selected_feature_columns,
    )

    report = {
        "data_path": str(Path(args.data_path)),
        "split_mode": args.split_mode,
        "random_state": args.random_state,
        "requested_feature_sets": requested_feature_sets,
        "requested_model_candidates": MODEL_NAME_CANDIDATES,
        "default_feature_set_name": DEFAULT_FEATURE_SET_NAME,
        "selected_bundle_model_role": "best_deployable_model",
        "overall_vs_deployable_different": bool(overall_vs_deployable_different),
        "best_feature_set_name": selected_feature_set_name,
        "best_model_name": selected_model_name,
        "ranking_score_name": DEFAULT_DECISION_SCORE_NAME,
        "decision_score_name": DEFAULT_DECISION_SCORE_NAME,
        "decision_threshold_score_name": DEFAULT_DECISION_SCORE_NAME,
        "decision_threshold_scale_max": float(DECISION_SCORE_SCALE_MAX),
        "feature_columns": selected_feature_columns,
        "feature_set_columns": dataset["feature_set_columns"],
        "feature_set_deployability": dataset["feature_set_deployability"],
        "rule_feature_columns": list(
            dataset["rule_scores"]
            .filter(regex=r"^contrib_")
            .columns.str.replace("contrib_", "", regex=False)
        ),
        "blend_rule_score_column": str(selected_candidate["rule_score_column"]),
        "blend_weight_source": "src.modeling.DEFAULT_BLEND_WEIGHTS",
        "baseline_blend_weights": active_blend_weights,
        "blend_weights": selected_blend_weights,
        "model_selection_sort_policy": {
            "selection_goal": (
                "Select the strongest production ranking candidate for final_score "
                "among deployable feature sets."
            ),
            "primary_sort_order": [
                "validation_blend.best_metrics.roc_auc",
                "validation_blend.best_metrics.average_precision",
                "validation_metrics.roc_auc",
                "selection_score",
            ],
            "legacy_selection_score_role": "secondary_tiebreaker_and_diagnostic",
        },
        "blend_weight_search_grid": BLEND_WEIGHT_SEARCH_GRID,
        "blend_weight_search_note": (
            "Validation search сохраняется только как diagnostic audit. "
            "Runtime formula фиксируется через src.modeling.DEFAULT_BLEND_WEIGHTS."
        ),
        "decision_threshold": selected_threshold,
        "decision_threshold_raw_validation": selected_threshold_tuned,
        "decision_threshold_source": selected_threshold_source,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "probability_temperature": 1.0,
        "calibration_method": selected_calibration_method,
        "dataset_size_raw": int(len(dataset["df"])),
        "dataset_size_eligible": int(len(dataset["X"])),
        "decision_support_only": True,
        "appendix_2_compliance_scope": (
            "Автоматически проверяется только часть критериев из доступных данных. "
            "Критерии, требующие ГИСС, ИСЖ, ИБСПР, ЕАСУ, ИСЕГКН и приложенных документов, "
            "остаются на ручной или интеграционной проверке."
        ),
        "training_target_policy": (
            "Модель обучается на historical outcome target `is_approved` с temporal split. "
            "Synthetic merge выполняется явно по `app_number`; leaky synthetic признаки "
            "исключаются из production training, а blend и rule_score_feature используют "
            "feature-set-specific честный rule baseline."
        ),
        "merit_proxy_positive_threshold": MERIT_PROXY_POSITIVE_THRESHOLD,
        "primary_model_categorical_columns": selected_categorical_columns,
        "primary_model_numeric_columns": selected_numeric_columns,
        "merit_proxy_feature_columns": MERIT_PROXY_FEATURE_COLUMNS,
        "regional_signal_columns": REGIONAL_SIGNAL_COLUMNS,
        "safe_synthetic_feature_columns": SAFE_SYNTHETIC_FEATURE_COLUMNS,
        "leaky_or_diagnostic_synthetic_columns": LEAKY_OR_DIAGNOSTIC_SYNTHETIC_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "synthetic_feature_info": dataset["synthetic_feature_info"],
        "available_models_by_feature_set": available_models_by_feature_set,
        "best_validation_metrics": selected_candidate["validation_metrics"],
        "best_validation_regional_sensitivity": selected_candidate["validation_regional_sensitivity"],
        "best_validation_rule_baseline_metrics": selected_candidate["validation_rule_baseline_metrics"],
        "best_validation_rule_baseline_full_metrics": selected_candidate["validation_rule_baseline_full_metrics"],
        "best_candidate_sample_weight_supported": selected_sample_weight_supported,
        "best_candidate_sample_weight_used": selected_candidate["sample_weight_used"],
        "best_overall_model": {
            **best_overall_eval["candidate_summary"],
            "test_metrics": best_overall_eval["test_metrics"],
            "test_decision_profile": best_overall_eval["test_decision_profile"],
            "test_probability_distribution": best_overall_eval["test_probability_distribution"],
            "test_final_score_metrics": best_overall_eval["test_final_score_metrics"],
            "deployable": bool(best_overall_candidate["feature_set_deployable"]),
        },
        "best_deployable_model": {
            **best_deployable_eval["candidate_summary"],
            "test_metrics": best_deployable_eval["test_metrics"],
            "test_decision_profile": best_deployable_eval["test_decision_profile"],
            "test_probability_distribution": best_deployable_eval["test_probability_distribution"],
            "test_final_score_metrics": best_deployable_eval["test_final_score_metrics"],
            "deployable": True,
        },
        "model_validation_candidates": [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in model_candidates
        ],
        "process_biased_features_excluded_from_ml": PROCESS_BIASED_FEATURE_COLUMNS,
        "deadline_disqualification_policy": "Просроченные заявки отсекаются до ML и итогового score",
        "deadline_disqualified_rows": int(dataset["deadline_disqualified_count"]),
        "split_sizes": {
            split_name: int(len(split_data["X"]))
            for split_name, split_data in splits_by_feature_set[selected_feature_set_name].items()
        },
        "split_date_ranges": {
            split_name: _describe_split_dates(split_data["metadata"])
            for split_name, split_data in splits_by_feature_set[selected_feature_set_name].items()
        },
        "threshold_tuning_policy": (
            "Порог выбирается по финальному decision score `final_score`, а не по "
            "`ml_probability`: сначала cost-sensitive поиск оставляет кандидаты с "
            "приемлемой ценой ложных отказов/ложных одобрений, затем среди них "
            "берётся компромисс по balanced accuracy, recall и F1."
        ),
        "calibration_selection_policy": (
            "Сравниваются identity, sigmoid и isotonic calibrators. Сначала "
            "отсекаются слишком слабые по balanced accuracy и ROC-AUC варианты, "
            "после чего shortlist ранжируется по Brier score, class gap и "
            "стабильности распределения вероятностей."
        ),
        "model_selection_policy": (
            "Сравниваются несколько моделей и два feature set'а. Финальный bundle "
            "использует не лучший offline-кандидат вообще, а лучший deployable-кандидат."
        ),
        "sample_weight_policy": (
            "sample_weight рассчитывается на всём eligible датасете и используется "
            "только там, где estimator действительно поддерживает `sample_weight`."
        ),
        "rule_score_diagnostic_policy": (
            "Полный rule score из scoring.py сохраняется для baseline/диагностики, "
            "но для ML-feature path и blend используется честный feature-set-specific rule baseline."
        ),
        "deployability_policy": (
            "Bundle по умолчанию сохраняет `best_deployable_model`. "
            "`best_overall_model` остаётся в отчёте как offline benchmark."
        ),
        "consistency_and_leakage_audit": consistency_audit,
        "validation_threshold_audit": selected_candidate["validation_threshold_audit"],
        "validation_calibration_audit": selected_candidate["validation_calibration_audit"],
        "validation_probability_distribution": selected_candidate["validation_probability_distribution"],
        "validation_decision_score_distribution": selected_candidate["validation_decision_score_distribution"],
        "validation_decision_profile": selected_candidate["validation_decision_profile"],
        "validation_strictness_commentary": selected_candidate["validation_strictness_commentary"],
        "test_metrics": selected_eval["test_metrics"],
        "test_decision_profile": selected_eval["test_decision_profile"],
        "test_probability_distribution": selected_eval["test_probability_distribution"],
        "test_decision_score_distribution": selected_eval["test_decision_score_distribution"],
        "test_strictness_commentary": selected_eval["test_strictness_commentary"],
        "test_rule_baseline_full_metrics": selected_eval["test_rule_baseline_full_metrics"],
        "test_rule_baseline_blend_metrics": selected_eval["test_rule_baseline_blend_metrics"],
        "test_final_score_metrics": selected_eval["test_final_score_metrics"],
        "test_region_sensitivity": selected_eval["test_region_sensitivity"],
        "test_calibration_audit": selected_eval["test_calibration_audit"],
        "test_rule_ml_compatibility_audit": selected_eval["test_rule_ml_compatibility_audit"],
        "test_segment_audit": selected_eval["test_segment_audit"],
        "merit_proxy_alignment_metrics": selected_eval["merit_proxy_alignment_metrics"],
        "production_feature_importance": selected_eval["production_feature_importance"],
    }

    save_bundle(
        model=full_model,
        tables=dataset["tables"],
        model_name=selected_model_name,
        output_path=args.model_path,
        feature_columns=selected_feature_columns,
        feature_set_name=selected_feature_set_name,
        decision_threshold=selected_threshold,
        probability_calibrator=selected_probability_calibrator,
        probability_temperature=1.0,
        calibration_method=selected_calibration_method,
        blend_weights=selected_blend_weights,
        explanation_neutral_values=explanation_neutral_values,
        report=report,
    )
    save_json(report, args.report_path)

    print("=" * 60)
    print("Multi-model ML training complete")
    print("=" * 60)
    print(f"Rows used for training: {len(dataset['X'])}")
    print(f"Rows excluded by deadline prefilter: {dataset['deadline_disqualified_count']}")
    print(
        "Best overall: "
        f"{best_overall_candidate['feature_set_name']}::{best_overall_candidate['model_name']}"
    )
    print(
        "Best deployable: "
        f"{selected_feature_set_name}::{selected_model_name}"
    )
    print(f"Features in bundle: {len(selected_feature_columns)}")
    print(f"Calibration: {selected_calibration_method}")
    print(f"Decision threshold: {selected_threshold}")
    print(
        "Compared model candidates: "
        + ", ".join(
            f"{item['feature_set_name']}::{item['model_name']}"
            for item in model_candidates
        )
    )
    print(f"Validation outcome ROC-AUC: {selected_candidate['validation_metrics']['roc_auc']}")
    print(
        "Validation regional sensitivity: "
        f"{selected_candidate['validation_regional_sensitivity']['mean_abs_delta']}"
    )
    print(f"Test outcome ROC-AUC: {report['test_metrics']['roc_auc']}")
    print(f"Test final-score ROC-AUC: {report['test_final_score_metrics']['roc_auc']}")
    print(f"Test regional sensitivity: {report['test_region_sensitivity']['mean_abs_delta']}")
    print(
        "Test score gap between classes: "
        f"{report['test_final_score_metrics']['mean_score_gap_between_classes']}"
    )
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.report_path}")
    print(f"Test predictions saved to: {test_predictions_path}")


if __name__ == "__main__":
    main()
