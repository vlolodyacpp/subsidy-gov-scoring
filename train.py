import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling import (
    FEATURE_COLUMNS,
    MERIT_PROXY_FEATURE_COLUMNS,
    MERIT_PROXY_POSITIVE_THRESHOLD,
    PRIMARY_MODEL_CATEGORICAL_COLUMNS,
    PRIMARY_MODEL_NUMERIC_COLUMNS,
    PROCESS_BIASED_FEATURE_COLUMNS,
    apply_probability_calibrator,
    build_explanation_neutral_values,
    build_training_dataset,
    choose_probability_calibrator,
    compute_permutation_feature_importance,
    evaluate_predictions,
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
]
FN_COST = 3.0
FP_COST = 1.0
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
        choices=["time", "random"],
        default="time",
        help="Split strategy for validation",
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

    dataset = build_training_dataset(args.data_path)
    splits = split_dataset(
        dataset["X"],
        dataset["y"],
        dataset["metadata"],
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        mode=args.split_mode,
        random_state=args.random_state,
    )

    train_X = splits["train"]["X"][FEATURE_COLUMNS].copy()
    valid_X = splits["valid"]["X"][FEATURE_COLUMNS].copy()
    test_X = splits["test"]["X"][FEATURE_COLUMNS].copy()
    train_y = splits["train"]["y"].copy()
    valid_y = splits["valid"]["y"].copy()
    test_y = splits["test"]["y"].copy()
    valid_rule_scores = splits["valid"]["metadata"]["rule_score"].astype(float)
    valid_disqualified_mask = (
        splits["valid"]["metadata"]["disqualified"].fillna(False).astype(bool)
    )

    model_candidates: list[dict[str, object]] = []
    for model_name in MODEL_NAME_CANDIDATES:
        valid_model = train_calibrated_model(
            model_name=model_name,
            X=train_X,
            y=train_y,
            sample_weight=None,
            calibration_method="raw",
            categorical_columns=PRIMARY_MODEL_CATEGORICAL_COLUMNS,
            numeric_columns=PRIMARY_MODEL_NUMERIC_COLUMNS,
            random_state=args.random_state,
        )
        valid_prob_raw = valid_model.predict_proba(valid_X)[:, 1]
        calibration_tuning = choose_probability_calibrator(valid_y, valid_prob_raw)
        probability_calibrator = calibration_tuning["best_calibrator"]
        valid_prob = apply_probability_calibrator(
            valid_prob_raw,
            probability_calibrator,
        )
        threshold_tuning = tune_decision_threshold(
            valid_y,
            valid_prob,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        metrics = threshold_tuning["best_metrics"]
        class_gap = _class_gap(valid_y, valid_prob)
        blend_tuning = tune_blend_weights(
            y_true=valid_y,
            rule_scores=valid_rule_scores,
            ml_probabilities=valid_prob,
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
                "model_name": model_name,
                "calibration_method": calibration_tuning["best_method"],
                "calibration_candidates": calibration_tuning["candidates"],
                "decision_threshold": float(threshold_tuning["best_threshold"]),
                "validation_metrics": metrics,
                "validation_class_gap": class_gap,
                "validation_regional_sensitivity": regional_sensitivity,
                "validation_blend": blend_tuning,
                "selection_score": _selection_score(
                    metrics=metrics,
                    class_gap=class_gap,
                    regional_sensitivity=regional_sensitivity,
                    blend_metrics=blend_tuning["best_metrics"],
                ),
                "threshold_candidates": threshold_tuning["candidates"],
                "_probability_calibrator": probability_calibrator,
            }
        )

    model_candidates.sort(
        key=lambda item: (
            item["selection_score"],
            item["validation_metrics"]["roc_auc"] or -1,
            item["validation_class_gap"],
            -item["validation_regional_sensitivity"]["mean_abs_delta"],
        ),
        reverse=True,
    )
    best_candidate = model_candidates[0]
    best_model_name = str(best_candidate["model_name"])
    best_threshold_raw = float(best_candidate["decision_threshold"])
    best_threshold = float(best_threshold_raw)
    best_probability_calibrator = best_candidate["_probability_calibrator"]
    best_calibration_method = str(best_candidate["calibration_method"])
    best_blend_weights = dict(best_candidate["validation_blend"]["best_weights"])

    train_valid_X = pd.concat([train_X, valid_X], axis=0)
    train_valid_y = pd.concat([train_y, valid_y], axis=0)
    test_model = train_calibrated_model(
        model_name=best_model_name,
        X=train_valid_X,
        y=train_valid_y,
        sample_weight=None,
        calibration_method="raw",
        categorical_columns=PRIMARY_MODEL_CATEGORICAL_COLUMNS,
        numeric_columns=PRIMARY_MODEL_NUMERIC_COLUMNS,
        random_state=args.random_state,
    )

    test_prob_raw = test_model.predict_proba(test_X)[:, 1]
    test_prob = apply_probability_calibrator(
        test_prob_raw,
        best_probability_calibrator,
    )
    test_predictions = splits["test"]["metadata"].copy()
    test_predictions["ml_probability"] = test_prob.round(4)
    test_predictions["ml_score"] = (test_prob * 100).round(1)
    test_predictions["ml_decision_threshold"] = best_threshold
    test_predictions["ml_predicted_positive"] = (
        test_predictions["ml_probability"] >= best_threshold
    )
    test_predictions["historical_is_approved"] = (
        test_predictions["historical_is_approved"].astype(int)
    )
    test_predictions["merit_proxy_positive"] = (
        test_predictions["merit_proxy_positive"].astype(int)
    )
    test_predictions["training_target"] = test_predictions["training_target"].astype(int)
    test_predictions["final_score"] = np.round(
        test_predictions["rule_score"].astype(float) * best_blend_weights["rule_score"]
        + test_predictions["ml_score"].astype(float) * best_blend_weights["ml_score"],
        1,
    )
    test_predictions.loc[
        test_predictions["disqualified"].fillna(False).astype(bool), "final_score"
    ] = 0.0
    test_predictions.loc[
        test_predictions["disqualified"].fillna(False).astype(bool), "ml_predicted_positive"
    ] = False
    test_predictions = test_predictions.sort_values("final_score", ascending=False)

    test_predictions_path = Path(args.test_predictions_path)
    test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")

    full_model = train_calibrated_model(
        model_name=best_model_name,
        X=dataset["X"][FEATURE_COLUMNS],
        y=dataset["y"],
        sample_weight=None,
        calibration_method="raw",
        categorical_columns=PRIMARY_MODEL_CATEGORICAL_COLUMNS,
        numeric_columns=PRIMARY_MODEL_NUMERIC_COLUMNS,
        random_state=args.random_state,
    )

    permutation_ranking = compute_permutation_feature_importance(
        model=test_model,
        X=test_X,
        y=test_y,
        random_state=args.random_state,
    )

    test_feature_sensitivity = _feature_shuffle_sensitivity(
        model=test_model,
        feature_frame=test_X,
        columns=REGIONAL_SIGNAL_COLUMNS,
        probability_calibrator=best_probability_calibrator,
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

    explanation_neutral_values = build_explanation_neutral_values(dataset["X"][FEATURE_COLUMNS])

    report = {
        "data_path": str(Path(args.data_path)),
        "split_mode": args.split_mode,
        "random_state": args.random_state,
        "feature_columns": FEATURE_COLUMNS,
        "rule_feature_columns": list(
            dataset["rule_scores"]
            .filter(regex=r"^contrib_")
            .columns.str.replace("contrib_", "", regex=False)
        ),
        "blend_weights": best_blend_weights,
        "decision_threshold": best_threshold,
        "decision_threshold_raw_validation": best_threshold_raw,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "probability_temperature": 1.0,
        "calibration_method": best_calibration_method,
        "dataset_size_raw": int(len(dataset["df"])),
        "dataset_size_eligible": int(len(dataset["X"])),
        "decision_support_only": True,
        "appendix_2_compliance_scope": (
            "Автоматически проверяется только часть критериев из доступных данных. "
            "Критерии, требующие ГИСС, ИСЖ, ИБСПР, ЕАСУ, ИСЕГКН и приложенных документов, "
            "остаются на ручной или интеграционной проверке."
        ),
        "training_target_policy": (
            "ML-пайплайн использует single-model scoring без raw region/direction в "
            "категориальных признаках. Итоговая модель строится как neural-network "
            "скоринг по набору signals заявки, который включает нормативные признаки, "
            "сжатые региональные priors, history counts, process-context и "
            "ограниченный набор rule-derived signals; итоговый score строится как "
            "обязательный blend rule_score + neural_score."
        ),
        "merit_proxy_positive_threshold": MERIT_PROXY_POSITIVE_THRESHOLD,
        "primary_model_categorical_columns": PRIMARY_MODEL_CATEGORICAL_COLUMNS,
        "primary_model_numeric_columns": PRIMARY_MODEL_NUMERIC_COLUMNS,
        "merit_proxy_feature_columns": MERIT_PROXY_FEATURE_COLUMNS,
        "regional_signal_columns": REGIONAL_SIGNAL_COLUMNS,
        "best_validation_metrics": best_candidate["validation_metrics"],
        "best_validation_regional_sensitivity": best_candidate["validation_regional_sensitivity"],
        "model_validation_candidates": [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in model_candidates
        ],
        "process_biased_features_excluded_from_ml": PROCESS_BIASED_FEATURE_COLUMNS,
        "deadline_disqualification_policy": "Просроченные заявки отсекаются до ML и итогового score",
        "deadline_disqualified_rows": int(dataset["deadline_disqualified_count"]),
        "split_sizes": {
            split_name: int(len(split_data["X"]))
            for split_name, split_data in splits.items()
        },
        "split_date_ranges": {
            split_name: _describe_split_dates(split_data["metadata"])
            for split_name, split_data in splits.items()
        },
        "best_model_name": best_model_name,
        "threshold_tuning_policy": (
            "Порог выбирается cost-sensitive поиском: цена ложного отказа выше "
            "цены ложного одобрения."
        ),
        "calibration_selection_policy": (
            "Сравниваются identity, sigmoid и isotonic calibrators. "
            "Лучший кандидат выбирается по Brier score и ROC-AUC."
        ),
        "model_selection_policy": (
            "Используется neural-network модель как primary ML-слой; итоговый score "
            "всегда сочетает rule_score и neural_score через validation-tuned blend "
            "с ненулевой долей rule-based компоненты."
        ),
        "test_metrics": {},
        "test_final_score_metrics": {},
        "test_region_sensitivity": test_feature_sensitivity,
        "merit_proxy_alignment_metrics": {},
    }

    report["test_metrics"] = evaluate_predictions(
        test_predictions["training_target"].astype(int),
        test_predictions["ml_probability"].astype(float),
        threshold=best_threshold,
    )
    report["test_final_score_metrics"] = {
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
    }
    report["merit_proxy_alignment_metrics"] = {
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
    }
    report["production_feature_importance"] = permutation_ranking[:15]

    save_bundle(
        model=full_model,
        tables=dataset["tables"],
        model_name=best_model_name,
        output_path=args.model_path,
        decision_threshold=best_threshold,
        probability_calibrator=best_probability_calibrator,
        probability_temperature=1.0,
        calibration_method=best_calibration_method,
        blend_weights=best_blend_weights,
        explanation_neutral_values=explanation_neutral_values,
        report=report,
    )
    save_json(report, args.report_path)

    print("=" * 60)
    print("Single-model ML training complete")
    print("=" * 60)
    print(f"Rows used for training: {len(dataset['X'])}")
    print(f"Rows excluded by deadline prefilter: {dataset['deadline_disqualified_count']}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Best model: {best_model_name}")
    print(f"Calibration: {best_calibration_method}")
    print(f"Decision threshold: {best_threshold}")
    print(
        "Compared model candidates: "
        + ", ".join(item["model_name"] for item in model_candidates)
    )
    print(f"Validation outcome ROC-AUC: {best_candidate['validation_metrics']['roc_auc']}")
    print(
        "Validation regional sensitivity: "
        f"{best_candidate['validation_regional_sensitivity']['mean_abs_delta']}"
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
