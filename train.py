import argparse
from pathlib import Path

import pandas as pd

from src.modeling import (
    CORE_NEUTRAL_NUMERIC_VALUES,
    DEFAULT_BLEND_WEIGHTS,
    DualBranchEnsembleModel,
    FEATURE_COLUMNS,
    MERIT_PROXY_POSITIVE_THRESHOLD,
    MERIT_PROXY_FEATURE_COLUMNS,
    PRIMARY_MODEL_CATEGORICAL_COLUMNS,
    PRIMARY_MODEL_NUMERIC_COLUMNS,
    PROCESS_BIASED_FEATURE_COLUMNS,
    apply_probability_temperature,
    build_training_dataset,
    choose_probability_temperature,
    compute_permutation_feature_importance,
    evaluate_predictions,
    save_bundle,
    save_json,
    split_dataset,
    tune_decision_threshold,
    train_calibrated_model,
)


DEFAULT_DATA_PATH = "data/subsidies.xlsx"
DEFAULT_MODEL_PATH = "models/artifacts/subsidy_model.joblib"
DEFAULT_REPORT_PATH = "models/reports/training_metrics.json"
DEFAULT_TEST_PREDICTIONS_PATH = "models/reports/test_predictions.csv"
CONTEXT_BRANCH_MODEL_NAME = "random_forest"
CORE_BRANCH_MODEL_NAME = "logistic_regression"
CONTEXT_WEIGHT_CANDIDATES = [0.55, 0.6, 0.65, 0.7]
THRESHOLD_SOFTENING_DELTA = 0.01


def _build_core_view(X: pd.DataFrame) -> pd.DataFrame:
    core = X.copy()
    if "region" in core.columns:
        core["region"] = ""
    for column_name, neutral_value in CORE_NEUTRAL_NUMERIC_VALUES.items():
        if column_name in core.columns:
            core[column_name] = neutral_value
    return core


def _combine_branch_probabilities(
    context_prob: pd.Series | pd.Index | list | tuple,
    core_prob: pd.Series | pd.Index | list | tuple,
    context_weight: float,
):
    context_arr = pd.Series(context_prob, copy=False).to_numpy(dtype=float)
    core_arr = pd.Series(core_prob, copy=False).to_numpy(dtype=float)
    return context_arr * float(context_weight) + core_arr * (1.0 - float(context_weight))


def _describe_split_dates(metadata: pd.DataFrame) -> dict[str, str | None]:
    dates = pd.to_datetime(metadata["submit_date"], errors="coerce").dropna()
    if dates.empty:
        return {"min_submit_date": None, "max_submit_date": None}
    return {
        "min_submit_date": str(dates.min().date()),
        "max_submit_date": str(dates.max().date()),
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

    train_X = splits["train"]["X"]
    valid_X = splits["valid"]["X"]
    test_X = splits["test"]["X"]
    train_y = splits["train"]["y"]
    valid_y = splits["valid"]["y"]
    test_y = splits["test"]["y"]
    train_sample_weight = splits["train"]["metadata"]["sample_weight"]
    train_valid_sample_weight = pd.concat(
        [
            splits["train"]["metadata"]["sample_weight"],
            splits["valid"]["metadata"]["sample_weight"],
        ],
        axis=0,
    )

    context_valid_model = train_calibrated_model(
        model_name=CONTEXT_BRANCH_MODEL_NAME,
        X=train_X,
        y=train_y,
        sample_weight=train_sample_weight,
        calibration_method="raw",
        random_state=args.random_state,
    )
    core_valid_model = train_calibrated_model(
        model_name=CORE_BRANCH_MODEL_NAME,
        X=_build_core_view(train_X),
        y=train_y,
        sample_weight=train_sample_weight,
        calibration_method="raw",
        random_state=args.random_state,
    )

    context_valid_prob = context_valid_model.predict_proba(valid_X)[:, 1]
    core_valid_prob = core_valid_model.predict_proba(_build_core_view(valid_X))[:, 1]

    ensemble_candidates = []
    for context_weight in CONTEXT_WEIGHT_CANDIDATES:
        valid_prob_raw = _combine_branch_probabilities(
            context_valid_prob,
            core_valid_prob,
            context_weight=context_weight,
        )
        temperature_tuning = choose_probability_temperature(valid_y, valid_prob_raw)
        probability_temperature = temperature_tuning["best_temperature"]
        valid_prob = apply_probability_temperature(
            valid_prob_raw,
            temperature=probability_temperature,
        )
        threshold_tuning = tune_decision_threshold(valid_y, valid_prob)
        metrics = threshold_tuning["best_metrics"]
        selection_score = (
            float(metrics["roc_auc"] or 0.0) * 0.40
            + float(metrics["balanced_accuracy"]) * 0.25
            + float(metrics["recall"]) * 0.25
            + float(metrics["precision"]) * 0.10
            - float(context_weight) * 0.02
        )
        ensemble_candidates.append(
            {
                "context_weight": round(float(context_weight), 2),
                "core_weight": round(float(1.0 - context_weight), 2),
                "probability_temperature": float(probability_temperature),
                "decision_threshold": float(threshold_tuning["best_threshold"]),
                "validation_metrics": metrics,
                "selection_score": round(float(selection_score), 6),
                "threshold_candidates": threshold_tuning["candidates"],
            }
        )

    best_validation_balanced_accuracy = max(
        item["validation_metrics"]["balanced_accuracy"] for item in ensemble_candidates
    )
    best_validation_auc = max(
        float(item["validation_metrics"]["roc_auc"] or 0.0)
        for item in ensemble_candidates
    )
    shortlisted_candidates = [
        item
        for item in ensemble_candidates
        if item["validation_metrics"]["balanced_accuracy"] >= best_validation_balanced_accuracy - 0.025
        and float(item["validation_metrics"]["roc_auc"] or 0.0) >= best_validation_auc - 0.02
    ]
    if not shortlisted_candidates:
        shortlisted_candidates = ensemble_candidates.copy()
    shortlisted_candidates.sort(
        key=lambda item: (
            item["validation_metrics"]["recall"],
            item["validation_metrics"]["balanced_accuracy"],
            item["validation_metrics"]["roc_auc"] or -1,
            -item["context_weight"],
        ),
        reverse=True,
    )
    best_candidate = shortlisted_candidates[0]
    best_context_weight = float(best_candidate["context_weight"])
    best_core_weight = float(best_candidate["core_weight"])
    best_threshold_raw = float(best_candidate["decision_threshold"])
    best_threshold = max(best_threshold_raw - THRESHOLD_SOFTENING_DELTA, 0.05)
    best_probability_temperature = float(best_candidate["probability_temperature"])
    best_model_name = "dual_branch_ensemble_rf_lr"
    best_calibration_method = "raw"
    best_blend_weights = DEFAULT_BLEND_WEIGHTS

    train_valid_X = pd.concat([train_X, valid_X], axis=0)
    train_valid_y = pd.concat([train_y, valid_y], axis=0)
    context_test_model = train_calibrated_model(
        model_name=CONTEXT_BRANCH_MODEL_NAME,
        X=train_valid_X,
        y=train_valid_y,
        sample_weight=train_valid_sample_weight,
        calibration_method="raw",
        random_state=args.random_state,
    )
    core_test_model = train_calibrated_model(
        model_name=CORE_BRANCH_MODEL_NAME,
        X=_build_core_view(train_valid_X),
        y=train_valid_y,
        sample_weight=train_valid_sample_weight,
        calibration_method="raw",
        random_state=args.random_state,
    )
    test_model = DualBranchEnsembleModel(
        context_model=context_test_model,
        core_model=core_test_model,
        context_weight=best_context_weight,
    )

    test_prob_raw = test_model.predict_proba(test_X)[:, 1]
    test_prob = apply_probability_temperature(
        test_prob_raw,
        temperature=best_probability_temperature,
    )
    test_predictions = splits["test"]["metadata"].copy()
    test_predictions["ml_probability"] = test_prob.round(4)
    test_predictions["ml_score"] = (test_prob * 100).round(1)
    test_predictions["ml_decision_threshold"] = best_threshold
    test_predictions["ml_predicted_positive"] = (
        test_predictions["ml_probability"] >= best_threshold
    )
    test_predictions["historical_is_approved"] = test_predictions["historical_is_approved"].astype(int)
    test_predictions["merit_proxy_positive"] = test_predictions["merit_proxy_positive"].astype(int)
    test_predictions["training_target"] = test_predictions["training_target"].astype(int)
    test_predictions["final_score"] = (
        test_predictions["rule_score"] * best_blend_weights["rule_score"]
        + test_predictions["ml_score"] * best_blend_weights["ml_score"]
    ).round(1)
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

    full_context_model = train_calibrated_model(
        model_name=CONTEXT_BRANCH_MODEL_NAME,
        X=dataset["X"],
        y=dataset["y"],
        sample_weight=dataset["metadata"]["sample_weight"],
        calibration_method="raw",
        random_state=args.random_state,
    )
    full_core_model = train_calibrated_model(
        model_name=CORE_BRANCH_MODEL_NAME,
        X=_build_core_view(dataset["X"]),
        y=dataset["y"],
        sample_weight=dataset["metadata"]["sample_weight"],
        calibration_method="raw",
        random_state=args.random_state,
    )
    full_model = DualBranchEnsembleModel(
        context_model=full_context_model,
        core_model=full_core_model,
        context_weight=best_context_weight,
    )

    permutation_ranking = compute_permutation_feature_importance(
        model=test_model,
        X=test_X,
        y=test_y,
        random_state=args.random_state,
    )
    context_branch_ranking = compute_permutation_feature_importance(
        model=context_test_model,
        X=test_X,
        y=test_y,
        random_state=args.random_state,
    )
    core_branch_ranking = compute_permutation_feature_importance(
        model=core_test_model,
        X=_build_core_view(test_X),
        y=test_y,
        random_state=args.random_state,
    )

    from sklearn.metrics import average_precision_score, roc_auc_score

    test_blended_auc = roc_auc_score(
        test_predictions["training_target"].astype(int),
        test_predictions["final_score"].astype(float),
    )
    test_blended_ap = average_precision_score(
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

    report = {
        "data_path": str(Path(args.data_path)),
        "split_mode": args.split_mode,
        "random_state": args.random_state,
        "feature_columns": FEATURE_COLUMNS,
        "rule_feature_columns": list(dataset["rule_scores"].filter(regex=r"^contrib_").columns.str.replace("contrib_", "", regex=False)),
        "blend_weights": best_blend_weights,
        "decision_threshold": best_threshold,
        "decision_threshold_raw_validation": best_threshold_raw,
        "decision_threshold_softening_delta": THRESHOLD_SOFTENING_DELTA,
        "probability_temperature": best_probability_temperature,
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
            "Primary ML использует dual-branch ensemble. Context branch видит region, "
            "direction и subsidy_type; core branch принципиально не видит region и "
            "получает нейтральное значение для region_specialization. "
            "Обе ветки обучаются на historical outcome label без approval-rate признаков, "
            "а merit_proxy_score используется только для sample weighting и alignment-диагностики."
        ),
        "merit_proxy_positive_threshold": MERIT_PROXY_POSITIVE_THRESHOLD,
        "primary_model_categorical_columns": PRIMARY_MODEL_CATEGORICAL_COLUMNS,
        "primary_model_numeric_columns": PRIMARY_MODEL_NUMERIC_COLUMNS,
        "merit_proxy_feature_columns": MERIT_PROXY_FEATURE_COLUMNS,
        "context_branch_model_name": CONTEXT_BRANCH_MODEL_NAME,
        "core_branch_model_name": CORE_BRANCH_MODEL_NAME,
        "ensemble_context_weight": best_context_weight,
        "ensemble_core_weight": best_core_weight,
        "ensemble_validation_candidates": ensemble_candidates,
        "ensemble_validation_shortlist_policy": (
            "Сначала отбираются кандидаты, близкие к лучшим по balanced_accuracy и ROC-AUC, "
            "после чего приоритет отдаётся higher recall и меньшему context_weight."
        ),
        "core_view_neutralized_columns": ["region"],
        "core_view_neutralized_numeric_defaults": CORE_NEUTRAL_NUMERIC_VALUES,
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
        "threshold_tuning_policy": "Сначала выбирается validation threshold с хорошим balanced_accuracy и f1, затем production threshold дополнительно смягчается на фиксированное значение ради снижения FN",
        "blend_tuning_policy": "Final score определяется primary dual-branch ensemble; history approval-rate остаются advisory-слоем и не входят в runtime features primary model",
        "calibration_selection_policy": "Сначала строится raw ensemble probability, затем подбирается temperature scaling для более мягкой и стабильной шкалы score",
        "model_selection_policy": "Выбирается dual-branch ensemble с компромиссом между ROC-AUC, balanced_accuracy, recall и снижением зависимости от region",
        "test_metrics": {},
        "test_blended_metrics": {},
        "merit_proxy_alignment_metrics": {},
    }

    report["test_metrics"] = evaluate_predictions(
        test_predictions["training_target"].astype(int),
        test_predictions["ml_probability"].astype(float),
        threshold=best_threshold,
    )
    report["test_blended_metrics"] = {
        "roc_auc": round(float(test_blended_auc), 4),
        "average_precision": round(float(test_blended_ap), 4),
        "mean_final_score": round(float(test_predictions["final_score"].mean()), 4),
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
    report["context_branch_feature_importance"] = context_branch_ranking[:15]
    report["core_branch_feature_importance"] = core_branch_ranking[:15]

    save_bundle(
        model=full_model,
        tables=dataset["tables"],
        model_name=best_model_name,
        output_path=args.model_path,
        decision_threshold=best_threshold,
        probability_temperature=best_probability_temperature,
        calibration_method=best_calibration_method,
        blend_weights=best_blend_weights,
        report=report,
    )
    save_json(report, args.report_path)

    print("=" * 60)
    print("Merit ML training complete")
    print("=" * 60)
    print(f"Rows used for training: {len(dataset['X'])}")
    print(f"Rows excluded by deadline prefilter: {dataset['deadline_disqualified_count']}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Merit proxy threshold: {MERIT_PROXY_POSITIVE_THRESHOLD}")
    print(f"Best model: {best_model_name}")
    print(
        f"Branch mix: context {CONTEXT_BRANCH_MODEL_NAME} ({best_context_weight:.0%}) + "
        f"core {CORE_BRANCH_MODEL_NAME} ({best_core_weight:.0%})"
    )
    print(f"Calibration: {best_calibration_method}")
    print(f"Probability temperature: {best_probability_temperature}")
    print(f"Decision threshold: {best_threshold} (raw validation {best_threshold_raw})")
    print(f"Blend weights: {best_blend_weights}")
    print(
        "Compared ensemble context weights: "
        + ", ".join(str(item["context_weight"]) for item in ensemble_candidates)
    )
    print(f"Validation outcome ROC-AUC: {best_candidate['validation_metrics']['roc_auc']}")
    print(f"Test outcome ROC-AUC: {report['test_metrics']['roc_auc']}")
    print(f"Test outcome balanced accuracy: {report['test_metrics']['balanced_accuracy']}")
    print(f"Merit proxy alignment ROC-AUC: {report['merit_proxy_alignment_metrics']['roc_auc_vs_merit_proxy_positive']}")
    print(f"Test final-score ROC-AUC: {report['test_blended_metrics']['roc_auc']}")
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.report_path}")
    print(f"Test predictions saved to: {test_predictions_path}")


if __name__ == "__main__":
    main()
