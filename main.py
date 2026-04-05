"""CLI и API для merit scoring с single-model ML и отдельным historical advisory-слоем."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_PATH = "data/subsidies.xlsx"
MODEL_PATH = "models/artifacts/subsidy_model.joblib"


def run_cli(data_path: str, model_path: str = MODEL_PATH):
    """Запуск CLI-пайплайна: rule-based + ML blended скоринг."""
    from src.pipeline import run_pipeline
    from src.features import build_feature_tables, extract_features_batch
    from src.scoring import (
        score_batch, get_score_distribution,
        score_single, WEIGHTS, FACTOR_GROUPS,
    )
    from src.eligibility import evaluate_batch_eligibility
    from src.advisory import build_history_advisory_batch
    from src.modeling import (
        DEFAULT_DECISION_THRESHOLD,
        build_primary_model_frame,
        score_features_with_model,
        load_bundle,
        score_to_risk_label,
    )

    # попытка загрузить ML-модель
    bundle = None
    ml_mode = False
    if model_path and Path(model_path).exists():
        try:
            bundle = load_bundle(model_path)
            ml_mode = True
        except Exception as e:
            print(f"  [!] Не удалось загрузить ML-модель: {e}")

    mode_label = "rule + neural blend" if ml_mode else "rule-only (модель не найдена)"
    print("=" * 60)
    print(f"SUBSIDY SCORING SYSTEM v4.0 — {mode_label}")
    print("=" * 60)

    # 1. загрузка и очистка
    print("\n[1/5] Загрузка данных + замена нормативов...")
    df = run_pipeline(data_path)

    # 2. feature engineering
    print("\n[2/5] Feature engineering...")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    print(f"  Признаков: {features.shape[1]}")
    print(f"  Записей: {features.shape[0]}")

    # 3. rule-based скоринг + eligibility
    print("\n[3/5] Rule-based скоринг + eligibility...")
    rule_scores = score_batch(features)
    eligibility = evaluate_batch_eligibility(df, tables["normative_lookup"])
    rule_scores[eligibility.columns] = eligibility

    rule_stats = get_score_distribution(rule_scores)
    print(f"  Rule средний балл: {rule_stats['mean']}")
    print(f"  Rule медиана: {rule_stats['median']}")
    print(f"  Rule диапазон: {rule_stats['min']} — {rule_stats['max']}")
    print(f"  Риск-распределение (rule): {rule_stats['risk_distribution']}")

    n_disqualified = eligibility["disqualified"].sum()
    print(f"  Дисквалифицировано eligibility: {n_disqualified}")

    # средние значения rule-метрик
    print("\n  Средние значения rule-метрик:")
    for factor in WEIGHTS:
        if factor in features.columns:
            mean_val = features[factor].mean()
            print(f"    {factor}: {mean_val:.3f}")

    # 4. ML blended скоринг (если модель есть)
    if ml_mode:
        print("\n[4/5] ML blended скоринг (neural network)...")
        model = bundle["model"]
        blend_weights = bundle.get("blend_weights", {"rule_score": 0.25, "ml_score": 0.75})
        decision_threshold = bundle.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
        probability_calibrator = bundle.get("probability_calibrator")
        probability_temperature = bundle.get("probability_temperature", 1.0)

        model_input = build_primary_model_frame(
            raw_input=df,
            extracted_features=features,
            rule_scores=rule_scores,
        )
        blended = score_features_with_model(
            model_input,
            model=model,
            rule_scores=rule_scores,
            blend_weights=blend_weights,
            decision_threshold=decision_threshold,
            probability_calibrator=probability_calibrator,
            probability_temperature=probability_temperature,
            disqualified_mask=rule_scores["disqualified"],
        )
        scores = blended

        print(f"  Blend weights: rule={blend_weights['rule_score']}, ml={blend_weights['ml_score']}")
        print(f"  Decision threshold: {decision_threshold}")
        print(f"  Средний final_score: {scores['final_score'].mean():.1f}")
        print(f"  Медиана final_score: {scores['final_score'].median():.1f}")
        print(f"  Диапазон: {scores['final_score'].min():.1f} — {scores['final_score'].max():.1f}")
        risk_dist = scores["final_risk_level"].value_counts().to_dict()
        print(f"  Риск-распределение (blended): {risk_dist}")
        decision_positive = scores["decision_predicted_positive"].sum()
        print(f"  Decision positive by final_score: {decision_positive}/{len(scores)}")
    else:
        print("\n[4/5] ML скоринг пропущен (модель не найдена)...")
        scores = rule_scores

    # 5. advisory + shortlist
    print("\n[5/5] Historical advisory + shortlist (топ-10)...")
    advisory = build_history_advisory_batch(df)

    shortlist = _generate_shortlist_blended(df, scores, advisory, top_n=10, ml_mode=ml_mode)

    # детальное объяснение для топ-1 заявки
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНОЕ ОБЪЯСНЕНИЕ — Топ-1 заявка")
    print("=" * 60)
    top_idx = scores["score"].idxmax()
    top_features = features.loc[top_idx].to_dict()
    rule_result = score_single(top_features)

    if ml_mode:
        top_final = scores.loc[top_idx, "final_score"]
        top_ml = scores.loc[top_idx, "ml_score"]
        top_rule = scores.loc[top_idx, "rule_score"]
        top_risk = scores.loc[top_idx, "final_risk_level"]
        print(f"  Final Score: {top_final}/100 (Риск: {top_risk})")
        print(f"    Rule: {top_rule:.1f} × {blend_weights['rule_score']}"
              f"  +  ML: {top_ml:.1f} × {blend_weights['ml_score']}")
    else:
        print(f"  Score: {rule_result.score}/100 (Риск: {rule_result.risk_level})")

    print("\n  Rule-факторы:")
    for line in rule_result.explanation:
        print(f"  {line}")

    top_advisory = advisory.loc[top_idx]
    print(f"\n  Advisory: {top_advisory['history_recommendation']}")
    print(f"    {top_advisory['history_note']}")

    return df, features, scores


def _generate_shortlist_blended(
    df, scores, advisory, top_n=50, output_dir="output", ml_mode=False,
):
    """Формирование shortlist с blended-колонками."""
    import pandas as pd
    combined = pd.concat([df, scores, advisory], axis=1)

    score_col = "final_score" if "final_score" in combined.columns else "score"
    combined = combined.sort_values(score_col, ascending=False).head(top_n)

    columns = ["app_number", "region", "direction", "subsidy_type", "amount"]
    if ml_mode and "final_score" in combined.columns:
        columns += ["final_score", "rule_score", "ml_score", "final_risk_level",
                     "ml_predicted_positive", "history_recommendation"]
    else:
        columns += ["score", "risk_level", "top_factor_label"]

    available = [c for c in columns if c in combined.columns]
    shortlist = combined[available]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "shortlist.csv"
    shortlist.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Shortlist сохранён → {csv_path}")
    print(shortlist.to_string(index=False))

    return shortlist


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    print("=" * 60)
    print("SUBSIDY SCORING API v4.0 (single-model merit ML + historical advisory)")
    print(f"Запуск на http://{host}:{port}")
    print(f"Swagger UI: http://{host}:{port}/docs")
    print("ML bundle path: models/artifacts/subsidy_model.joblib")
    print("=" * 60)

    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsidy Scoring System v4.0")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Запустить FastAPI-сервер вместо CLI-пайплайна",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Хост для сервера (по умолчанию: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для сервера (по умолчанию: 8000)",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Путь к ML-модели (по умолчанию: models/artifacts/subsidy_model.joblib)",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Принудительно использовать только rule-based скоринг",
    )

    args = parser.parse_args()

    if args.serve:
        run_server(host=args.host, port=args.port)
    else:
        mp = None if args.rule_only else args.model_path
        run_cli(DATA_PATH, model_path=mp)
