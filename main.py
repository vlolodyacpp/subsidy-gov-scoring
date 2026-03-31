"""запуск полного пайплайна скоринга.(временное решение на первичном этапе)"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_PATH = "data/subsidies.xlsx"

from src.pipeline import run_pipeline
from src.features import build_feature_tables, extract_features_batch
from src.scoring import score_batch, generate_shortlist, get_score_distribution, score_single


def run_cli(data_path: str):
    """Запуск CLI-пайплайна скоринга v2."""
    from src.pipeline import run_pipeline
    from src.features import build_feature_tables, extract_features_batch
    from src.scoring import (
        score_batch, generate_shortlist, get_score_distribution,
        score_single, WEIGHTS, FACTOR_GROUPS,
    )

    print("=" * 60)
    print("SUBSIDY SCORING SYSTEM v2.0 (12 факторов)")
    print("=" * 60)

    # 1. загрузка и очистка
    print("\n[1/4] Загрузка данных + замена нормативов...")
    df = run_pipeline(data_path)

    # 2. построение справочных таблиц
    print("\n[2/4] Feature engineering v2...")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    print(f"  Признаков: {features.shape[1]}")
    print(f"  Записей: {features.shape[0]}")
    print(f"  Метрики: {list(features.columns)}")

    # 3. скоринг
    print("\n[3/4] Скоринг заявок (12 факторов)...")
    scores = score_batch(features)
    stats = get_score_distribution(scores)
    print(f"  Средний балл: {stats['mean']}")
    print(f"  Медиана: {stats['median']}")
    print(f"  Диапазон: {stats['min']} — {stats['max']}")
    print(f"  Риск-распределение: {stats['risk_distribution']}")

    # средние значения каждой метрики
    print("\n  Средние значения метрик:")
    for factor in WEIGHTS:
        if factor in features.columns:
            mean_val = features[factor].mean()
            print(f"    {factor}: {mean_val:.3f}")

    # 4. шортлист
    print("\n[4/4] Формирование shortlist (топ-10)...")
    shortlist = generate_shortlist(df, scores, top_n=10)

    # пример детального объяснения для топ-1 заявки
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНОЕ ОБЪЯСНЕНИЕ — Топ-1 заявка")
    print("=" * 60)
    top_idx = scores["score"].idxmax()
    top_features = features.loc[top_idx].to_dict()
    result = score_single(top_features)
    print(f"  Score: {result.score}/100 (Риск: {result.risk_level})")
    for line in result.explanation:
        print(f"  {line}")

    return df, features, scores


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    print("=" * 60)
    print("SUBSIDY SCORING API v2.0 (12 факторов)")
    print(f"Запуск на http://{host}:{port}")
    print(f"Swagger UI: http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsidy Scoring System v2")
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

    args = parser.parse_args()

    if args.serve:
        run_server(host=args.host, port=args.port)
    else:
        run_cli(DATA_PATH)
