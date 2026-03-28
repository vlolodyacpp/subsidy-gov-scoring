"""запуск полного пайплайна скоринга.(временное решение на первичном этапе)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline
from src.features import build_feature_tables, extract_features_batch
from src.scoring import score_batch, generate_shortlist, get_score_distribution, score_single


def main(data_path: str):
    print("=" * 60)
    print("SUBSIDY SCORING SYSTEM v0.1 (Rule-based)")
    print("=" * 60)

    # 1. загрузка и очистка
    print("\n[1/4] Загрузка данных...")
    df = run_pipeline(data_path)

    # 2. построение справочных таблиц
    print("\n[2/4] Feature engineering...")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    print(f"  Признаков: {features.shape[1]}")
    print(f"  Записей: {features.shape[0]}")

    # 3. скоринг
    print("\n[3/4] Скоринг заявок...")
    scores = score_batch(features)
    stats = get_score_distribution(scores)
    print(f"  Средний балл: {stats['mean']}")
    print(f"  Медиана: {stats['median']}")
    print(f"  Диапазон: {stats['min']} — {stats['max']}")
    print(f"  Риск-распределение: {stats['risk_distribution']}")

    # 4. шортлист
    print("\n[4/4] Формирование shortlist (топ-10)...")
    shortlist = generate_shortlist(df, scores, top_n=10)

    # пример детального объяснения для топ-1 заявки
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНОЕ ОБЪЯСНЕНИЕ — Топ-1 заявка")
    print("=" * 60)
    top_idx = scores["score"].idxmax()
    top_features = features.iloc[top_idx].to_dict()
    result = score_single(top_features)
    print(f"  Score: {result.score}/100 (Риск: {result.risk_level})")
    for line in result.explanation:
        print(f"  {line}")

    return df, features, scores


if __name__ == "__main__":
    DATA_PATH = "data/subsidies.xlsx"
    main(DATA_PATH)
