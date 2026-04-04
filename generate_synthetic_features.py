"""
Генерация синтетических данных для новых факторов из cond-файлов.

Логика:
  1. Присоединяем справочные нормы (pasture, mortality, criteria) к каждой заявке
  2. Генерируем синтетические "фактические" значения, привязанные к статусу одобрения:
     - одобренные заявки — факт ближе к норме (compliant)
     - отклонённые — факт чаще отклоняется от нормы
  3. Вычисляем итоговые признаки: отклонение факта от нормы [0, 1]
  4. Сохраняем результат в data/cleaned/synthetic_features.csv

Когда появятся реальные данные — заменить генератор на настоящие значения,
итоговые признаки останутся в том же формате.
"""

import pandas as pd
import numpy as np
import re
import sys

sys.path.insert(0, ".")
from src.pipeline import load_raw_data, clean_data

SEED = 42

# ---------------------------------------------------------------------------
# Маппинги между основным датасетом и cond-файлами
# ---------------------------------------------------------------------------

REGION_MAP = {
    "Акмолинская область": "Акмолинская",
    "Актюбинская область": "Актюбинская",
    "Алматинская область": "Алматинская",
    "Атырауская область": "Атырауская",
    "Восточно-Казахстанская область": "Восточно-Казахстанская",
    "Жамбылская область": "Жамбылская",
    "Западно-Казахстанская область": "Западно-Казахстанская",
    "Карагандинская область": "Карагандинская",
    "Костанайская область": "Костанайская",
    "Кызылординская область": "Кызылординская",
    "Мангистауская область": "Мангистауская",
    "Павлодарская область": "Павлодарская",
    "Северо-Казахстанская область": "Северо-Казахстанская",
    "Туркестанская область": "Туркестанская",
    "г.Шымкент": "Туркестанская",  # Шымкент выделен, ближайшая — Туркестанская
    "область Абай": "Абай",
    "область Жетісу": "Жетісу",
    "область Ұлытау": "Ұлытау",
}

# direction в основном датасете → direction в mortality (cond1*)
DIRECTION_MORTALITY_MAP = {
    "Субсидирование в скотоводстве": "Мясное и молочное скотоводство",
    "Субсидирование в овцеводстве": "Овцеводство и козоводство",
    "Субсидирование в козоводстве": "Овцеводство и козоводство",
    "Субсидирование в коневодстве": "Коневодство",
    "Субсидирование в верблюдоводстве": "Верблюдоводство",
    "Субсидирование в свиноводстве": "Свиноводство",
    "Субсидирование в птицеводстве": "Птицеводство",
    "Субсидирование в пчеловодстве": "Пчеловодство",
    "Субсидирование затрат по искусственному осеменению": "Мясное и молочное скотоводство",
}

# direction → какой вид скота использовать для пастбищных норм
DIRECTION_PASTURE_ANIMAL = {
    "Субсидирование в скотоводстве": "cattle",
    "Субсидирование в овцеводстве": "sheep_goats",
    "Субсидирование в козоводстве": "sheep_goats",
    "Субсидирование в коневодстве": "horses",
    "Субсидирование в верблюдоводстве": "camels",
    "Субсидирование в свиноводстве": "cattle",  # свиньи не пастбищные, берём КРС как proxy
    "Субсидирование в птицеводстве": "cattle",  # аналогично
    "Субсидирование в пчеловодстве": "cattle",  # аналогично
    "Субсидирование затрат по искусственному осеменению": "cattle",
}

# direction в cond32 → direction в основном датасете (для criteria_count)
DIRECTION_CRITERIA_MAP = {
    "Мясное и мясо-молочное скотоводство": "Субсидирование в скотоводстве",
    "Молочное и молочно-мясное скотоводство": "Субсидирование в скотоводстве",
    "Скотоводство": "Субсидирование в скотоводстве",
    "Овцеводство": "Субсидирование в овцеводстве",
    "Коневодство": "Субсидирование в коневодстве",
    "Верблюдоводство": "Субсидирование в верблюдоводстве",
    "Свиноводство": "Субсидирование в свиноводстве",
    "Мясное птицеводство": "Субсидирование в птицеводстве",
    "Яичное птицеводство": "Субсидирование в птицеводстве",
}


# ---------------------------------------------------------------------------
# 1. Загрузка и присоединение справочных норм
# ---------------------------------------------------------------------------

def load_and_join_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Присоединяет справочные нормы к каждой заявке."""
    df = df.copy()

    # --- Пастбищные нормы (cond21) ---
    pasture = pd.read_csv("data/cleaned/pasture_norms.csv")
    # Средняя норма по области (усреднение по зонам и типам пастбищ)
    pasture_avg = (
        pasture.groupby("oblast")
        .agg(
            cattle_norm=("cattle_restored", "mean"),
            sheep_goats_norm=("sheep_goats_restored", "mean"),
            horses_norm=("horses_restored", "mean"),
            camels_norm=("camels_restored", "mean"),
            grazing_days=("grazing_period_days", "mean"),
        )
        .reset_index()
    )

    df["oblast_key"] = df["region"].map(REGION_MAP)
    df = df.merge(pasture_avg, left_on="oblast_key", right_on="oblast", how="left")

    # Выбираем норму под вид скота заявки
    df["pasture_animal"] = df["direction"].map(DIRECTION_PASTURE_ANIMAL)
    df["pasture_norm"] = np.nan
    for animal in ["cattle", "sheep_goats", "horses", "camels"]:
        mask = df["pasture_animal"] == animal
        df.loc[mask, "pasture_norm"] = df.loc[mask, f"{animal}_norm"]

    # --- Нормы убыли (cond1*) ---
    mortality = pd.read_csv("data/cleaned/mortality_norms.csv")
    # Средний и максимальный % падежа по направлению
    mort_agg = (
        mortality.groupby("direction")
        .agg(
            mortality_mean=("mortality_pct", "mean"),
            mortality_max=("mortality_pct", "max"),
        )
        .reset_index()
    )

    df["mortality_direction"] = df["direction"].map(DIRECTION_MORTALITY_MAP)
    df = df.merge(
        mort_agg, left_on="mortality_direction", right_on="direction",
        how="left", suffixes=("", "_mort"),
    )

    # --- Количество критериев (cond32) ---
    criteria = pd.read_csv("data/cleaned/eligibility_criteria.csv")
    criteria["criteria_count"] = criteria["criteria"].apply(_count_criteria)

    # Средний criteria_count по direction (cond32 direction → main direction)
    criteria["main_direction"] = criteria["direction"].map(DIRECTION_CRITERIA_MAP)
    crit_by_dir = (
        criteria.groupby("main_direction")["criteria_count"]
        .mean()
        .reset_index()
        .rename(columns={"main_direction": "direction", "criteria_count": "avg_criteria_count"})
    )
    df = df.merge(
        crit_by_dir, left_on="direction", right_on="direction",
        how="left", suffixes=("", "_crit"),
    )

    # Заполняем пропуски медианами
    df["pasture_norm"] = df["pasture_norm"].fillna(df["pasture_norm"].median())
    df["grazing_days"] = df["grazing_days"].fillna(df["grazing_days"].median())
    df["mortality_mean"] = df["mortality_mean"].fillna(df["mortality_mean"].median())
    df["mortality_max"] = df["mortality_max"].fillna(df["mortality_max"].median())
    df["avg_criteria_count"] = df["avg_criteria_count"].fillna(df["avg_criteria_count"].median())

    return df


def _count_criteria(text: str) -> int:
    if pd.isna(text) or not text:
        return 0
    return len(re.findall(r"(?:^|\s)\d+\.\s", str(text)))


# ---------------------------------------------------------------------------
# 2. Генерация синтетических "фактических" значений
# ---------------------------------------------------------------------------

def generate_synthetic_actuals(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Генерирует синтетические фактические значения, привязанные к is_approved.

    Принцип:
      - Одобренные (is_approved=1): факт ≈ norm * N(mu_good, sigma), т.е. ниже нормы
      - Отклонённые (is_approved=0): факт ≈ norm * N(mu_bad, sigma), т.е. около/выше нормы
      - Значительное перекрытие классов, чтобы признак не был идеальным предиктором
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    is_approved = df["is_approved"].values

    df = df.copy()

    # --- Фактическая нагрузка на пастбища (га/голову) ---
    # Норма — максимально допустимая нагрузка
    # Фактическая нагрузка ниже нормы = хорошо (меньше давление на пастбища)
    pasture_ratio_approved = rng.normal(0.65, 0.18, n)  # обычно 47-83% от нормы
    pasture_ratio_rejected = rng.normal(0.95, 0.25, n)  # обычно 70-120% от нормы
    pasture_ratio = np.where(is_approved == 1, pasture_ratio_approved, pasture_ratio_rejected)
    pasture_ratio = np.clip(pasture_ratio, 0.1, 2.0)
    df["actual_pasture_load"] = df["pasture_norm"].values * pasture_ratio

    # --- Фактический падёж (%) ---
    # Норма — максимально допустимый %. Факт ниже нормы = хорошо
    mort_ratio_approved = rng.normal(0.55, 0.20, n)  # обычно 35-75% от нормы
    mort_ratio_rejected = rng.normal(0.90, 0.30, n)  # обычно 60-120% от нормы
    mort_ratio = np.where(is_approved == 1, mort_ratio_approved, mort_ratio_rejected)
    mort_ratio = np.clip(mort_ratio, 0.0, 2.0)
    df["actual_mortality_pct"] = df["mortality_mean"].values * mort_ratio

    # --- Фактический период выпаса (дней) ---
    # Больше дней выпаса = лучше для хозяйства
    graze_ratio_approved = rng.normal(0.90, 0.10, n)  # 80-100% от нормы
    graze_ratio_rejected = rng.normal(0.70, 0.15, n)  # 55-85% от нормы
    graze_ratio = np.where(is_approved == 1, graze_ratio_approved, graze_ratio_rejected)
    graze_ratio = np.clip(graze_ratio, 0.3, 1.1)
    df["actual_grazing_days"] = df["grazing_days"].values * graze_ratio

    return df


# ---------------------------------------------------------------------------
# 3. Вычисление итоговых признаков [0, 1]
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет нормализованные признаки из фактических значений и норм."""
    df = df.copy()

    # --- pasture_compliance: насколько факт ниже нормы (1 = хорошо, 0 = плохо) ---
    # ratio = actual / norm.  ratio < 1 = хорошо, ratio > 1 = перегрузка
    pasture_ratio = df["actual_pasture_load"] / df["pasture_norm"].clip(lower=1)
    # Инвертируем и нормализуем: 1 при ratio=0, 0 при ratio>=1.5
    df["pasture_compliance"] = np.clip(1.0 - (pasture_ratio / 1.5), 0, 1)

    # --- mortality_compliance: насколько факт ниже нормы (1 = хорошо, 0 = плохо) ---
    mort_ratio = df["actual_mortality_pct"] / df["mortality_mean"].clip(lower=0.1)
    df["mortality_compliance"] = np.clip(1.0 - (mort_ratio / 1.5), 0, 1)

    # --- grazing_utilization: использование пастбищного сезона (1 = полный, 0 = неполный) ---
    graze_ratio = df["actual_grazing_days"] / df["grazing_days"].clip(lower=1)
    df["grazing_utilization"] = np.clip(graze_ratio, 0, 1)

    # --- criteria_complexity: сложность проверки (нормализованная) ---
    # Больше критериев = сложнее = ниже score
    max_crit = df["avg_criteria_count"].max()
    df["criteria_complexity"] = 1.0 - (df["avg_criteria_count"] / max_crit) if max_crit > 0 else 0.5

    # --- direction_risk: рисковость направления по базовой норме убыли (не синтетика) ---
    # Высокая средняя убыль = рисковое направление = ниже score
    max_mort = df["mortality_mean"].max()
    df["direction_risk"] = 1.0 - (df["mortality_mean"] / max_mort) if max_mort > 0 else 0.5

    # --- regional_pasture_capacity: ресурсная обеспеченность региона (не синтетика) ---
    # Маленькая норма (мало га на голову) = высокая ёмкость, хорошо
    # Большая норма (много га на голову) = низкая ёмкость, хуже
    max_pn = df["pasture_norm"].max()
    df["regional_pasture_capacity"] = 1.0 - (df["pasture_norm"] / max_pn) if max_pn > 0 else 0.5

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Генерация синтетических признаков из cond-данных")
    print("=" * 60)
    print()

    # Загрузка основного датасета
    print("[1/4] Загрузка основного датасета...")
    df = clean_data(load_raw_data("data/subsidies.xlsx"))
    print(f"  Строк: {len(df)}")
    print()

    # Присоединение справочных норм
    print("[2/4] Присоединение справочных норм...")
    df = load_and_join_norms(df)
    print(f"  pasture_norm:       {df['pasture_norm'].notna().sum()} / {len(df)} заполнено")
    print(f"  mortality_mean:     {df['mortality_mean'].notna().sum()} / {len(df)} заполнено")
    print(f"  avg_criteria_count: {df['avg_criteria_count'].notna().sum()} / {len(df)} заполнено")
    print(f"  grazing_days:       {df['grazing_days'].notna().sum()} / {len(df)} заполнено")
    print()

    # Генерация синтетических фактов
    print("[3/4] Генерация синтетических фактических значений...")
    df = generate_synthetic_actuals(df, seed=SEED)
    print("  Готово (seed=42)")
    print()

    # Вычисление итоговых признаков
    print("[4/4] Вычисление итоговых признаков...")
    df = compute_features(df)

    # Выбираем только нужные колонки для экспорта
    FEATURE_COLS = [
        "app_number",
        # Справочные нормы (из cond-файлов, join)
        "pasture_norm",
        "grazing_days",
        "mortality_mean",
        "mortality_max",
        "avg_criteria_count",
        # Синтетические фактические значения
        "actual_pasture_load",
        "actual_mortality_pct",
        "actual_grazing_days",
        # Итоговые признаки [0, 1]
        "pasture_compliance",
        "mortality_compliance",
        "grazing_utilization",
        "criteria_complexity",
        "direction_risk",
        "regional_pasture_capacity",
    ]
    result = df[FEATURE_COLS].copy()
    result.to_csv("data/cleaned/synthetic_features.csv", index=False)
    print(f"  Сохранено: data/cleaned/synthetic_features.csv ({len(result)} строк)")
    print()

    # --- Отчёт ---
    print("=" * 60)
    print("ОТЧЁТ ПО СГЕНЕРИРОВАННЫМ ПРИЗНАКАМ")
    print("=" * 60)
    print()

    feature_names = [
        "pasture_compliance",
        "mortality_compliance",
        "grazing_utilization",
        "criteria_complexity",
        "direction_risk",
        "regional_pasture_capacity",
    ]

    is_approved = df["is_approved"]
    print(f"{'Признак':<30} {'Mean':>6} {'Std':>6}  | {'Appr':>6} {'Rej':>6} {'Delta':>6}")
    print("-" * 75)
    for feat in feature_names:
        vals = df[feat]
        m_appr = vals[is_approved == 1].mean()
        m_rej = vals[is_approved == 0].mean()
        print(
            f"{feat:<30} {vals.mean():>6.3f} {vals.std():>6.3f}  | "
            f"{m_appr:>6.3f} {m_rej:>6.3f} {m_appr - m_rej:>+6.3f}"
        )

    print()
    print("Признаки с Delta > 0: одобренные выше (ожидаемо для синтетических)")
    print("Признаки с Delta ~ 0: справочные, не зависят от статуса")
    print()

    # Корреляция с target
    print("Корреляция с is_approved (point-biserial):")
    for feat in feature_names:
        corr = df[feat].corr(is_approved)
        bar = "#" * int(abs(corr) * 50)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:<30} r={corr:>+.3f}  {sign}{bar}")

    print()
    print("Готово.")


if __name__ == "__main__":
    main()
