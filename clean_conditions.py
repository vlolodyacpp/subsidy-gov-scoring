"""
Скрипт очистки дополнительных условий (cond1*, cond21, cond31, cond32).

Результаты сохраняются в data/cleaned/:
  - mortality_norms.csv       — нормы естественной убыли (падежа) по направлениям
  - pasture_norms.csv         — нормы нагрузки на пастбища по областям
  - subsidy_normatives.csv    — справочник нормативов субсидий (Приложение 1)
  - eligibility_criteria.csv  — критерии допуска и методы проверки ГИСС (Приложение 2)
"""

import pandas as pd
import numpy as np
import re
import os

OUT_DIR = "data/cleaned"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# COND1*: нормы естественной убыли (падежа)
# ---------------------------------------------------------------------------

COND1_FILES = {
    "data/cond11.xls": "Мясное и молочное скотоводство",
    "data/cond12.xls": "Овцеводство и козоводство",
    "data/cond13.xls": "Коневодство",
    "data/cond14.xls": "Верблюдоводство",
    "data/cond15.xls": "Мараловодство",
    "data/cond16.xls": "Свиноводство",
    "data/cond17.xls": "Звероводство",
    "data/cond18.xls": "Птицеводство",
    "data/cond19.xls": "Рыбоводство",
    "data/cond110.xls": "Пчеловодство",
}


def clean_mortality() -> pd.DataFrame:
    """Парсит все cond1* файлы → единая таблица норм убыли."""
    rows = []
    for path, direction in COND1_FILES.items():
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue

        dfs = pd.read_html(path, encoding="utf-8")
        df = dfs[0]
        # Колонки: 0=номер, 1=вид/группа, 2=процент убыли
        df.columns = ["row_num", "animal_group", "mortality_pct"]

        # Убираем строку-заголовок (первая строка = шапка)
        df = df.iloc[1:]

        # Определяем текущую категорию верхнего уровня
        current_category = None
        for _, row in df.iterrows():
            num = str(row["row_num"]).strip()
            group = str(row["animal_group"]).strip()
            pct_raw = str(row["mortality_pct"]).strip()

            # Пропускаем пустые строки
            if group == "nan" or group == "":
                continue

            # Строки верхнего уровня (целое число вроде "1.", "2.") — категории
            if re.match(r"^\d+\.$", num) and (pct_raw == "nan" or pct_raw == group):
                current_category = group
                continue

            # Пытаемся извлечь процент
            pct = _parse_number(pct_raw)
            if pct is None:
                # Это подзаголовок без числа
                if ":" in group or pct_raw == "nan":
                    current_category = group.rstrip(":").strip() if current_category is None else current_category
                continue

            rows.append({
                "direction": direction,
                "category": current_category or "",
                "animal_group": group,
                "mortality_pct": pct,
            })

    result = pd.DataFrame(rows)
    print(f"[cond1*] Нормы убыли: {len(result)} записей, "
          f"{result['direction'].nunique()} направлений")
    return result


def _parse_number(s: str):
    """Пытается извлечь число из строки (может быть '25', '065', 'nan')."""
    s = s.strip().replace("\xa0", "").replace(",", ".")
    if s in ("nan", "", "-"):
        return None
    # Убираем ведущие нули, но сохраняем '0' и дробные
    try:
        return float(s)
    except ValueError:
        # Попробуем извлечь первое число (минимум одна цифра)
        m = re.search(r"\d[\d.]*", s)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# COND21: нормы нагрузки на пастбища
# ---------------------------------------------------------------------------

def clean_pasture() -> pd.DataFrame:
    """Парсит cond21.xls → таблица нагрузок на пастбища по областям."""
    path = "data/cond21.xls"
    dfs = pd.read_html(path, encoding="utf-8")
    df = dfs[0]

    # Первые 4 строки — многоуровневый заголовок
    # Строка 0: общие названия
    # Строка 1: виды животных (КРС, овцы/козы, лошади, верблюды)
    # Строка 2: восстановленные / деградированные
    # Строка 3: номера колонок (1-13)
    # Данные начинаются со строки 4

    data = df.iloc[4:].copy()
    data.columns = [
        "oblast",
        "nature_zone",
        "eco_district",
        "pasture_type",
        "grazing_period_days",
        "cattle_restored",
        "cattle_degraded",
        "sheep_goats_restored",
        "sheep_goats_degraded",
        "horses_restored",
        "horses_degraded",
        "camels_restored",
        "camels_degraded",
    ]

    data = data.reset_index(drop=True)

    # Убираем строки, где oblast — нумерация или NaN
    data = data[data["oblast"].notna()]
    data = data[~data["oblast"].astype(str).str.match(r"^\d+$")]

    # Парсим числовые колонки
    num_cols = [
        "cattle_restored", "cattle_degraded",
        "sheep_goats_restored", "sheep_goats_degraded",
        "horses_restored", "horses_degraded",
        "camels_restored", "camels_degraded",
    ]
    for col in num_cols:
        data[col] = data[col].apply(_parse_number)

    # Парсим период выпаса: "210-230" → берём среднее
    def parse_period(s):
        s = str(s).strip()
        m = re.findall(r"\d+", s)
        if len(m) >= 2:
            return (int(m[0]) + int(m[1])) / 2
        elif len(m) == 1:
            return float(m[0])
        return np.nan

    data["grazing_period_days"] = data["grazing_period_days"].apply(parse_period)

    # Чистим строковые поля
    for col in ["oblast", "nature_zone", "eco_district", "pasture_type"]:
        data[col] = data[col].astype(str).str.strip()

    # forward-fill области и природных зон (объединённые ячейки)
    data["oblast"] = data["oblast"].replace("nan", np.nan).ffill()
    data["nature_zone"] = data["nature_zone"].replace("nan", np.nan).ffill()

    data = data.dropna(subset=num_cols, how="all")

    print(f"[cond21] Пастбищные нормы: {len(data)} записей, "
          f"{data['oblast'].nunique()} областей")
    return data


# ---------------------------------------------------------------------------
# COND31: справочник нормативов субсидий
# ---------------------------------------------------------------------------

def clean_normatives() -> pd.DataFrame:
    """Парсит cond31.xls → справочник нормативов."""
    path = "data/cond31.xls"
    dfs = pd.read_html(path, encoding="utf-8")
    df = dfs[0]

    # Колонки: 0=номер, 1=вид субсидии, 2=единица, 3=норматив (тенге)
    df.columns = ["row_num", "subsidy_type", "unit", "normative_tenge"]

    # Убираем заголовочную строку
    df = df.iloc[1:]

    current_direction = None
    rows = []

    for _, row in df.iterrows():
        num = str(row["row_num"]).strip()
        stype = str(row["subsidy_type"]).strip()
        unit = str(row["unit"]).strip()
        norm_raw = str(row["normative_tenge"]).strip()

        if stype == "nan" or stype == "":
            continue

        # Строки-заголовки направлений: все 4 колонки одинаковые
        if stype == unit == norm_raw or (norm_raw == stype):
            current_direction = stype
            continue

        norm = _parse_number(norm_raw)
        if norm is None and unit == "nan":
            # Подзаголовок (например "Приобретение маточного поголовья:")
            continue

        rows.append({
            "direction": current_direction or "",
            "row_num": num,
            "subsidy_type": stype,
            "unit": unit if unit != "nan" else "",
            "normative_tenge": norm,
        })

    result = pd.DataFrame(rows)
    # Некоторые строки — подзаголовки без норматива, но с единицей
    result = result.dropna(subset=["normative_tenge"])

    print(f"[cond31] Нормативы субсидий: {len(result)} записей, "
          f"{result['direction'].nunique()} направлений")
    return result


# ---------------------------------------------------------------------------
# COND32: критерии допуска (Приложение 2)
# ---------------------------------------------------------------------------

def clean_criteria() -> pd.DataFrame:
    """Парсит cond32.xls → критерии и методы проверки."""
    path = "data/cond32.xls"
    dfs = pd.read_html(path, encoding="utf-8")
    df = dfs[0]

    # Колонки: 0=номер, 1=вид субсидии, 2=критерии, 3=метод проверки ГИСС, 4=срок подачи
    df.columns = ["row_num", "subsidy_type", "criteria", "verification_method", "deadline"]

    df = df.iloc[1:]

    current_direction = None
    rows = []

    for _, row in df.iterrows():
        num = str(row["row_num"]).strip()
        stype = str(row["subsidy_type"]).strip()
        criteria = str(row["criteria"]).strip()
        method = str(row["verification_method"]).strip()
        deadline = str(row["deadline"]).strip()

        if stype == "nan" or stype == "":
            continue

        # Строки-заголовки направлений
        if stype == criteria or (method == stype and deadline == stype):
            current_direction = stype
            continue

        # Чистим критерии: разбиваем по номерам
        criteria_clean = criteria if criteria != "nan" else ""

        rows.append({
            "direction": current_direction or "",
            "row_num": num,
            "subsidy_type": stype,
            "criteria": criteria_clean,
            "verification_method": method if method != "nan" else "",
            "deadline": deadline if deadline != "nan" else "",
        })

    result = pd.DataFrame(rows)
    print(f"[cond32] Критерии допуска: {len(result)} записей, "
          f"{result['direction'].nunique()} направлений")
    return result


# ---------------------------------------------------------------------------
# Подсчёт числа критериев для каждого вида субсидии
# ---------------------------------------------------------------------------

def count_criteria(criteria_df: pd.DataFrame) -> pd.DataFrame:
    """Считает количество критериев в каждой строке cond32."""
    def _count(text: str) -> int:
        if not text:
            return 0
        # Считаем пронумерованные пункты: "1.", "2.", "3." и т.д.
        return len(re.findall(r"(?:^|\s)\d+\.\s", text))

    df = criteria_df.copy()
    df["criteria_count"] = df["criteria"].apply(_count)
    return df[["direction", "subsidy_type", "criteria_count"]]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Очистка дополнительных условий")
    print("=" * 60)
    print()

    # 1. Нормы убыли
    mortality = clean_mortality()
    mortality.to_csv(f"{OUT_DIR}/mortality_norms.csv", index=False)
    print(f"  → {OUT_DIR}/mortality_norms.csv")
    print()

    # 2. Пастбищные нормы
    pasture = clean_pasture()
    pasture.to_csv(f"{OUT_DIR}/pasture_norms.csv", index=False)
    print(f"  → {OUT_DIR}/pasture_norms.csv")
    print()

    # 3. Нормативы субсидий
    normatives = clean_normatives()
    normatives.to_csv(f"{OUT_DIR}/subsidy_normatives.csv", index=False)
    print(f"  → {OUT_DIR}/subsidy_normatives.csv")
    print()

    # 4. Критерии допуска
    criteria = clean_criteria()
    criteria.to_csv(f"{OUT_DIR}/eligibility_criteria.csv", index=False)
    print(f"  → {OUT_DIR}/eligibility_criteria.csv")
    print()

    # Сводка: потенциальные новые фичи
    print("=" * 60)
    print("ПОТЕНЦИАЛЬНЫЕ НОВЫЕ ПРИЗНАКИ ДЛЯ МОДЕЛИ")
    print("=" * 60)
    print()

    print("Из cond1* (нормы убыли):")
    print(f"  Направлений: {mortality['direction'].nunique()}")
    print(f"  Записей: {len(mortality)}")
    print(f"  Диапазон %: {mortality['mortality_pct'].min():.0f}–{mortality['mortality_pct'].max():.0f}")
    print()

    print("Из cond21 (пастбищные нормы):")
    print(f"  Областей: {pasture['oblast'].nunique()}")
    print(f"  Записей: {len(pasture)}")
    print(f"  Уникальных природных зон: {pasture['nature_zone'].nunique()}")
    print()

    print("Из cond31 (нормативы):")
    print(f"  Направлений: {normatives['direction'].nunique()}")
    print(f"  Типов субсидий: {len(normatives)}")
    print()

    print("Из cond32 (критерии):")
    criteria_counts = count_criteria(criteria)
    print(f"  Типов субсидий: {len(criteria)}")
    print(f"  Среднее кол-во критериев: {criteria_counts['criteria_count'].mean():.1f}")
    print(f"  Макс кол-во критериев: {criteria_counts['criteria_count'].max()}")
    print()


if __name__ == "__main__":
    main()
