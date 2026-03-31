# загрузка, очистка, подготовка данных
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.normatives import build_normative_lookup, get_normative_for_type


def load_raw_data(path: str) -> pd.DataFrame:
    # загрузка xlsx
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Page 1"]

    COL_MAP = {
        0: "row_id", 1: "date_str", 4: "region", 5: "akimat",
        6: "app_number", 7: "direction", 8: "subsidy_type",
        9: "status", 10: "normative", 11: "amount", 12: "district",
    }
    records = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5:  # пропускаем заголовок
            continue
        record = {name: row[idx] for idx, name in COL_MAP.items()}
        if record["app_number"] is not None:
            records.append(record)

    wb.close()
    return pd.DataFrame(records)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # уточнённая целевая переменная
    # замена нормативов из датасета на эталонные из справочника
    df = df.dropna(subset=["app_number", "status", "region"])

    # парсинг даты
    df["submit_date"] = pd.to_datetime(
        df["date_str"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    )
    df["submit_month"] = df["submit_date"].dt.month
    df["submit_quarter"] = df["submit_date"].dt.quarter

    # числовые колонки
    df["normative_original"] = pd.to_numeric(df["normative"], errors="coerce").fillna(0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # категории
    for col in ["region", "direction", "subsidy_type", "district", "status", "akimat"]:
        df[col] = df[col].astype(str).str.strip()

    # замена нормативов
    norm_lookup = build_normative_lookup()
    df["normative"] = df["subsidy_type"].apply(
        lambda st: get_normative_for_type(st, norm_lookup) or 0
    )

    # уточнение целевой переменной
    # positive: заявка одобрена/исполнена
    # negative: заявка отклонена
    # exclude: заявка отозвана заявителем (не отказ комиссии)
    positive_statuses = {"Исполнена", "Одобрена", "Сформировано поручение", "Получена"}
    exclude_statuses = {"Отозвано"}

    # удаляем отозванные заявки из выборки
    n_before = len(df)
    df = df[~df["status"].isin(exclude_statuses)].copy()
    n_excluded = n_before - len(df)
    print(f"  Исключено записей со статусом 'Отозвано': {n_excluded}")

    # бинарная целевая переменная
    df["is_approved"] = df["status"].isin(positive_statuses).astype(int)

    return df.reset_index(drop=True)


def run_pipeline(path: str) -> pd.DataFrame:
    raw = load_raw_data(path)
    clean = clean_data(raw)
    print(f"Pipeline v2 complete: {len(clean)} records, "
          f"{clean['is_approved'].sum()} approved, "
          f"{(~clean['is_approved'].astype(bool)).sum()} rejected")
    return clean


if __name__ == "__main__":
    DATA_PATH = "data/subsidies.xlsx"
    df = run_pipeline(DATA_PATH)
    print(df.info())
    print(df.head())
