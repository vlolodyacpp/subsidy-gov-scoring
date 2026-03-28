"""загрузка, очистка, подготовка данных."""
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_raw_data(path: str) -> pd.DataFrame:
    """загрузка xlsx."""
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
    """очищаем датафрейм от пустот, парсим даты, типизируем"""
    df = df.dropna(subset=["app_number", "status", "region"])

    # парсинг даты
    df["submit_date"] = pd.to_datetime(
        df["date_str"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    )
    df["submit_month"] = df["submit_date"].dt.month
    df["submit_quarter"] = df["submit_date"].dt.quarter

    # числовые колонки
    df["normative"] = pd.to_numeric(df["normative"], errors="coerce").fillna(0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # упрощаем статусы до бинарного знчения для дальнейшей работы
    approved = {"Исполнена", "Одобрена", "Сформировано поручение"}
    df["is_approved"] = df["status"].isin(approved).astype(int)

    # категории
    for col in ["region", "direction", "subsidy_type", "district", "status"]:
        df[col] = df[col].astype(str).str.strip()

    return df.reset_index(drop=True)


def run_pipeline(path: str) -> pd.DataFrame:
    """запускаем пайплайн"""
    raw = load_raw_data(path)
    clean = clean_data(raw)
    print(f"Pipeline complete: {len(clean)} records, "
          f"{clean['is_approved'].sum()} approved, "
          f"{(~clean['is_approved'].astype(bool)).sum()} rejected/other")
    return clean


if __name__ == "__main__":
    DATA_PATH = "data/Выгрузка_по_выданным_субсидиям_2025_год_обезлич.xlsx"
    df = run_pipeline(DATA_PATH)
    print(df.info())
    print(df.head())
