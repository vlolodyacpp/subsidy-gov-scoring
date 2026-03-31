# модуль для работы с эталонными нормативами субсидий
from normatives_dict import NORMATIVES


def build_normative_lookup() -> dict[str, int]:
    lookup = {}
    for direction, entries in NORMATIVES.items():
        for entry in entries:
            stype = entry["subsidy_type"].strip()
            lookup[stype] = entry["normative"]
    return lookup


def build_direction_lookup() -> dict[str, str]:
    lookup = {}
    for direction, entries in NORMATIVES.items():
        for entry in entries:
            stype = entry["subsidy_type"].strip()
            lookup[stype] = direction
    return lookup


def build_unit_lookup() -> dict[str, str]:
    lookup = {}
    for direction, entries in NORMATIVES.items():
        for entry in entries:
            stype = entry["subsidy_type"].strip()
            lookup[stype] = entry["unit"]
    return lookup


# единое окно подачи для всех типов субсидий (п. Приложение 2)
DEADLINE_WINDOW = {
    "start_month": 1,
    "start_day": 20,
    "end_month": 12,
    "end_day": 20,
}


def get_normative_for_type(subsidy_type: str, lookup: dict[str, int] | None = None) -> int | None:

    if lookup is None:
        lookup = build_normative_lookup()

    stype = subsidy_type.strip()

    # точное совпадение
    if stype in lookup:
        return lookup[stype]

    # попытка нечёткого совпадения
    stype_lower = stype.lower()
    for key, val in lookup.items():
        if key.lower() == stype_lower:
            return val

    # поиск по вхождению подстроки для частично совпадающих названий
    for key, val in lookup.items():
        key_lower = key.lower()
        if len(stype_lower) > 30 and len(key_lower) > 30:
            if stype_lower in key_lower or key_lower in stype_lower:
                return val

    return None


def check_deadline_compliance(submit_date) -> float:
    import pandas as pd

    if pd.isna(submit_date):
        return 0.5

    month = submit_date.month
    day = submit_date.day

    start_m, start_d = DEADLINE_WINDOW["start_month"], DEADLINE_WINDOW["start_day"]
    end_m, end_d = DEADLINE_WINDOW["end_month"], DEADLINE_WINDOW["end_day"]

    if month < start_m or (month == start_m and day < start_d):
        return 0.0

    if month > end_m or (month == end_m and day > end_d):
        return 0.0

    return 1.0
