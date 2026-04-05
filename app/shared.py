import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from api_client import (
    check_health,
    get_regions,
    get_directions,
    get_stats,
    rank_applications,
)

RISK_COLORS = {
    "Низкий": "#2d6a4f",
    "Средний": "#e9c46a",
    "Высокий": "#e63946",
}

LEVEL_COLORS = {
    "высокий": "#2d6a4f",
    "средний": "#e9c46a",
    "низкий": "#e63946",
}

WEIGHTS = {
    "normative_match": 0.08,
    "amount_normative_integrity": 0.06,
    "amount_adequacy": 0.06,
    "budget_pressure": 0.13,
    "queue_position": 0.09,
    "region_specialization": 0.07,
    "region_direction_approval_rate": 0.08,
    "akimat_approval_rate": 0.05,
    "unit_count": 0.04,
    "direction_approval_rate": 0.03,
    "subsidy_type_approval_rate": 0.03,
    "pasture_compliance": 0.08,
    "mortality_compliance": 0.07,
    "grazing_utilization": 0.05,
    "criteria_complexity": 0.03,
    "direction_risk": 0.03,
    "regional_pasture_capacity": 0.02,
}

DEFAULT_VALUE_FACTORS = {"budget_pressure", "queue_position", "unit_count",
                         "pasture_compliance", "mortality_compliance", "grazing_utilization",
                         "criteria_complexity", "direction_risk", "regional_pasture_capacity"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#d0d0e0",
)


def load_css():
    css_path = Path(__file__).parent / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def check_api() -> dict | None:
    """Проверяет API и показывает ошибку если недоступен. Возвращает health или None."""
    health = check_health()
    if not health:
        st.error("API недоступен. Запустите сервер: `python main.py --serve`")
    return health


def render_sidebar() -> dict:
    """Отрисовывает фильтры в sidebar, возвращает dict фильтров."""
    st.sidebar.header("🔍 Фильтры")

    regions_data = get_regions()
    directions_data = get_directions()

    region_names = ["Все"] + [r["region"] for r in regions_data]
    selected_region = st.sidebar.selectbox("Регион", region_names)

    direction_names = ["Все"] + [d["direction"] for d in directions_data]
    selected_direction = st.sidebar.selectbox("Направление", direction_names)

    # сброс типа субсидии при смене направления
    prev_direction = st.session_state.get("_prev_direction")
    if prev_direction != selected_direction:
        st.session_state["_prev_direction"] = selected_direction
        st.session_state["subsidy_types"] = []

    stype_names = ["Все"] + st.session_state.get("subsidy_types", [])
    selected_stype = st.sidebar.selectbox("Тип субсидии", stype_names)

    risk_names = ["Все", "Низкий", "Средний", "Высокий"]
    selected_risk = st.sidebar.selectbox("Уровень риска", risk_names)

    score_min, score_max = st.sidebar.slider(
        "Диапазон баллов",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
    )

    filters = {
        "region": selected_region if selected_region != "Все" else None,
        "direction": selected_direction if selected_direction != "Все" else None,
        "subsidy_type": selected_stype if selected_stype != "Все" else None,
        "risk_level": selected_risk if selected_risk != "Все" else None,
        "min_score": score_min if score_min > 0 else None,
        "max_score": score_max if score_max < 100 else None,
    }

    return filters


def load_filtered_data(filters: dict) -> tuple[dict, dict]:
    """Загружает stats и rank_data по фильтрам, обновляет subsidy_types."""
    stats = get_stats(**filters)
    rank_data = rank_applications(**filters, top_n=1000)
    applications = rank_data["applications"]

    if not filters.get("subsidy_type"):
        new_types = sorted(set(a["subsidy_type"] for a in applications))
        old_types = st.session_state.get("subsidy_types", [])
        st.session_state["subsidy_types"] = new_types
        if old_types != new_types:
            st.rerun()

    st.sidebar.metric("Записей после фильтра", f"{rank_data['total_filtered']:,}")

    return stats, rank_data


def page_setup(title: str, icon: str = "🏛️") -> tuple[dict, dict, dict] | None:
    """Общая инициализация страницы: CSS, API check, фильтры, данные.
    Возвращает (filters, stats, rank_data) или None если API недоступен.
    """
    load_css()
    health = check_api()
    if not health:
        return None
    filters = render_sidebar()
    stats, rank_data = load_filtered_data(filters)
    return filters, stats, rank_data
