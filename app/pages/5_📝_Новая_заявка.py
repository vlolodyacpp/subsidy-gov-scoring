import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import load_css, check_api
from api_client import (
    get_regions,
    get_directions,
    get_subsidy_types,
    get_akimats,
    score_new_application,
)
from detail_components import render_details


load_css()

st.markdown('<p class="main-title">📝 Оценка новой заявки</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Введите параметры заявки — система рассчитает скоринговый балл и факторный анализ</p>',
    unsafe_allow_html=True,
)

if not check_api():
    st.stop()

st.markdown('<p class="section-header">Параметры заявки</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    regions_data = get_regions()
    region_names = [r["region"] for r in regions_data]
    selected_region = st.selectbox("Регион *", region_names, key="new_region")

    directions_data = get_directions()
    direction_names = [d["direction"] for d in directions_data]
    selected_direction = st.selectbox("Направление субсидирования *", direction_names, key="new_direction")

with col2:
    subsidy_types_data = get_subsidy_types(selected_direction)
    subsidy_type_names = [s["subsidy_type"] for s in subsidy_types_data]
    selected_subsidy_type = st.selectbox(
        "Тип субсидии *",
        subsidy_type_names if subsidy_type_names else ["— нет данных —"],
        key="new_subsidy_type",
    )

    amount = st.number_input(
        "Сумма заявки (тенге) *",
        min_value=0,
        max_value=1_000_000_000,
        value=500_000,
        step=10_000,
        key="new_amount",
    )

# Акимат берётся автоматически — первый доступный для выбранного региона
akimats_data = get_akimats(selected_region)
default_akimat = akimats_data[0]["akimat"] if akimats_data else None

col3, col4 = st.columns(2)
with col3:
    submit_month = st.slider("Месяц подачи", min_value=1, max_value=12, value=3, key="new_month")
with col4:
    submit_day = st.slider("День подачи", min_value=1, max_value=31, value=15, key="new_day")

# --- Кнопка ---
st.markdown("")
st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #2d6a4f 0%, #1a4a35 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(45, 106, 79, 0.4);
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3a8a65 0%, #2d6a4f 100%);
        box-shadow: 0 6px 20px rgba(45, 106, 79, 0.6);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Оценить заявку →", type="primary"):
    if not selected_subsidy_type or selected_subsidy_type == "— нет данных —":
        st.error("Выберите тип субсидии.")
    else:
        payload = {
            "region": selected_region,
            "direction": selected_direction,
            "subsidy_type": selected_subsidy_type,
            "amount": float(amount),
            "submit_month": submit_month,
            "submit_day": submit_day,
        }
        if default_akimat:
            payload["akimat"] = default_akimat

        with st.spinner("Считаем оценку..."):
            try:
                result = score_new_application(payload)
                result.setdefault("region", selected_region)
                result.setdefault("direction", selected_direction)
                result.setdefault("subsidy_type", selected_subsidy_type)
                result.setdefault("amount", amount)
                result.setdefault("status", "Новая")
                st.session_state["new_app_result"] = result
            except Exception as e:
                st.error(f"Ошибка при оценке: {e}")
                st.session_state.pop("new_app_result", None)

if "new_app_result" in st.session_state:
    result = st.session_state["new_app_result"]
    st.divider()
    st.markdown('<p class="section-header">Результат оценки</p>', unsafe_allow_html=True)
    render_details(result)
