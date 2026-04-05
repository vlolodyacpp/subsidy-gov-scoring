import streamlit as st

from shared import page_setup
from api_client import get_explanation
from detail_components import render_details


st.markdown('<p class="main-title">🔎 Детали заявки</p>', unsafe_allow_html=True)

result = page_setup("Детали")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]
    app_numbers = [a["app_number"] for a in applications[:50]]

    tab_list, tab_search = st.tabs(["Из шортлиста", "Поиск по номеру"])

    with tab_list:
        if app_numbers:
            selected_app = st.selectbox("Выберите заявку", app_numbers, key="detail_select")
            detail = get_explanation(selected_app)
            render_details(detail)
        else:
            st.info("Нет заявок в текущем фильтре.")

    with tab_search:
        search_id = st.text_input("Введите номер заявки", key="detail_search")
        if search_id:
            search_id = search_id.strip()
            try:
                detail = get_explanation(search_id)
                render_details(detail)
            except Exception:
                st.error(f"Заявка **{search_id}** не найдена.")
