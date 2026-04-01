import streamlit as st
import pandas as pd

from shared import page_setup


def render_shortlist(applications: list[dict]):
    if not applications:
        st.info("Нет заявок для отображения.")
        return

    shortlist = pd.DataFrame(applications).reset_index(drop=True)

    st.dataframe(
        shortlist,
        use_container_width=True,
        height=600,
        column_config={
            "app_number": "Номер заявки",
            "region": "Регион",
            "district": "Район",
            "direction": "Направление",
            "subsidy_type": "Тип субсидии",
            "amount": st.column_config.NumberColumn("Сумма", format="%.0f ₸"),
            "score": st.column_config.ProgressColumn("Балл", min_value=0, max_value=100),
            "risk_level": "Риск",
            "top_factor": "Главный фактор",
        },
    )


st.set_page_config(page_title="Шортлист", page_icon="📋", layout="wide")
st.markdown('<p class="main-title">📋 Шортлист заявок</p>', unsafe_allow_html=True)

result = page_setup("Шортлист")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]

    st.markdown(
        f'<p class="subtitle">Найдено: {rank_data["total_filtered"]:,} · '
        f'Показано: {min(len(applications), 50)}</p>',
        unsafe_allow_html=True,
    )

    render_shortlist(applications[:50])
