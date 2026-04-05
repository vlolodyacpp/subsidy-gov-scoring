import streamlit as st
import pandas as pd

from shared import page_setup

STATUS_LABELS = {
    "Исполнена": "Одобрена",
    "Одобрена": "Одобрена",
    "Сформировано поручение": "Одобрена",
    "Получена": "Одобрена",
    "Отклонена": "Отклонена",
}


def render_shortlist(applications: list[dict], scoring_engine: str | None = None):
    if not applications:
        st.info("Нет заявок для отображения.")
        return

    shortlist = pd.DataFrame(applications).reset_index(drop=True)

    shortlist["status"] = shortlist["status"].map(
        lambda s: STATUS_LABELS.get(s, s)
    )

    has_ml = "ml_score" in shortlist.columns and shortlist["ml_score"].notna().any()

    column_config = {
        "app_number": "Номер заявки",
        "region": "Регион",
        "district": "Район",
        "direction": "Направление",
        "subsidy_type": "Тип субсидии",
        "amount": st.column_config.NumberColumn("Сумма", format="%.0f ₸"),
        "status": "Статус",
        "score": st.column_config.NumberColumn("Итог. балл", format="%.1f"),
        "risk_level": "Риск",
        "top_factor": "Главный фактор",
    }

    if has_ml:
        column_config["rule_score"] = st.column_config.NumberColumn("По правилам", format="%.1f")
        column_config["ml_score"] = st.column_config.NumberColumn("ML-модель", format="%.1f")
        column_config["history_recommendation"] = "Рекомендация"

    display_cols = [
        "app_number", "region", "direction", "subsidy_type",
        "amount", "status", "score",
    ]
    if has_ml:
        display_cols += ["rule_score", "ml_score"]
    display_cols += ["risk_level"]
    if has_ml:
        display_cols += ["history_recommendation"]
    else:
        display_cols += ["top_factor"]

    available = [c for c in display_cols if c in shortlist.columns]

    st.dataframe(
        shortlist[available],
        use_container_width=True,
        height=600,
        column_config=column_config,
    )

    csv = shortlist[available].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Скачать шортлист (CSV)",
        data=csv,
        file_name="shortlist.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="Шортлист", page_icon="📋", layout="wide")
st.markdown('<p class="main-title">📋 Шортлист заявок</p>', unsafe_allow_html=True)

result = page_setup("Шортлист")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]

    scoring_engine = rank_data.get("scoring_engine", "")
    model_name = rank_data.get("model_name")
    if model_name:
        st.caption(f"Движок: {scoring_engine} · Модель: {model_name}")

    top_n = st.radio("Количество заявок", [50, 100, 200], horizontal=True, key="shortlist_top_n")

    st.markdown(
        f'<p class="subtitle">Найдено: {rank_data["total_filtered"]:,} · '
        f'Показано: {min(len(applications), top_n)}</p>',
        unsafe_allow_html=True,
    )

    render_shortlist(applications[:top_n], scoring_engine)
