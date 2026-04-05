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
        width="stretch",
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



st.markdown('<p class="main-title">📋 Шортлист заявок</p>', unsafe_allow_html=True)

result = page_setup("Шортлист")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]

    scoring_engine = rank_data.get("scoring_engine", "")

    col_count, col_info = st.columns([1, 3])
    with col_count:
        top_n = st.number_input(
            "Количество заявок",
            min_value=1,
            max_value=len(applications),
            value=min(50, len(applications)),
            step=10,
            key="shortlist_top_n",
        )
    with col_info:
        st.markdown(
            f'<p class="subtitle" style="margin-top:2rem;">Найдено: {rank_data["total_filtered"]:,} · '
            f'Показано: {min(len(applications), top_n)}</p>',
            unsafe_allow_html=True,
        )

    render_shortlist(applications[:top_n], scoring_engine)
