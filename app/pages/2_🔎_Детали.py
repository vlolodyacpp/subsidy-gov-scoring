import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from shared import (
    WEIGHTS,
    LEVEL_COLORS,
    DEFAULT_VALUE_FACTORS,
    PLOTLY_LAYOUT,
    page_setup,
)
from api_client import get_explanation

FACTOR_GROUPS = {
    "Нормативное соответствие": ["normative_match", "amount_normative_integrity", "amount_adequacy"],
    "Сроки и бюджет": ["deadline_compliance", "budget_pressure", "queue_position"],
    "Региональная специфика": ["region_specialization", "region_direction_approval_rate", "akimat_approval_rate"],
    "Характеристики заявки": ["unit_count", "direction_approval_rate", "subsidy_type_approval_rate"],
}

GROUP_ICONS = {
    "Нормативное соответствие": "📋",
    "Сроки и бюджет": "📅",
    "Региональная специфика": "🗺️",
    "Характеристики заявки": "📌",
}


def render_details(detail: dict):
    risk_class = {
        "низкий": "risk-low", "средний": "risk-medium", "высокий": "risk-high",
    }.get(detail["risk_level"].lower(), "risk-medium")

    # --- карточка скора + информация о заявке ---
    col_score, col_info = st.columns([1, 3])

    with col_score:
        st.markdown(f"""
        <div class="detail-card">
            <p class="detail-score-label">Score</p>
            <p class="detail-score">{detail['score']}</p>
            <span class="risk-badge {risk_class}">{detail['risk_level']} риск</span>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="detail-card">
            <div class="app-info-grid">
                <div>Регион: <span>{detail.get("region", "—")}</span></div>
                <div>Направление: <span>{detail.get("direction", "—")}</span></div>
                <div>Тип: <span>{detail.get("subsidy_type", "—")}</span></div>
                <div>Сумма: <span>{detail.get("amount", 0):,.0f} ₸</span></div>
                <div>Статус: <span>{detail.get("status", "—")}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- норматив: заявленный vs эталонный ---
    ref_norm = detail.get("ref_normative")
    app_norm = detail.get("normative")
    if ref_norm or app_norm:
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            st.metric("Норматив (эталон)", f"{ref_norm:,.0f} ₸" if ref_norm else "—")
        with col_n2:
            st.metric("Норматив (заявка)", f"{app_norm:,.0f} ₸" if app_norm else "—")
        with col_n3:
            if ref_norm and app_norm:
                match = "Совпадает" if abs(ref_norm - app_norm) < 0.01 else "Не совпадает"
                st.metric("Соответствие", match)
            else:
                st.metric("Соответствие", "—")

    # --- график факторов ---
    factors = detail.get("factors", [])
    if factors:
        st.markdown('<p class="section-header">Разбивка по факторам</p>', unsafe_allow_html=True)

        factors_df = pd.DataFrame(factors)
        factors_df["max_contribution"] = factors_df["name"].map(
            lambda n: WEIGHTS.get(n, 0) * 100
        )
        factors_df = factors_df.sort_values("contribution", ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=factors_df["label"],
            x=factors_df["max_contribution"],
            orientation="h",
            name="Макс. вклад (вес)",
            marker_color="rgba(100, 100, 140, 0.3)",
            hovertemplate="%{y}: вес %{x:.0f}%<extra>Максимум</extra>",
        ))

        colors = factors_df["level"].map(LEVEL_COLORS).tolist()
        fig.add_trace(go.Bar(
            y=factors_df["label"],
            x=factors_df["contribution"],
            orientation="h",
            name="Фактический вклад",
            marker_color=colors,
            customdata=factors_df[["value", "name"]].values,
            hovertemplate=(
                "%{y}: %{x:.1f} баллов<br>"
                "Значение: %{customdata[0]:.2f}<extra>Факт</extra>"
            ),
        ))

        fig.update_layout(
            barmode="overlay",
            height=420,
            margin=dict(l=0, r=20, t=10, b=30),
            yaxis_title="",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    # предупреждение о дефолтных значениях
    default_factors = [
        f for f in factors
        if f["name"] in DEFAULT_VALUE_FACTORS and abs(f["value"] - 0.5) < 0.01
    ]
    if default_factors:
        names = ", ".join(f["label"] for f in default_factors)
        st.warning(
            f"Метрики **{names}** имеют значение 0.5 (дефолт). "
            "Это может означать, что заявка была оценена без контекста батча "
            "(через /score), и реальные значения могут отличаться."
        )

    # --- объяснение по группам факторов ---
    st.markdown('<p class="explanation-header">📝 Почему такой балл?</p>', unsafe_allow_html=True)

    factors_by_name = {f["name"]: f for f in factors}

    # 4 группы по 2 в ряд
    group_items = list(FACTOR_GROUPS.items())
    for row_start in range(0, len(group_items), 2):
        cols = st.columns(2)
        for col_idx, (group_name, group_factor_names) in enumerate(group_items[row_start:row_start + 2]):
            with cols[col_idx]:
                group_total = 0.0
                group_max = 0.0
                group_factors_data = []

                for fn in group_factor_names:
                    w = WEIGHTS.get(fn, 0)
                    group_max += w * 100
                    fd = factors_by_name.get(fn)
                    if fd:
                        group_total += fd["contribution"]
                        group_factors_data.append(fd)

                group_pct = group_total / group_max if group_max > 0 else 0

                if group_pct >= 0.7:
                    level_class = "high"
                    level_text = "сильная"
                elif group_pct >= 0.4:
                    level_class = "medium"
                    level_text = "средняя"
                else:
                    level_class = "low"
                    level_text = "слабая"

                icon = GROUP_ICONS.get(group_name, "📊")

                st.markdown(f"""
                <div class="group-card {level_class}">
                    <div class="group-card-header">
                        <span>{icon} {group_name}</span>
                        <span class="group-card-score">{group_total:.1f}/{group_max:.0f}</span>
                    </div>
                    <div class="group-card-bar-bg">
                        <div class="group-card-bar-fill {level_class}" style="width: {group_pct * 100:.0f}%"></div>
                    </div>
                    <div class="group-card-level">{level_text} ({group_pct:.0%})</div>
                </div>
                """, unsafe_allow_html=True)

                for fd in group_factors_data:
                    w = WEIGHTS.get(fd["name"], 0) * 100
                    fill_pct = (fd["contribution"] / w * 100) if w > 0 else 0
                    level_color = LEVEL_COLORS.get(fd["level"], "#888")

                    st.markdown(f"""
                    <div class="factor-row">
                        <div class="factor-row-header">
                            <span class="factor-row-label">{fd['label']}</span>
                            <span class="factor-row-value">+{fd['contribution']:.1f} <span class="factor-row-max">/ {w:.0f}</span></span>
                        </div>
                        <div class="factor-row-bar-bg">
                            <div class="factor-row-bar-fill" style="width: {fill_pct:.0f}%; background: {level_color}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# --- page ---

st.set_page_config(page_title="Детали заявки", page_icon="🔎", layout="wide")
st.markdown('<p class="main-title">🔎 Детали заявки</p>', unsafe_allow_html=True)

result = page_setup("Детали")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]
    app_numbers = [a["app_number"] for a in applications[:50]]

    # два способа выбрать заявку: из списка или ввести номер
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
