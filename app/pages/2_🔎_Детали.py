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


def render_details(app_numbers: list[str]):
    if not app_numbers:
        st.warning("Нет заявок для отображения.")
        return

    selected_app = st.selectbox("Выберите заявку", app_numbers)

    detail = get_explanation(selected_app)

    risk_class = {
        "низкий": "risk-low", "средний": "risk-medium", "высокий": "risk-high",
    }.get(detail["risk_level"].lower(), "risk-medium")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(f"""
        <div class="detail-card">
            <p class="detail-score-label">Score</p>
            <p class="detail-score">{detail['score']}</p>
            <span class="risk-badge {risk_class}">{detail['risk_level']} риск</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        factors = detail.get("factors", [])
        if factors:
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
            st.plotly_chart(fig, use_container_width=True)

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

    # информация о заявке
    st.markdown(f"""
    <div class="app-info-row">
        <div>Регион: <span>{detail.get("region", "—")}</span></div>
        <div>Направление: <span>{detail.get("direction", "—")}</span></div>
        <div>Тип: <span>{detail.get("subsidy_type", "—")}</span></div>
        <div>Сумма: <span>{detail.get("amount", 0):,.0f} ₸</span></div>
    </div>
    """, unsafe_allow_html=True)

    # текстовое объяснение
    st.markdown('<p class="explanation-header">📝 Почему такой балл?</p>', unsafe_allow_html=True)
    for line in detail.get("explanation", []):
        if "✓" in line:
            level = "high"
        elif "●" in line:
            level = "medium"
        else:
            level = "low"
        st.markdown(
            f'<div class="explanation-item {level}">{line}</div>',
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Детали заявки", page_icon="🔎", layout="wide")
st.markdown('<p class="main-title">🔎 Детали заявки</p>', unsafe_allow_html=True)

result = page_setup("Детали")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]
    app_numbers = [a["app_number"] for a in applications[:50]]
    render_details(app_numbers)
