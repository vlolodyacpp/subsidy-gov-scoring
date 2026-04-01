import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from shared import (
    RISK_COLORS,
    WEIGHTS,
    PLOTLY_LAYOUT,
    page_setup,
)
from api_client import get_factor_stats


def _fmt(val, fmt=".1f"):
    return f"{val:{fmt}}" if val is not None else "—"


def render_metrics(stats: dict):
    st.markdown('<p class="section-header">📊 Ключевые метрики</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Средний балл", _fmt(stats.get("mean_score")))
    col2.metric("Медиана", _fmt(stats.get("median_score")))
    col3.metric("Всего заявок", f"{stats.get('total_records', 0):,}")

    high_risk = stats.get("risk_distribution", {}).get("Высокий", 0)
    col4.metric("Высокий риск", f"{high_risk:,}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Ст. отклонение", _fmt(stats.get("std_score")))
    col6.metric("Минимум", _fmt(stats.get("min_score")))
    col7.metric("Максимум", _fmt(stats.get("max_score")))


def render_risk_donut(stats: dict):
    st.markdown('<p class="section-header">🎯 Распределение по риску</p>', unsafe_allow_html=True)

    risk_dist = stats["risk_distribution"]
    labels = list(risk_dist.keys())
    values = list(risk_dist.values())
    colors = [RISK_COLORS.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=13),
        hovertemplate="%{label}: %{value:,} заявок<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")


def render_top_regions(stats: dict):
    st.markdown('<p class="section-header">🏆 Топ регионов</p>', unsafe_allow_html=True)

    top_regions = stats.get("top_regions", [])
    if not top_regions:
        st.info("Нет данных по регионам.")
        return

    df = pd.DataFrame(top_regions).sort_values("avg_score", ascending=True)

    fig = px.bar(
        df,
        x="avg_score",
        y="region",
        orientation="h",
        color="avg_score",
        color_continuous_scale="Tealgrn",
        hover_data={"count": True, "approval_rate": ":.2%", "avg_score": ":.1f"},
        labels={
            "avg_score": "Средний балл",
            "region": "",
            "count": "Заявок",
            "approval_rate": "Одобряемость",
        },
    )
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=30, l=0, r=20),
        coloraxis_showscale=False,
        yaxis_title="",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")


def render_charts(applications: list[dict]):
    st.markdown('<p class="section-header">📈 Распределение скоров</p>', unsafe_allow_html=True)

    if not applications:
        st.info("Нет данных для графика.")
        return

    df = pd.DataFrame(applications)

    fig = px.histogram(
        df,
        x="score",
        nbins=30,
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        labels={"score": "Балл", "risk_level": "Уровень риска", "count": "Кол-во"},
    )
    fig.update_layout(
        bargap=0.05,
        height=350,
        margin=dict(t=10, b=30),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")


def render_factor_distributions(filters: dict):
    """Средние значения каждого фактора — bar chart."""
    st.markdown('<p class="section-header">📊 Средние значения факторов</p>', unsafe_allow_html=True)

    factor_data = get_factor_stats(
        region=filters.get("region"),
        direction=filters.get("direction"),
        subsidy_type=filters.get("subsidy_type"),
        min_score=filters.get("min_score"),
        max_score=filters.get("max_score"),
    )

    if not factor_data:
        st.info("Нет данных.")
        return

    rows = []
    for name, data in factor_data.items():
        weight = WEIGHTS.get(name, 0)
        rows.append({
            "factor": data["label"],
            "mean": data["mean"],
            "weight": weight,
            "weighted_contrib": round(data["mean"] * weight * 100, 1),
            "max_contrib": round(weight * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("weighted_contrib", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["factor"],
        x=df["max_contrib"],
        orientation="h",
        name="Макс. возможный",
        marker_color="rgba(100, 100, 140, 0.3)",
        hovertemplate="%{y}: макс. %{x:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=df["factor"],
        x=df["weighted_contrib"],
        orientation="h",
        name="Средний вклад",
        marker_color="#4cc9f0",
        customdata=df["mean"].values,
        hovertemplate="%{y}: %{x:.1f} баллов (среднее значение: %{customdata:.2f})<extra></extra>",
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


def main():
    st.set_page_config(
        page_title="Subsidy Scoring",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown('<p class="main-title">🏛️ Subsidy Scoring System</p>', unsafe_allow_html=True)

    result = page_setup("Главная")
    if not result:
        return
    filters, stats, rank_data = result
    applications = rank_data["applications"]

    st.markdown(
        f'<p class="subtitle">Всего заявок: {rank_data["total_filtered"]:,}</p>',
        unsafe_allow_html=True,
    )

    render_metrics(stats)

    col_left, col_right = st.columns(2)
    with col_left:
        render_risk_donut(stats)
    with col_right:
        render_top_regions(stats)

    render_charts(applications)
    render_factor_distributions(filters)


if __name__ == "__main__":
    main()
