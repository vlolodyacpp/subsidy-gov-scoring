import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from shared import (
    RISK_COLORS,
    PLOTLY_LAYOUT,
    page_setup,
)


def render_metrics(stats: dict):
    st.markdown('<p class="section-header">📊 Ключевые метрики</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Средний балл", f"{stats['mean_score']:.1f}")
    col2.metric("Медиана", f"{stats['median_score']:.1f}")
    col3.metric("Всего заявок", f"{stats['total_records']:,}")

    high_risk = stats["risk_distribution"].get("Высокий", 0)
    col4.metric("Высокий риск", f"{high_risk:,}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Ст. отклонение", f"{stats['std_score']:.1f}")
    col6.metric("Минимум", f"{stats['min_score']:.1f}")
    col7.metric("Максимум", f"{stats['max_score']:.1f}")


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


def main():
    st.set_page_config(
        page_title="Subsidy Scoring",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown('<p class="main-title">🏛️ Subsidy Scoring System</p>', unsafe_allow_html=True)

    result = page_setup("Обзор")
    if not result:
        return
    _filters, stats, rank_data = result
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


if __name__ == "__main__":
    main()
