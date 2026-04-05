import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from shared import (
    RISK_COLORS,
    WEIGHTS,
    PLOTLY_LAYOUT,
    page_setup,
    load_css,
)
from api_client import get_factor_stats, upload_dataset, check_health, get_retrain_status


def _fmt(val, fmt=".1f"):
    return f"{val:{fmt}}" if val is not None else "—"


def render_quick_summary(stats: dict, rank_data: dict):
    applications = rank_data.get("applications", [])
    if not applications:
        return

    st.markdown('<p class="section-header">Сводка по рискам</p>', unsafe_allow_html=True)

    import collections
    risk_scores = collections.defaultdict(list)
    for app in applications:
        risk_scores[app.get("risk_level", "—")].append(app["score"])

    cols = st.columns(len(risk_scores))
    for col, (risk, scores) in zip(cols, sorted(risk_scores.items())):
        avg = sum(scores) / len(scores)
        color = RISK_COLORS.get(risk.capitalize(), "#888")
        col.markdown(
            f'<div style="background:#1a1a2e;border:1px solid #3a3a5a;border-left:4px solid {color};'
            f'border-radius:12px;padding:12px 16px;">'
            f'<span style="font-size:0.85rem;color:#8888aa;text-transform:uppercase;">{risk} риск</span><br>'
            f'<span style="font-size:1.8rem;font-weight:700;color:{color}">{avg:.1f}</span>'
            f'<span style="font-size:0.9rem;color:#6a6a80"> ср. балл</span><br>'
            f'<span style="font-size:0.95rem;color:#b0b0c8">{len(scores):,} заявок</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_metrics(stats: dict):
    st.markdown('<p class="section-header">Ключевые метрики</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Итоговый балл (ср.)", _fmt(stats.get("mean_score")))
    col2.metric("Медиана", _fmt(stats.get("median_score")))
    col3.metric("Всего заявок", f"{stats.get('total_records', 0):,}")

    high_risk = stats.get("risk_distribution", {}).get("Высокий", 0)
    col4.metric("Высокий риск", f"{high_risk:,}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Ст. отклонение", _fmt(stats.get("std_score")))
    col6.metric("Минимум", _fmt(stats.get("min_score")))
    col7.metric("Максимум", _fmt(stats.get("max_score")))


def render_risk_donut(stats: dict):
    st.markdown('<p class="section-header">Распределение по уровню риска</p>', unsafe_allow_html=True)

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
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value:,} заявок<extra></extra>",
    ))
    fig.update_layout(
        height=400,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_regions(stats: dict):
    st.markdown('<p class="section-header">Топ регионов по итоговому баллу</p>', unsafe_allow_html=True)

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
            "avg_score": "Средний итоговый балл",
            "region": "",
            "count": "Количество заявок",
            "approval_rate": "Доля одобренных",
        },
    )
    fig.update_layout(
        height=400,
        margin=dict(t=10, b=30, l=0, r=20),
        coloraxis_showscale=False,
        yaxis_title="",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_charts(applications: list[dict]):
    st.markdown('<p class="section-header">Распределение итоговых баллов</p>', unsafe_allow_html=True)

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
        labels={"score": "Итоговый балл", "risk_level": "Уровень риска", "count": "Количество"},
    )
    fig.update_layout(
        bargap=0.05,
        height=620,
        margin=dict(t=10, b=30),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
            title_text="",
        ),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_factor_distributions(filters: dict):
    st.markdown('<p class="section-header">Средний вклад каждого фактора в итоговый балл</p>', unsafe_allow_html=True)

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
        name="Максимально возможный вклад",
        marker_color="rgba(100, 100, 140, 0.3)",
        hovertemplate="%{y}: макс. %{x:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=df["factor"],
        x=df["weighted_contrib"],
        orientation="h",
        name="Фактический средний вклад",
        marker_color="#4cc9f0",
        customdata=df["mean"].values,
        hovertemplate="%{y}: %{x:.1f} баллов (среднее: %{customdata:.2f})<extra></extra>",
    ))

    fig.update_layout(
        barmode="overlay",
        height=max(500, len(df) * 30),
        margin=dict(l=0, r=20, t=10, b=30),
        yaxis_title="",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
            title_text="",
        ),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dataset_panel():
    health = check_health()
    ds_name = (health or {}).get("dataset_name") or st.session_state.get(
        "dataset_name", "subsidies.xlsx (по умолчанию)"
    )
    records = (health or {}).get("records_loaded", 0)

    retrain = get_retrain_status()
    status = retrain.get("status", "idle")

    col_status, col_upload = st.columns([2, 1])
    with col_status:
        st.markdown(
            f'<div style="background:#1a1a2e;border:1px solid #3a3a5a;border-radius:12px;padding:1rem 1.2rem;">'
            f'<span style="color:#8888aa;font-size:0.85rem;">Текущий датасет:</span> '
            f'<span style="color:#d0d0e0;font-weight:600;">{ds_name}</span> '
            f'<span style="color:#6a6a80;">({records:,} записей)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_upload:
        uploaded = st.file_uploader(
            "Загрузить .xlsx",
            type=["xlsx", "xls"],
            key="dataset_uploader",
            label_visibility="collapsed",
        )
        if uploaded and st.button("Загрузить", type="primary", use_container_width=True):
            with st.spinner("Загрузка и скоринг..."):
                try:
                    res = upload_dataset(uploaded.getvalue(), uploaded.name)
                    st.session_state["dataset_loaded"] = True
                    st.session_state["dataset_name"] = uploaded.name
                    st.cache_data.clear()
                    st.success(f"{uploaded.name} — {res['records_loaded']:,} записей загружено")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    if status == "training":
        st.info("Система обновляет оценки на новых данных...")
        import time
        time.sleep(5)
        st.rerun()
    elif status == "done":
        st.success("Оценки пересчитаны на новых данных!")
        st.cache_data.clear()
    elif status == "error":
        st.error(f"Ошибка обновления: {retrain.get('error', '')[:150]}")


load_css()

st.markdown('<p class="main-title">🏛️ Система оценки заявок на субсидии</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Аналитика, скоринг и ранжирование заявок на государственные субсидии</p>', unsafe_allow_html=True)

render_dataset_panel()

result = page_setup("Главная")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]

    st.markdown(
        f'<p class="subtitle">Всего заявок: {rank_data["total_filtered"]:,}</p>',
        unsafe_allow_html=True,
    )

    render_metrics(stats)
    render_quick_summary(stats, rank_data)

    col_left, col_right = st.columns(2)
    with col_left:
        render_risk_donut(stats)
    with col_right:
        render_top_regions(stats)

    render_charts(applications)
    render_factor_distributions(filters)
