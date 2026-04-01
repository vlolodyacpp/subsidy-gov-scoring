import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from api_client import (
    check_health,
    get_regions,
    get_directions,
    get_stats,
    rank_applications,
    get_explanation,
)

RISK_COLORS = {
    "Низкий": "#2d6a4f",
    "Средний": "#e9c46a",
    "Высокий": "#e63946",
}

LEVEL_COLORS = {
    "высокий": "#2d6a4f",
    "средний": "#e9c46a",
    "низкий": "#e63946",
}

# веса метрик из scoring.py — для отображения максимального вклада
WEIGHTS = {
    "normative_match": 0.10,
    "amount_normative_integrity": 0.08,
    "amount_adequacy": 0.08,
    "deadline_compliance": 0.08,
    "budget_pressure": 0.12,
    "queue_position": 0.10,
    "region_specialization": 0.10,
    "region_direction_approval_rate": 0.12,
    "akimat_approval_rate": 0.07,
    "unit_count": 0.05,
    "direction_approval_rate": 0.05,
    "subsidy_type_approval_rate": 0.05,
}

# факторы, которые получают дефолтное значение 0.5 при одиночном скоринге
DEFAULT_VALUE_FACTORS = {"budget_pressure", "queue_position", "unit_count"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#d0d0e0",
)


# стили

def load_css():
    css_path = Path(__file__).parent / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# sidebar с фильтрами — возвращает выбранные значения

def render_sidebar() -> dict:
    st.sidebar.header("🔍 Фильтры")

    # справочники из API
    regions_data = get_regions()
    directions_data = get_directions()

    region_names = ["Все"] + [r["region"] for r in regions_data]
    selected_region = st.sidebar.selectbox("Регион", region_names)

    direction_names = ["Все"] + [d["direction"] for d in directions_data]
    selected_direction = st.sidebar.selectbox("Направление", direction_names)

    # тип субсидии (заполняется из session_state после первого запроса)
    stype_names = ["Все"] + st.session_state.get("subsidy_types", [])
    selected_stype = st.sidebar.selectbox("Тип субсидии", stype_names)

    score_min, score_max = st.sidebar.slider(
        "Диапазон баллов",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
    )

    filters = {
        "region": selected_region if selected_region != "Все" else None,
        "direction": selected_direction if selected_direction != "Все" else None,
        "subsidy_type": selected_stype if selected_stype != "Все" else None,
        "min_score": score_min if score_min > 0 else None,
        "max_score": score_max if score_max < 100 else None,
    }

    return filters


# метрики

def render_metrics(stats: dict):
    st.markdown('<p class="section-header">📊 Ключевые метрики</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Средний балл", f"{stats['mean_score']:.1f}")
    col2.metric("Медиана", f"{stats['median_score']:.1f}")
    col3.metric("Всего заявок", f"{stats['total_records']:,}")

    high_risk = stats["risk_distribution"].get("Высокий", 0)
    col4.metric("Высокий риск", f"{high_risk:,}")

    # дополнительная статистика
    col5, col6, col7 = st.columns(3)
    col5.metric("Ст. отклонение", f"{stats['std_score']:.1f}")
    col6.metric("Минимум", f"{stats['min_score']:.1f}")
    col7.metric("Максимум", f"{stats['max_score']:.1f}")


# распределение по рискам (donut)

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


# топ регионов по среднему баллу

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


# графики распределения скоров

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


# таблица шортлиста

def render_shortlist(applications: list[dict]):
    st.markdown('<p class="section-header">📋 Shortlist заявок</p>', unsafe_allow_html=True)

    if not applications:
        st.info("Нет заявок для отображения.")
        return

    shortlist = pd.DataFrame(applications).reset_index(drop=True)

    st.dataframe(
        shortlist,
        width="stretch",
        height=400,
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


# детали заявки

def render_details(app_numbers: list[str]):
    st.markdown('<p class="section-header">🔎 Детали заявки</p>', unsafe_allow_html=True)

    if not app_numbers:
        st.warning("Нет заявок для отображения.")
        return

    selected_app = st.selectbox("Выберите заявку", app_numbers)

    detail = get_explanation(selected_app)

    # карточка со скором + график факторов
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
        # grouped bar chart: фактический вклад vs максимально возможный
        factors = detail.get("factors", [])
        if factors:
            factors_df = pd.DataFrame(factors)
            factors_df["max_contribution"] = factors_df["name"].map(
                lambda n: WEIGHTS.get(n, 0) * 100
            )
            factors_df = factors_df.sort_values("contribution", ascending=True)

            fig = go.Figure()

            # максимально возможный вклад (вес × 100) — полупрозрачный фон
            fig.add_trace(go.Bar(
                y=factors_df["label"],
                x=factors_df["max_contribution"],
                orientation="h",
                name="Макс. вклад (вес)",
                marker_color="rgba(100, 100, 140, 0.3)",
                hovertemplate="%{y}: вес %{x:.0f}%<extra>Максимум</extra>",
            ))

            # фактический вклад — цвет по уровню
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

    # информация о заявке
    st.markdown(f"""
    <div class="app-info-row">
        <div>Регион: <span>{detail.get("region", "—")}</span></div>
        <div>Направление: <span>{detail.get("direction", "—")}</span></div>
        <div>Тип: <span>{detail.get("subsidy_type", "—")}</span></div>
        <div>Сумма: <span>{detail.get("amount", 0):,.0f} ₸</span></div>
    </div>
    """, unsafe_allow_html=True)

    # текстовое объяснение — видимо по умолчанию
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


# сравнение двух заявок

def render_comparison(app_numbers: list[str]):
    st.markdown('<p class="section-header">⚖️ Сравнение заявок</p>', unsafe_allow_html=True)

    if len(app_numbers) < 2:
        st.info("Для сравнения нужно минимум 2 заявки.")
        return

    col1, col2 = st.columns(2)
    with col1:
        app_a = st.selectbox("Заявка A", app_numbers, index=0, key="cmp_a")
    with col2:
        app_b = st.selectbox("Заявка B", app_numbers, index=1, key="cmp_b")

    if app_a == app_b:
        st.info("Выберите две разные заявки для сравнения.")
        return

    detail_a = get_explanation(app_a)
    detail_b = get_explanation(app_b)

    factors_a = {f["name"]: f for f in detail_a.get("factors", [])}
    factors_b = {f["name"]: f for f in detail_b.get("factors", [])}

    all_names = list(WEIGHTS.keys())
    labels = [factors_a.get(n, factors_b.get(n, {})).get("label", n) for n in all_names]
    values_a = [factors_a.get(n, {}).get("contribution", 0) for n in all_names]
    values_b = [factors_b.get(n, {}).get("contribution", 0) for n in all_names]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=f"{app_a} ({detail_a['score']:.1f})",
        line_color="#4cc9f0",
        fillcolor="rgba(76, 201, 240, 0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=f"{app_b} ({detail_b['score']:.1f})",
        line_color="#f72585",
        fillcolor="rgba(247, 37, 133, 0.15)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color="#555", gridcolor="#333"),
            angularaxis=dict(color="#aaa", gridcolor="#333"),
        ),
        height=500,
        margin=dict(t=40, b=40, l=80, r=80),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    # таблица сравнения
    comparison_data = []
    for n in all_names:
        fa = factors_a.get(n, {})
        fb = factors_b.get(n, {})
        label = fa.get("label", fb.get("label", n))
        weight = WEIGHTS.get(n, 0)
        val_a = fa.get("contribution", 0)
        val_b = fb.get("contribution", 0)
        diff = val_a - val_b
        comparison_data.append({
            "Метрика": label,
            "Вес": f"{weight:.0%}",
            f"A ({app_a})": f"{val_a:.1f}",
            f"B ({app_b})": f"{val_b:.1f}",
            "Разница (A-B)": f"{diff:+.1f}",
        })
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, width="stretch")


# точка входа

def main():
    st.set_page_config(
        page_title="Subsidy Scoring",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()

    st.markdown('<p class="main-title">🏛️ Subsidy Scoring System</p>', unsafe_allow_html=True)

    # проверка доступности API
    health = check_health()
    if not health:
        st.error(
            "API недоступен. Запустите сервер: `python main.py --serve`"
        )
        return

    # фильтры
    filters = render_sidebar()

    # статистика (/stats не принимает subsidy_type)
    stats_filters = {k: v for k, v in filters.items() if k != "subsidy_type"}
    stats = get_stats(**stats_filters)

    # ранжирование для графика и таблицы
    rank_data = rank_applications(**filters, top_n=1000)
    applications = rank_data["applications"]

    # обновляем список типов субсидий для фильтра
    st.session_state["subsidy_types"] = sorted(set(
        a["subsidy_type"] for a in applications
    ))

    st.sidebar.metric("Записей после фильтра", f"{rank_data['total_filtered']:,}")

    st.markdown(
        f'<p class="subtitle">Всего заявок: {health["records_loaded"]:,} · '
        f'После фильтра: {rank_data["total_filtered"]:,}</p>',
        unsafe_allow_html=True,
    )

    render_metrics(stats)

    # donut рисков + топ регионов рядом
    col_left, col_right = st.columns(2)
    with col_left:
        render_risk_donut(stats)
    with col_right:
        render_top_regions(stats)

    render_charts(applications)
    render_shortlist(applications[:50])

    # детали — номера заявок из shortlist
    app_numbers = [a["app_number"] for a in applications[:50]]
    render_details(app_numbers)

    # сравнение двух заявок
    render_comparison(app_numbers)


if __name__ == "__main__":
    main()
