import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from api_client import (
    check_health,
    get_regions,
    get_directions,
    get_stats,
    rank_applications,
    get_explanation,
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


# графики

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
        color_discrete_map={
            "Низкий": "#2d6a4f",
            "Средний": "#e9c46a",
            "Высокий": "#e63946",
        },
        labels={"score": "Балл", "risk_level": "Уровень риска", "count": "Кол-во"},
    )
    fig.update_layout(bargap=0.05, height=350, margin=dict(t=10, b=30))
    st.plotly_chart(fig, width="stretch")


# таблица шортлиста

def render_shortlist(applications: list[dict]) -> pd.DataFrame:
    st.markdown('<p class="section-header">📋 Shortlist заявок</p>', unsafe_allow_html=True)

    if not applications:
        st.info("Нет заявок для отображения.")
        return pd.DataFrame()

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

    return shortlist


# детали заявки

def render_details(app_numbers: list[str]):
    st.markdown('<p class="section-header">🔎 Детали заявки</p>', unsafe_allow_html=True)

    if not app_numbers:
        st.warning("Нет заявок для отображения.")
        return

    selected_app = st.selectbox("Выберите заявку", app_numbers)

    detail = get_explanation(selected_app)

    # карточка со скором и риском
    risk_class = {
        "низкий": "risk-low", "средний": "risk-medium", "высокий": "risk-high",
    }.get(detail["risk_level"].lower(), "risk-medium")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="detail-card">
            <p class="detail-score-label">Score</p>
            <p class="detail-score">{detail['score']}</p>
            <span class="risk-badge {risk_class}">{detail['risk_level']} риск</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="detail-card">', unsafe_allow_html=True)
        for line in detail["explanation"]:
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
        st.markdown('</div>', unsafe_allow_html=True)

    # информация о заявке
    st.markdown(f"""
    <div class="app-info-row">
        <div>Регион: <span>{detail.get("region", "—")}</span></div>
        <div>Направление: <span>{detail.get("direction", "—")}</span></div>
        <div>Тип: <span>{detail.get("subsidy_type", "—")}</span></div>
        <div>Сумма: <span>{detail.get("amount", 0):,.0f} ₸</span></div>
    </div>
    """, unsafe_allow_html=True)


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

    # статистика с учётом фильтров
    stats = get_stats(**filters)

    # ранжирование для графика и таблицы (до 1000 для гистограммы)
    rank_data = rank_applications(**filters, top_n=1000)
    applications = rank_data["applications"]

    st.sidebar.metric("Записей после фильтра", f"{rank_data['total_filtered']:,}")

    st.markdown(
        f'<p class="subtitle">Всего заявок: {health["records_loaded"]:,} · '
        f'После фильтра: {rank_data["total_filtered"]:,}</p>',
        unsafe_allow_html=True,
    )

    render_metrics(stats)
    render_charts(applications)
    render_shortlist(applications[:50])

    # детали — номера заявок из shortlist
    app_numbers = [a["app_number"] for a in applications[:50]]
    render_details(app_numbers)


if __name__ == "__main__":
    main()
