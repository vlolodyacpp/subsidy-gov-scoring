import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from src.pipeline import run_pipeline
from src.features import build_feature_tables, extract_features_batch
from src.scoring import score_batch, score_single, generate_shortlist, get_score_distribution


# стили 

def load_css():
    css_path = Path(__file__).parent / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# загрузка данных 

@st.cache_data
def load_data():
    df = run_pipeline("data/subsidies.xlsx")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)
    scores = score_batch(features)

    combined = df.copy()
    combined["score"] = scores["score"].values
    combined["risk_level"] = scores["risk_level"].values
    combined["top_factor_label"] = scores["top_factor_label"].values

    return combined, features, scores


# блок 3: sidebar с фильтрами 

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Фильтры")

    regions = ["Все"] + sorted(df["region"].unique().tolist())
    selected_region = st.sidebar.selectbox("Регион", regions)

    directions = ["Все"] + sorted(df["direction"].unique().tolist())
    selected_direction = st.sidebar.selectbox("Направление", directions)

    score_min, score_max = st.sidebar.slider(
        "Диапазон баллов",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
    )

    filtered = df.copy()
    if selected_region != "Все":
        filtered = filtered[filtered["region"] == selected_region]
    if selected_direction != "Все":
        filtered = filtered[filtered["direction"] == selected_direction]
    filtered = filtered[
        (filtered["score"] >= score_min) & (filtered["score"] <= score_max)
    ]

    st.sidebar.divider()
    st.sidebar.metric("Записей после фильтра", f"{len(filtered):,}")

    return filtered


# блок 4: метрики 

def render_metrics(filtered: pd.DataFrame):
    st.markdown('<p class="section-header">📊 Ключевые метрики</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Средний балл", f"{filtered['score'].mean():.1f}")
    col2.metric("Медиана", f"{filtered['score'].median():.1f}")
    col3.metric("Всего заявок", f"{len(filtered):,}")

    risk_counts = filtered["risk_level"].value_counts()
    high_risk = risk_counts.get("Высокий", 0)
    col4.metric("Высокий риск", f"{high_risk:,}")


# блок 5: графики 

def render_charts(filtered: pd.DataFrame):
    st.markdown('<p class="section-header">📈 Распределение скоров</p>', unsafe_allow_html=True)

    fig = px.histogram(
        filtered,
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
    st.plotly_chart(fig, use_container_width=True)


# блок 6: таблица шортлиста 

def render_shortlist(filtered: pd.DataFrame):
    st.markdown('<p class="section-header">📋 Shortlist заявок</p>', unsafe_allow_html=True)

    display_cols = [
        "app_number", "region", "district", "direction",
        "subsidy_type", "amount", "score", "risk_level", "top_factor_label",
    ]
    shortlist = (
        filtered[display_cols]
        .sort_values("score", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    st.dataframe(
        shortlist,
        use_container_width=True,
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
            "top_factor_label": "Главный фактор",
        },
    )

    return shortlist


# блок 7: детали заявки 

def render_details(filtered: pd.DataFrame, features: pd.DataFrame):
    st.markdown('<p class="section-header">🔎 Детали заявки</p>', unsafe_allow_html=True)

    app_numbers = filtered["app_number"].tolist()
    if not app_numbers:
        st.warning("Нет заявок для отображения.")
        return

    selected_app = st.selectbox("Выберите заявку", app_numbers)

    row = filtered[filtered["app_number"] == selected_app].iloc[0]
    row_idx = filtered[filtered["app_number"] == selected_app].index[0]
    app_features = features.loc[row_idx].to_dict()
    result = score_single(app_features)

    # карточка со скором и риском
    risk_class = {
        "низкий": "risk-low", "средний": "risk-medium", "высокий": "risk-high",
    }.get(result.risk_level, "risk-medium")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="detail-card">
            <p class="detail-score-label">Score</p>
            <p class="detail-score">{result.score}</p>
            <span class="risk-badge {risk_class}">{result.risk_level} риск</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="detail-card">', unsafe_allow_html=True)
        for line in result.explanation:
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
        <div>Регион: <span>{row.get("region", "—")}</span></div>
        <div>Направление: <span>{row.get("direction", "—")}</span></div>
        <div>Тип: <span>{row.get("subsidy_type", "—")}</span></div>
        <div>Сумма: <span>{row.get("amount", 0):,.0f} ₸</span></div>
    </div>
    """, unsafe_allow_html=True)


# точка входа 

def main():
    st.set_page_config(page_title="Subsidy Scoring", page_icon="🏛️", layout="wide")
    load_css()

    st.markdown('<p class="main-title">🏛️ Subsidy Scoring System</p>', unsafe_allow_html=True)

    combined, features, scores = load_data()
    filtered = render_sidebar(combined)

    st.markdown(
        f'<p class="subtitle">Всего заявок: {len(combined):,} · После фильтра: {len(filtered):,}</p>',
        unsafe_allow_html=True,
    )

    render_metrics(filtered)
    render_charts(filtered)
    render_shortlist(filtered)
    render_details(filtered, features)


if __name__ == "__main__":
    main()