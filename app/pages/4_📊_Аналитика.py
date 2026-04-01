import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from shared import WEIGHTS, PLOTLY_LAYOUT, page_setup
from api_client import get_factor_stats, get_region_factors, get_timeline

# короткие подписи факторов для heatmap
FACTOR_SHORT = {
    "normative_match": "Норматив",
    "amount_normative_integrity": "Корректность суммы",
    "amount_adequacy": "Адекватность суммы",
    "deadline_compliance": "Срок подачи",
    "budget_pressure": "Бюджет",
    "queue_position": "Очередь",
    "region_specialization": "Специализация",
    "region_direction_approval_rate": "Одобр. напр. в регионе",
    "akimat_approval_rate": "Одобр. акимата",
    "unit_count": "Кол-во единиц",
    "direction_approval_rate": "Одобр. направления",
    "subsidy_type_approval_rate": "Одобр. типа",
}

MONTH_NAMES = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
    7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}


def render_factor_histograms(filters: dict):
    """Гистограммы распределения по каждому фактору."""
    st.markdown('<p class="section-header">📊 Распределение по метрикам</p>', unsafe_allow_html=True)

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

    # выбор фактора
    factor_names = list(factor_data.keys())
    factor_labels = {k: v["label"] for k, v in factor_data.items()}
    label_to_name = {v: k for k, v in factor_labels.items()}

    selected_label = st.selectbox(
        "Выберите метрику",
        [factor_labels[n] for n in factor_names],
        key="factor_hist_select",
    )
    selected_factor = label_to_name[selected_label]
    data = factor_data[selected_factor]

    # статистика
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Среднее", f"{data['mean']:.3f}")
    col2.metric("Медиана", f"{data['median']:.3f}")
    col3.metric("Ст. откл.", f"{data['std']:.3f}")
    col4.metric("Вес", f"{WEIGHTS.get(selected_factor, 0):.0%}")

    # гистограмма
    values = data["values"]
    fig = px.histogram(
        x=values,
        nbins=40,
        labels={"x": selected_label, "count": "Кол-во заявок"},
        color_discrete_sequence=["#4cc9f0"],
    )
    fig.add_vline(
        x=data["mean"], line_dash="dash", line_color="#f72585",
        annotation_text=f"Среднее: {data['mean']:.3f}",
        annotation_font_color="#f72585",
    )
    fig.update_layout(
        height=400,
        bargap=0.05,
        margin=dict(t=30, b=30),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")


def render_region_breakdown(filters: dict):
    """Тепловая карта + таблица: регионы x факторы."""
    st.markdown('<p class="section-header">🗺️ Регионы: разбивка по факторам</p>', unsafe_allow_html=True)

    region_data = get_region_factors(
        direction=filters.get("direction"),
        subsidy_type=filters.get("subsidy_type"),
    )

    if not region_data:
        st.info("Нет данных.")
        return

    factor_keys = list(WEIGHTS.keys())
    short_labels = [FACTOR_SHORT.get(fk, fk) for fk in factor_keys]

    # полные подписи из API для hover
    full_labels = []
    for fk in factor_keys:
        f = region_data[0]["factors"].get(fk, {})
        full_labels.append(f.get("label", fk))

    # собираем данные
    regions = []
    counts = []
    matrix = []

    for entry in region_data[:15]:
        regions.append(entry["region"])
        counts.append(entry["count"])
        row = []
        for fk in factor_keys:
            f = entry["factors"].get(fk, {})
            row.append(round(f.get("mean", 0), 3))
        matrix.append(row)

    tab_heatmap, tab_table = st.tabs(["Тепловая карта", "Таблица"])

    with tab_heatmap:
        # hover показывает полное название, ось X — короткое
        hover_text = []
        for i, region in enumerate(regions):
            hover_row = []
            for j, fk in enumerate(factor_keys):
                hover_row.append(
                    f"Регион: {region}<br>"
                    f"{full_labels[j]}<br>"
                    f"Среднее: {matrix[i][j]:.3f}"
                )
            hover_text.append(hover_row)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=short_labels,
            y=[f"{r} ({c})" for r, c in zip(regions, counts)],
            colorscale="Tealgrn",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            texttemplate="%{z:.2f}",
            textfont=dict(size=10),
        ))
        fig.update_layout(
            height=max(450, len(regions) * 40),
            margin=dict(t=10, b=80, l=0, r=0),
            xaxis=dict(tickangle=45, side="bottom"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    with tab_table:
        # таблица с полными названиями
        table_data = []
        for i, region in enumerate(regions):
            row_dict = {"Регион": region, "Заявок": counts[i]}
            for j, fk in enumerate(factor_keys):
                row_dict[full_labels[j]] = matrix[i][j]
            table_data.append(row_dict)

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, hide_index=True, width="stretch", height=500)


def render_timeline(filters: dict):
    """Динамика бюджетного давления и позиции в очереди по месяцам."""
    st.markdown(
        '<p class="section-header">📅 Динамика по месяцам: бюджет и очередь</p>',
        unsafe_allow_html=True,
    )

    timeline = get_timeline(
        region=filters.get("region"),
        direction=filters.get("direction"),
    )

    if not timeline:
        st.info("Нет данных.")
        return

    df = pd.DataFrame(timeline)
    df["month_name"] = df["month"].map(MONTH_NAMES)

    # два графика рядом
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Бюджетное давление**")
        st.caption("Чем ниже — тем больше бюджет исчерпан к моменту подачи")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["month_name"],
            y=df["avg_budget_pressure"],
            marker_color=df["avg_budget_pressure"].apply(
                lambda v: "#2d6a4f" if v > 0.6 else "#e9c46a" if v > 0.3 else "#e63946"
            ).tolist(),
            hovertemplate="Месяц: %{x}<br>Бюджетное давление: %{y:.3f}<br>Заявок: %{customdata}<extra></extra>",
            customdata=df["count"],
        ))
        fig.update_layout(
            height=350,
            margin=dict(t=10, b=30),
            yaxis=dict(range=[0, 1], title="Доступность бюджета"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("**Позиция в очереди**")
        st.caption("Чем выше — тем раньше подана заявка относительно конкурентов")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["month_name"],
            y=df["avg_queue_position"],
            marker_color="#4cc9f0",
            hovertemplate="Месяц: %{x}<br>Позиция в очереди: %{y:.3f}<br>Заявок: %{customdata}<extra></extra>",
            customdata=df["count"],
        ))
        fig.update_layout(
            height=350,
            margin=dict(t=10, b=30),
            yaxis=dict(range=[0, 1], title="Позиция (0 = последний, 1 = первый)"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    # количество заявок по месяцам + средний балл
    st.markdown("**Объём заявок и средний балл**")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["month_name"],
        y=df["count"],
        name="Кол-во заявок",
        marker_color="rgba(76, 201, 240, 0.4)",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=df["month_name"],
        y=df["avg_score"],
        name="Средний балл",
        mode="lines+markers",
        line=dict(color="#f72585", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        height=350,
        margin=dict(t=10, b=30),
        yaxis=dict(title="Заявок", side="left"),
        yaxis2=dict(title="Средний балл", side="right", overlaying="y", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")


# --- page ---

st.set_page_config(page_title="Аналитика", page_icon="📊", layout="wide")
st.markdown('<p class="main-title">📊 Аналитика факторов</p>', unsafe_allow_html=True)

result = page_setup("Аналитика")
if result:
    filters, stats, rank_data = result

    render_factor_histograms(filters)
    render_region_breakdown(filters)
    render_timeline(filters)
