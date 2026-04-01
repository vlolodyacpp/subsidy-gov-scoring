import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from shared import WEIGHTS, PLOTLY_LAYOUT, page_setup
from api_client import get_explanation


def render_comparison(app_numbers: list[str]):
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
    st.plotly_chart(fig, use_container_width=True)

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
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)


st.set_page_config(page_title="Сравнение", page_icon="⚖️", layout="wide")
st.markdown('<p class="main-title">⚖️ Сравнение заявок</p>', unsafe_allow_html=True)

result = page_setup("Сравнение")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]
    app_numbers = [a["app_number"] for a in applications[:50]]
    render_comparison(app_numbers)
