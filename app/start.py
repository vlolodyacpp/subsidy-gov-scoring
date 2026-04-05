import streamlit as st

st.set_page_config(
    page_title="Subsidy Scoring",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("home.py",                      title="Обзор",        icon="🏛️", default=True),
    st.Page("pages/1_📋_Шортлист.py",       title="Шортлист",     icon="📋"),
    st.Page("pages/2_🔎_Детали.py",         title="Детали",       icon="🔎"),
    st.Page("pages/3_⚖️_Сравнение.py",      title="Сравнение",    icon="⚖️"),
    st.Page("pages/4_📊_Аналитика.py",      title="Аналитика",    icon="📊"),
    st.Page("pages/5_📝_Новая_заявка.py",   title="Новая заявка", icon="📝"),
])
pg.run()
