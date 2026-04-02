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

FACTOR_GROUPS = {
    "Нормативное соответствие": ["normative_match", "amount_normative_integrity", "amount_adequacy"],
    "Бюджет и очередь": ["budget_pressure", "queue_position"],
    "Региональная специфика": ["region_specialization", "region_direction_approval_rate", "akimat_approval_rate"],
    "Характеристики заявки": ["unit_count", "direction_approval_rate", "subsidy_type_approval_rate"],
}

GROUP_ICONS = {
    "Нормативное соответствие": "📋",
    "Бюджет и очередь": "📅",
    "Региональная специфика": "🗺️",
    "Характеристики заявки": "📌",
}

# Подробные описания для каждой метрики — для члена комиссии
FACTOR_DESCRIPTIONS = {
    "normative_match": {
        "what": "Совпадение норматива в заявке с эталонным значением из справочника (Приложение 1 к Правилам).",
        "why": "Если заявитель указал норматив, отличающийся от утверждённого, это может быть ошибка или намеренное завышение.",
        "high": "Норматив в заявке полностью совпадает с эталоном — документы оформлены корректно.",
        "medium": "Небольшое расхождение с эталоном — рекомендуется уточнить у заявителя.",
        "low": "Существенное расхождение — заявленный норматив значительно отличается от установленного.",
    },
    "amount_normative_integrity": {
        "what": "Проверка арифметической корректности: сумма заявки = норматив × количество единиц.",
        "why": "Несоответствие суммы формуле расчёта может указывать на ошибку в расчётах или некорректное оформление.",
        "high": "Сумма в заявке точно соответствует расчёту — арифметика верна.",
        "medium": "Есть незначительная погрешность расчёта — возможна ошибка округления.",
        "low": "Значительная ошибка в расчёте — сумма не соответствует формуле.",
    },
    "amount_adequacy": {
        "what": "Насколько запрашиваемая сумма адекватна для данного типа субсидии и региона.",
        "why": "Аномально высокие или низкие суммы заслуживают дополнительной проверки.",
        "high": "Сумма находится в типичном диапазоне для данного направления и региона.",
        "medium": "Сумма несколько отклоняется от типичного диапазона.",
        "low": "Сумма значительно выходит за пределы обычных значений для таких заявок.",
    },
    "budget_pressure": {
        "what": "Уровень конкуренции за бюджетные средства в момент подачи заявки.",
        "why": "Чем больше заявок уже подано к моменту обращения, тем выше конкуренция за ограниченный бюджет.",
        "high": "Заявка подана в период низкой загруженности — бюджет свободен.",
        "medium": "Средний уровень конкуренции — бюджет частично использован.",
        "low": "Высокая конкуренция за бюджет — значительная часть средств уже зарезервирована.",
    },
    "queue_position": {
        "what": "Позиция заявки в хронологической очереди подачи (более ранние получают преимущество).",
        "why": "Раннее обращение свидетельствует о подготовленности заявителя и увеличивает шансы на рассмотрение.",
        "high": "Заявка подана одной из первых — высокий приоритет в очереди.",
        "medium": "Заявка подана в середине потока — стандартная позиция.",
        "low": "Заявка подана поздно — позиция в конце очереди.",
    },
    "region_specialization": {
        "what": "Насколько заявленное направление сельского хозяйства характерно для данного региона.",
        "why": "Субсидии для профильных направлений региона одобряются чаще — это подтверждено статистикой.",
        "high": "Направление является профильным для данного региона — высокая специализация.",
        "medium": "Направление умеренно представлено в регионе.",
        "low": "Направление нетипично для данного региона.",
    },
    "region_direction_approval_rate": {
        "what": "Историческая доля одобрённых заявок по данному направлению в данном регионе.",
        "why": "Высокий процент одобрений в прошлом говорит о востребованности направления в регионе.",
        "high": "Большинство аналогичных заявок в этом регионе были одобрены ранее.",
        "medium": "Одобряемость аналогичных заявок находится на среднем уровне.",
        "low": "Аналогичные заявки в этом регионе одобрялись редко.",
    },
    "akimat_approval_rate": {
        "what": "Исторический уровень одобрения заявок, поступивших через данный акимат.",
        "why": "Разные акиматы имеют разную историю подготовки качественных заявок.",
        "high": "Акимат имеет высокий исторический показатель одобрения заявок.",
        "medium": "Средний уровень одобрения заявок данного акимата.",
        "low": "Низкий исторический показатель одобрения у данного акимата.",
    },
    "unit_count": {
        "what": "Оценка количества заявленных единиц (голов скота, гектаров и т.д.).",
        "why": "Необычно большие или малые объёмы могут указывать на нетипичную заявку.",
        "high": "Количество единиц в пределах нормы для данного типа субсидии.",
        "medium": "Объём несколько отличается от типичного.",
        "low": "Количество единиц значительно отклоняется от обычных значений.",
    },
    "direction_approval_rate": {
        "what": "Общий процент одобрения заявок по данному направлению субсидирования (по всем регионам).",
        "why": "Высокий показатель означает, что направление субсидирования в целом востребовано и одобряется.",
        "high": "Направление имеет высокий общий процент одобрения.",
        "medium": "Средний процент одобрения по направлению.",
        "low": "Направление имеет низкий процент одобрения в целом по стране.",
    },
    "subsidy_type_approval_rate": {
        "what": "Историческая одобряемость конкретного типа субсидии.",
        "why": "Некоторые типы субсидий одобряются чаще других — это отражает приоритеты государственной политики.",
        "high": "Данный тип субсидии одобряется часто.",
        "medium": "Одобряемость типа субсидии на среднем уровне.",
        "low": "Данный тип субсидии одобряется редко.",
    },
}

# Описания для ML-факторов
ML_FACTOR_DESCRIPTIONS = {
    "normative_match": "Совпадение норматива с эталонным значением из справочника.",
    "amount_normative_integrity": "Арифметическая корректность расчёта суммы заявки.",
    "amount_adequacy": "Насколько запрашиваемая сумма типична для данного вида субсидии.",
    "budget_pressure": "Уровень конкуренции за бюджет на момент подачи.",
    "queue_position": "Позиция заявки в очереди обращений.",
    "region_specialization": "Профильность направления для данного региона.",
    "region_direction_approval_rate": "Историческая одобряемость направления в регионе.",
    "akimat_approval_rate": "Исторический уровень одобрения заявок от данного акимата.",
    "unit_count": "Оценка количества заявленных единиц.",
    "direction_approval_rate": "Общий процент одобрения по направлению.",
    "subsidy_type_approval_rate": "Историческая одобряемость данного типа субсидии.",
    "rule_score": "Суммарная оценка по всем правилам нормативного соответствия.",
    "contrib_budget_pressure": "Вклад бюджетного давления в оценку по правилам.",
    "contrib_region_direction_approval_rate": "Вклад региональной одобряемости в оценку.",
    "contrib_normative_match": "Вклад нормативного совпадения в оценку.",
    "contrib_queue_position": "Вклад позиции в очереди в оценку.",
    "contrib_akimat_approval_rate": "Вклад показателя акимата в оценку.",
    "contrib_amount_adequacy": "Вклад адекватности суммы в оценку.",
    "contrib_region_specialization": "Вклад региональной специализации в оценку.",
    "contrib_amount_normative_integrity": "Вклад арифметической корректности в оценку.",
    "contrib_direction_approval_rate": "Вклад одобряемости направления в оценку.",
    "contrib_subsidy_type_approval_rate": "Вклад одобряемости типа субсидии в оценку.",
    "contrib_unit_count": "Вклад показателя количества единиц в оценку.",
    "region_encoded": "Числовой код региона для модели.",
    "direction_encoded": "Числовой код направления субсидирования.",
    "submit_month": "Месяц подачи заявки (влияет на сезонность и бюджет).",
    "log_amount": "Логарифм запрашиваемой суммы (нормализация).",
    "amount_per_unit": "Сумма субсидии в расчёте на одну единицу.",
}


def _score_verdict(score: float) -> str:
    """Human-readable verdict based on score."""
    if score >= 80:
        return "Заявка набрала **высокий балл** — документы соответствуют требованиям, показатели выше среднего."
    elif score >= 70:
        return "Заявка демонстрирует **хорошие показатели** — большинство критериев пройдены успешно."
    elif score >= 55:
        return "Заявка набрала **средний балл** — есть потенциал для улучшения по ряду критериев."
    elif score >= 45:
        return "Заявка получила **пограничную оценку** — рекомендуется обратить внимание на слабые стороны."
    else:
        return "Заявка получила **низкий балл** — выявлены существенные расхождения с требованиями."


def _group_narrative(group_name: str, group_pct: float) -> str:
    """Narrative explanation for a group of factors."""
    if group_pct >= 0.7:
        strength = "хорошо"
    elif group_pct >= 0.4:
        strength = "на среднем уровне"
    else:
        strength = "ниже ожиданий"

    narratives = {
        "Нормативное соответствие": {
            "хорошо": "Заявленные нормативы и суммы полностью соответствуют установленным правилам.",
            "на среднем уровне": "Часть нормативных показателей требует уточнения — есть незначительные расхождения.",
            "ниже ожиданий": "Обнаружены существенные расхождения между заявленными и эталонными нормативами.",
        },
        "Бюджет и очередь": {
            "хорошо": "Заявка подана вовремя, бюджетное давление в рамках нормы.",
            "на среднем уровне": "Заявка подана в период средней загруженности, возможна конкуренция за бюджет.",
            "ниже ожиданий": "Высокая конкуренция за бюджет или поздняя подача — это снижает шансы.",
        },
        "Региональная специфика": {
            "хорошо": "Направление хорошо развито в регионе, исторические показатели одобрения высокие.",
            "на среднем уровне": "Направление умеренно представлено в регионе, одобряемость на среднем уровне.",
            "ниже ожиданий": "Направление нетипично для региона или исторические показатели одобрения низкие.",
        },
        "Характеристики заявки": {
            "хорошо": "Параметры заявки (тип, направление, объём) характерны для успешных обращений.",
            "на среднем уровне": "Параметры заявки в целом типичны, но не выделяются среди одобренных.",
            "ниже ожиданий": "Параметры заявки нехарактерны для обычно одобряемых обращений.",
        },
    }

    return narratives.get(group_name, {}).get(strength, f"Оценка группы: {strength}.")


def _factor_level_from_value(name: str, value: float) -> str:
    """Determine level text for a factor based on its value."""
    desc = FACTOR_DESCRIPTIONS.get(name, {})
    if value >= 0.7:
        return desc.get("high", "Высокий показатель.")
    elif value >= 0.4:
        return desc.get("medium", "Средний показатель.")
    else:
        return desc.get("low", "Низкий показатель.")


def render_details(detail: dict):
    risk_class = {
        "низкий": "risk-low", "средний": "risk-medium", "высокий": "risk-high",
    }.get(detail["risk_level"].lower(), "risk-medium")

    score = detail["score"]
    rule_score = detail.get("rule_score")
    ml_score = detail.get("ml_score")
    ml_probability = detail.get("ml_probability")
    has_ml = ml_score is not None

    # --- карточка скора + информация о заявке ---
    st.markdown(f"""
    <div class="detail-header">
        <div class="detail-score-block">
            <p class="detail-score-label">{"Итоговый балл" if has_ml else "Балл"}</p>
            <p class="detail-score">{score:.1f}</p>
            <span class="risk-badge {risk_class}">{detail['risk_level']} риск</span>
        </div>
        <div class="detail-info-block">
            <div class="detail-info-row">
                <div class="detail-info-item">
                    <span class="detail-info-label">Регион</span>
                    <span class="detail-info-value">{detail.get("region", "—")}</span>
                </div>
                <div class="detail-info-item">
                    <span class="detail-info-label">Сумма</span>
                    <span class="detail-info-value">{detail.get("amount", 0):,.0f} ₸</span>
                </div>
                <div class="detail-info-item">
                    <span class="detail-info-label">Статус</span>
                    <span class="detail-info-value">{detail.get("status", "—")}</span>
                </div>
            </div>
            <div class="detail-info-row">
                <div class="detail-info-item wide">
                    <span class="detail-info-label">Направление</span>
                    <span class="detail-info-value">{detail.get("direction", "—")}</span>
                </div>
            </div>
            <div class="detail-info-row">
                <div class="detail-info-item wide">
                    <span class="detail-info-label">Тип субсидии</span>
                    <span class="detail-info-value">{detail.get("subsidy_type", "—")}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- итоговые оценки — блок с ML и Rule Score ---
    if has_ml:
        rule_color = "#4cc9f0" if rule_score and rule_score >= 60 else "#e9c46a" if rule_score and rule_score >= 40 else "#e63946"
        ml_color = "#4cc9f0" if ml_score and ml_score >= 60 else "#e9c46a" if ml_score and ml_score >= 40 else "#e63946"
        prob_pct = f"{ml_probability:.0%}" if ml_probability else "—"
        prob_color = "#2d6a4f" if ml_probability and ml_probability >= 0.7 else "#e9c46a" if ml_probability and ml_probability >= 0.4 else "#e63946"

        st.markdown(f"""
        <div class="scores-breakdown">
            <div class="score-card">
                <div class="score-card-label">Оценка по правилам</div>
                <div class="score-card-value" style="color: {rule_color}">{rule_score:.1f}<span class="score-card-max">/100</span></div>
                <div class="score-card-desc">Соответствие нормативным критериям и статистическим показателям</div>
            </div>
            <div class="score-card">
                <div class="score-card-label">Оценка ML-модели</div>
                <div class="score-card-value" style="color: {ml_color}">{ml_score:.1f}<span class="score-card-max">/100</span></div>
                <div class="score-card-desc">Прогноз нейросетевой модели на основе анализа исторических данных</div>
            </div>
            <div class="score-card">
                <div class="score-card-label">Вероятность одобрения</div>
                <div class="score-card-value" style="color: {prob_color}">{prob_pct}</div>
                <div class="score-card-desc">Оценка итоговой силы заявки по модели машинного обучения</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- advisory ---
    history_rec = detail.get("history_recommendation")
    history_note = detail.get("history_note")
    if history_rec:
        advisory_icon = {"история поддерживает": "✅", "история предупреждает": "⚠️"}.get(
            history_rec, "ℹ️"
        )
        st.info(f"{advisory_icon} **{history_rec}** — {history_note or ''}")

    # --- норматив: заявленный vs эталонный ---
    ref_norm = detail.get("ref_normative")
    app_norm = detail.get("normative")
    if ref_norm or app_norm:
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            st.metric("Норматив (эталон)", f"{ref_norm:,.0f} ₸" if ref_norm else "—")
        with col_n2:
            st.metric("Норматив (заявка)", f"{app_norm:,.0f} ₸" if app_norm else "—")
        with col_n3:
            if ref_norm and app_norm:
                match = "Совпадает" if abs(ref_norm - app_norm) < 0.01 else "Не совпадает"
                st.metric("Соответствие", match)
            else:
                st.metric("Соответствие", "—")

    # --- объяснение по группам факторов с двумя вкладками ---
    st.markdown('<p class="section-header">📝 Почему такой балл?</p>', unsafe_allow_html=True)

    # Общий вердикт
    st.markdown(f"""
    <div class="verdict-card">
        <div class="verdict-text">{_score_verdict(score)}</div>
    </div>
    """, unsafe_allow_html=True)

    factors = detail.get("factors", [])
    ml_factors = detail.get("ml_factors", [])

    # Вкладки: Оценка по правилам | Оценка ML-модели
    if has_ml and ml_factors:
        tab_rule, tab_ml = st.tabs(["📋 Оценка по правилам", "🤖 Оценка ML-модели"])
    else:
        tab_rule = st.container()
        tab_ml = None

    # ===== TAB 1: Rule-based =====
    with tab_rule:
        if factors:
            # график факторов
            factors_df = pd.DataFrame(factors)
            factors_df["max_contribution"] = factors_df["name"].map(
                lambda n: WEIGHTS.get(n, 0) * 100
            )
            factors_df = factors_df.sort_values("contribution", ascending=True)

            def get_rule_color(row):
                max_c = row["max_contribution"]
                if max_c <= 0: return "#e9c46a"
                pct = row["contribution"] / max_c
                if pct >= 0.7: return "#2d6a4f" # green (высокий)
                elif pct >= 0.4: return "#e9c46a" # yellow (средний)
                else: return "#e63946" # red (низкий)

            colors = factors_df.apply(get_rule_color, axis=1).tolist()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=factors_df["label"],
                x=factors_df["max_contribution"],
                orientation="h",
                name="Макс. вклад (вес)",
                marker_color="rgba(100, 100, 140, 0.3)",
                hovertemplate="%{y}: вес %{x:.0f}%<extra>Максимум</extra>",
            ))

            fig.add_trace(go.Bar(
                y=factors_df["label"],
                x=factors_df["contribution"],
                orientation="h",
                name="Фактический вклад",
                marker_color=colors,
                customdata=factors_df[["value", "name"]].values if "value" in factors_df.columns else factors_df[["contribution", "name"]].values,
                hovertemplate=(
                    "%{y}: %{x:.1f} баллов<extra>Факт</extra>"
                ),
            ))

            fig.update_layout(
                barmode="overlay",
                height=max(450, len(factors_df) * 40),
                margin=dict(l=280, r=20, t=10, b=30),
                yaxis_title="",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, width="stretch", use_container_width=True)

        # предупреждение о дефолтных значениях
        default_factors = [
            f for f in factors
            if f["name"] in DEFAULT_VALUE_FACTORS
            and f.get("value") is not None
            and abs(float(f["value"]) - 0.5) < 0.01
        ]
        if default_factors:
            names = ", ".join(f["label"] for f in default_factors)
            st.warning(
                f"Метрики **{names}** имеют значение 0.5 (по умолчанию). "
                "Это означает, что для данных метрик не было достаточно контекста — "
                "фактические значения могут отличаться."
            )

        # группы факторов с нарративами
        factors_by_name = {f["name"]: f for f in factors}

        group_items = list(FACTOR_GROUPS.items())
        for row_start in range(0, len(group_items), 2):
            cols = st.columns(2)
            for col_idx, (group_name, group_factor_names) in enumerate(group_items[row_start:row_start + 2]):
                with cols[col_idx]:
                    group_total = 0.0
                    group_max = 0.0
                    group_factors_data = []

                    for fn in group_factor_names:
                        w = WEIGHTS.get(fn, 0)
                        group_max += w * 100
                        fd = factors_by_name.get(fn)
                        if fd:
                            group_total += fd["contribution"]
                            group_factors_data.append(fd)

                    group_pct = group_total / group_max if group_max > 0 else 0

                    if group_pct >= 0.7:
                        level_class = "high"
                    elif group_pct >= 0.4:
                        level_class = "medium"
                    else:
                        level_class = "low"

                    icon = GROUP_ICONS.get(group_name, "📊")
                    narrative = _group_narrative(group_name, group_pct)

                    st.markdown(f"""
                    <div class="group-card {level_class}">
                        <div class="group-card-header">
                            <span>{icon} {group_name}</span>
                            <span class="group-card-score">{group_total:.1f}/{group_max:.0f}</span>
                        </div>
                        <div class="group-card-bar-bg">
                            <div class="group-card-bar-fill {level_class}" style="width: {group_pct * 100:.0f}%"></div>
                        </div>
                        <div class="group-card-narrative">{narrative}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for fd in group_factors_data:
                        w = WEIGHTS.get(fd["name"], 0) * 100
                        fill_pct = (fd["contribution"] / w * 100) if w > 0 else 0
                        level_color = LEVEL_COLORS.get(fd["level"], "#888")
                        raw_value = fd.get("value")
                        value_pct = f"{float(raw_value):.0%}" if raw_value is not None else "—"

                        # Описание и пояснение
                        desc = FACTOR_DESCRIPTIONS.get(fd["name"], {})
                        what_text = desc.get("what", "")
                        level_text = _factor_level_from_value(fd["name"], float(raw_value) if raw_value is not None else 0.5)

                        st.markdown(f"""
                        <div class="factor-detail-card">
                            <div class="factor-row-header">
                                <span class="factor-row-label">{fd['label']}</span>
                                <span class="factor-row-value">+{fd['contribution']:.1f} <span class="factor-row-max">/ {w:.0f}</span></span>
                            </div>
                            <div class="factor-row-bar-bg">
                                <div class="factor-row-bar-fill" style="width: {fill_pct:.0f}%; background: {level_color}"></div>
                            </div>
                            <div class="factor-detail-assessment">{level_text}</div>
                            <div class="factor-detail-what">{what_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # ===== TAB 2: ML-model =====
    if tab_ml is not None:
        with tab_ml:
            if ml_factors:
                st.markdown("""
                <div class="ml-intro-card">
                    <div class="ml-intro-text">
                        Нейросетевая модель анализирует заявку целиком и оценивает, насколько она похожа
                        на исторически одобрённые заявки. Ниже показаны факторы, которые <strong>больше всего повлияли</strong>
                        на решение модели — как в положительную, так и в отрицательную сторону.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Разделяем на положительные и отрицательные
                positive_ml = [f for f in ml_factors if f["contribution"] > 0]
                negative_ml = [f for f in ml_factors if f["contribution"] < 0]
                neutral_ml = [f for f in ml_factors if f["contribution"] == 0]

                # График по ML
                ml_df = pd.DataFrame(ml_factors)
                ml_df = ml_df[ml_df["contribution"] != 0].copy()
                if not ml_df.empty:
                    # Сортируем как в первом графике (от меньшего к большему по абсолютному значению или просто)
                    ml_df["abs_contrib"] = ml_df["contribution"].abs()
                    ml_df = ml_df.sort_values("abs_contrib", ascending=True)
                    
                    def get_ml_color(c):
                        if c >= 1.0: return "#2d6a4f"     # высокий положительный (зеленый)
                        elif c > 0: return "#e9c46a"      # средний/небольшой (желтый)
                        else: return "#e63946"            # отрицательный (красный)
                        
                    colors_ml = ml_df["contribution"].apply(get_ml_color).tolist()
                    
                    fig_ml = go.Figure()

                    # Псевдо-фон (визуально объединяет стиль с первой вкладкой)
                    max_abs = ml_df["abs_contrib"].max()
                    bg_x = [max_abs * 1.1 if c > 0 else -max_abs * 1.1 for c in ml_df["contribution"]]
                    
                    fig_ml.add_trace(go.Bar(
                        y=ml_df["label"],
                        x=bg_x,
                        orientation="h",
                        name="Возможный разброс",
                        marker_color="rgba(100, 100, 140, 0.15)",
                        hovertemplate="<extra></extra>",
                        hoverinfo="skip"
                    ))

                    fig_ml.add_trace(go.Bar(
                        y=ml_df["label"],
                        x=ml_df["contribution"],
                        orientation="h",
                        name="Фактический вклад",
                        marker_color=colors_ml,
                        hovertemplate="%{y}: %{x:+.1f} баллов<extra>Влияние</extra>"
                    ))
                    
                    fig_ml.update_layout(
                        barmode="overlay",
                        title="Вклад факторов по оценке ML-модели",
                        height=max(450, len(ml_df) * 40),
                        margin=dict(l=280, r=20, t=40, b=30),
                        xaxis_title="Влияние (баллы)",
                        yaxis_title="",
                        showlegend=False,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_ml, width="stretch", use_container_width=True)

                if positive_ml:
                    st.markdown('<p class="ml-section-label">✅ Положительно повлияло на оценку ML</p>', unsafe_allow_html=True)
                    for mf in positive_ml:
                        impact = mf["contribution"]
                        desc_text = ML_FACTOR_DESCRIPTIONS.get(mf["name"], "")
                        impact_bar_width = min(abs(impact) * 5, 100)

                        st.markdown(f"""
                        <div class="ml-factor-card positive">
                            <div class="ml-factor-header">
                                <span class="ml-factor-label">{mf['label']}</span>
                                <span class="ml-factor-impact positive">+{impact:.1f} б.</span>
                            </div>
                            <div class="ml-factor-bar-bg">
                                <div class="ml-factor-bar-fill positive" style="width: {impact_bar_width:.0f}%"></div>
                            </div>
                            <div class="ml-factor-desc">{desc_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if negative_ml:
                    st.markdown('<p class="ml-section-label">⚠️ Отрицательно повлияло на оценку ML</p>', unsafe_allow_html=True)
                    for mf in negative_ml:
                        impact = mf["contribution"]
                        desc_text = ML_FACTOR_DESCRIPTIONS.get(mf["name"], "")
                        impact_bar_width = min(abs(impact) * 5, 100)

                        st.markdown(f"""
                        <div class="ml-factor-card negative">
                            <div class="ml-factor-header">
                                <span class="ml-factor-label">{mf['label']}</span>
                                <span class="ml-factor-impact negative">{impact:.1f} б.</span>
                            </div>
                            <div class="ml-factor-bar-bg">
                                <div class="ml-factor-bar-fill negative" style="width: {impact_bar_width:.0f}%"></div>
                            </div>
                            <div class="ml-factor-desc">{desc_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if neutral_ml:
                    with st.expander(f"Нейтральные факторы ({len(neutral_ml)})"):
                        for mf in neutral_ml:
                            desc_text = ML_FACTOR_DESCRIPTIONS.get(mf["name"], "")
                            st.markdown(f"**{mf['label']}** — {desc_text}" if desc_text else f"**{mf['label']}**")

            else:
                st.info("ML-модель не предоставила детализацию по факторам для данной заявки.")

    # --- подробное текстовое заключение для комиссии ---
    st.markdown('<p class="section-header">📜 Заключение для комиссии</p>', unsafe_allow_html=True)
    st.info("Этот раздел написан простым языком и представляет выжимку сильных и слабых сторон заявки.")
    
    st.markdown(f"**Итоговый вердикт системы:** {_score_verdict(score)}")
    
    col_str, col_weak = st.columns(2)
    
    with col_str:
        st.markdown("#### ✅ Сильные стороны")
        has_strengths = False
        for group_name, group_factor_names in FACTOR_GROUPS.items():
            group_total = sum(f.get("contribution", 0) for f in factors if f["name"] in group_factor_names)
            group_max = sum(WEIGHTS.get(fn, 0) * 100 for fn in group_factor_names)
            group_pct = group_total / group_max if group_max > 0 else 0
            if group_pct >= 0.7:
                has_strengths = True
                narrative = _group_narrative(group_name, group_pct)
                st.markdown(f"**{group_name}**:<br>{narrative}", unsafe_allow_html=True)
        if not has_strengths:
            st.markdown("— *Явных сильных сторон по показателям не выявлено*")

    with col_weak:
        st.markdown("#### ⚠️ Зоны риска")
        has_weaknesses = False
        for group_name, group_factor_names in FACTOR_GROUPS.items():
            group_total = sum(f.get("contribution", 0) for f in factors if f["name"] in group_factor_names)
            group_max = sum(WEIGHTS.get(fn, 0) * 100 for fn in group_factor_names)
            group_pct = group_total / group_max if group_max > 0 else 0
            if group_pct < 0.7:
                has_weaknesses = True
                narrative = _group_narrative(group_name, group_pct)
                st.markdown(f"**{group_name}**:<br>{narrative}", unsafe_allow_html=True)
        if not has_weaknesses:
            st.markdown("— *Значительных зон риска не выявлено*")

    explanation = detail.get("explanation", [])
    if explanation:
        with st.expander("🛠️ Технический лог скоринга (для аналитиков)", expanded=False):
            for line in explanation:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if line_stripped.startswith("⚠") or line_stripped.startswith("✗"):
                    st.error(line_stripped)
                elif line_stripped.startswith("✅") or line_stripped.startswith("✓"):
                    st.success(line_stripped)
                elif line_stripped.startswith("ℹ") or line_stripped.startswith("Система"):
                    st.info(line_stripped)
                elif line_stripped.startswith("  •") or line_stripped.startswith("•"):
                    st.markdown(f"  {line_stripped}")
                elif line_stripped.endswith(":"):
                    st.markdown(f"**{line_stripped}**")
                else:
                    st.markdown(line_stripped)


# --- page ---

st.set_page_config(page_title="Детали заявки", page_icon="🔎", layout="wide")
st.markdown('<p class="main-title">🔎 Детали заявки</p>', unsafe_allow_html=True)

result = page_setup("Детали")
if result:
    filters, stats, rank_data = result
    applications = rank_data["applications"]
    app_numbers = [a["app_number"] for a in applications[:50]]

    tab_list, tab_search = st.tabs(["Из шортлиста", "Поиск по номеру"])

    with tab_list:
        if app_numbers:
            selected_app = st.selectbox("Выберите заявку", app_numbers, key="detail_select")
            detail = get_explanation(selected_app)
            render_details(detail)
        else:
            st.info("Нет заявок в текущем фильтре.")

    with tab_search:
        search_id = st.text_input("Введите номер заявки", key="detail_search")
        if search_id:
            search_id = search_id.strip()
            try:
                detail = get_explanation(search_id)
                render_details(detail)
            except Exception:
                st.error(f"Заявка **{search_id}** не найдена.")
