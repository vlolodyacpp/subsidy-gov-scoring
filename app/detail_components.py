import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from shared import WEIGHTS, DEFAULT_VALUE_FACTORS, PLOTLY_LAYOUT

FACTOR_GROUPS = {
    "Нормативное соответствие": ["normative_match", "amount_normative_integrity", "amount_adequacy"],
    "Бюджет и очередь": ["budget_pressure", "queue_position"],
    "Региональная специфика": ["region_specialization", "region_direction_approval_rate", "akimat_approval_rate"],
    "Характеристики заявки": ["unit_count", "direction_approval_rate", "subsidy_type_approval_rate"],
    "Условия содержания": ["pasture_compliance", "mortality_compliance", "grazing_utilization"],
    "Регуляторная сложность": ["criteria_complexity", "direction_risk", "regional_pasture_capacity"],
}

GROUP_ICONS = {
    "Нормативное соответствие": "📋",
    "Бюджет и очередь": "📅",
    "Региональная специфика": "🗺️",
    "Характеристики заявки": "📌",
    "Условия содержания": "🐄",
    "Регуляторная сложность": "⚖️",
}

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
    "pasture_compliance": {
        "what": "Соответствие фактической нагрузки на пастбища установленным нормам для данного региона и вида животных.",
        "why": "Перегрузка пастбищ снижает продуктивность и может привести к деградации земель.",
        "high": "Нагрузка на пастбища в пределах нормы — условия содержания соответствуют требованиям.",
        "medium": "Нагрузка на пастбища приближается к предельным значениям.",
        "low": "Нагрузка на пастбища значительно превышает допустимые нормы.",
    },
    "mortality_compliance": {
        "what": "Соответствие уровня падежа животных допустимым нормам по данному направлению.",
        "why": "Высокий падёж может указывать на проблемы с ветеринарным обеспечением или условиями содержания.",
        "high": "Уровень падежа значительно ниже предельных норм.",
        "medium": "Уровень падежа в пределах допустимого, но требует внимания.",
        "low": "Уровень падежа превышает допустимые нормы — высокий риск.",
    },
    "grazing_utilization": {
        "what": "Степень использования пастбищного сезона (доля фактических дней выпаса от нормативного периода).",
        "why": "Эффективное использование пастбищного сезона — показатель рациональности хозяйствования.",
        "high": "Пастбищный сезон используется эффективно.",
        "medium": "Пастбищный сезон используется частично.",
        "low": "Пастбищный сезон используется неэффективно.",
    },
    "criteria_complexity": {
        "what": "Оценка простоты регуляторной проверки для данного направления (обратная сложность критериев).",
        "why": "Чем больше регуляторных критериев — тем выше вероятность несоответствия по одному из них.",
        "high": "Направление имеет относительно простые критерии проверки.",
        "medium": "Средний уровень регуляторной сложности.",
        "low": "Направление имеет множество критериев — повышен риск несоответствия.",
    },
    "direction_risk": {
        "what": "Биологическая безопасность направления: оценка на основе среднего уровня падежа по отрасли.",
        "why": "Направления с высоким средним падежом более рискованны для субсидирования.",
        "high": "Направление имеет низкий средний уровень падежа — биологически безопасно.",
        "medium": "Средний уровень биологического риска по направлению.",
        "low": "Направление имеет высокий средний уровень падежа — повышенный риск.",
    },
    "regional_pasture_capacity": {
        "what": "Ресурсная ёмкость пастбищ региона: насколько регион обеспечен пастбищными угодьями.",
        "why": "Регионы с дефицитом пастбищ имеют повышенный риск перегрузки при субсидировании.",
        "high": "Регион хорошо обеспечен пастбищными ресурсами.",
        "medium": "Средняя обеспеченность пастбищами.",
        "low": "Регион испытывает дефицит пастбищных ресурсов.",
    },
}

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
    "pasture_compliance": "Соответствие нагрузки на пастбища нормам региона.",
    "mortality_compliance": "Соответствие уровня падежа допустимым нормам.",
    "grazing_utilization": "Эффективность использования пастбищного сезона.",
    "criteria_complexity": "Простота регуляторной проверки для направления.",
    "direction_risk": "Биологическая безопасность направления по уровню падежа.",
    "regional_pasture_capacity": "Обеспеченность региона пастбищными ресурсами.",
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
    "normative_log": "Логарифм нормативного значения — нормализация для модели.",
    "normative_original_match": "Совпадение исходного норматива заявки с эталоном.",
    "normative_reference_gap": "Отклонение заявленного норматива от эталонного значения.",
    "normative_reference_typicality": "Насколько норматив типичен для данного типа субсидии.",
    "unit_count_log": "Логарифм количества заявленных единиц.",
    "unit_count_original_log": "Логарифм исходного количества единиц до нормализации.",
    "submit_month_sin": "Сезонная компонента месяца подачи (синусоидальное кодирование).",
    "region_approval_rate": "Общая доля одобренных заявок в данном регионе.",
    "region_direction_lift": "Прирост одобряемости направления в регионе относительно среднего.",
    "akimat_lift": "Прирост одобряемости акимата относительно среднего.",
    "direction_history_count_log": "Объём истории заявок по данному направлению.",
    "subsidy_type_history_count_log": "Объём истории заявок по данному типу субсидии.",
    "region_direction_history_count_log": "Объём истории заявок данного направления в регионе.",
    "akimat_history_count_log": "Объём истории заявок от данного акимата.",
    "amount_to_normative_ratio": "Отношение запрашиваемой суммы к нормативу.",
    "amount_to_type_median_ratio": "Отношение суммы к медиане по типу субсидии.",
    "adequacy_x_direction_rate": "Взаимодействие адекватности суммы и одобряемости направления.",
    "adequacy_x_budget_pressure": "Взаимодействие адекватности суммы и бюджетного давления.",
    "criteria_complexity_x_subsidy_type_rate": "Взаимодействие регуляторной сложности и одобряемости типа субсидии.",
    "direction_risk_x_mortality_compliance": "Взаимодействие биориска направления и уровня падежа.",
    "rule_score_x_budget_pressure": "Взаимодействие оценки по правилам и бюджетного давления.",
    "amount_log_x_rule_score": "Взаимодействие размера суммы и оценки по правилам.",
    "submit_month_cos": "Сезонная компонента месяца подачи (косинусоидальное кодирование).",
    "amount_log": "Логарифм запрашиваемой суммы (нормализация для модели).",
    "contrib_pasture_compliance": "Вклад нагрузки на пастбища в оценку по правилам.",
    "contrib_mortality_compliance": "Вклад уровня падежа в оценку по правилам.",
    "contrib_grazing_utilization": "Вклад использования пастбищ в оценку по правилам.",
    "contrib_criteria_complexity": "Вклад регуляторной сложности в оценку по правилам.",
    "contrib_direction_risk": "Вклад биориска направления в оценку по правилам.",
    "contrib_regional_pasture_capacity": "Вклад ёмкости пастбищ в оценку по правилам.",
    "rule_score_feature": "Суммарная оценка заявки по правилам, используемая как признак модели.",
    "subsidy_type": "Тип субсидии (категориальный признак).",
}

ML_FACTOR_LABELS = {
    "normative_match": "Соответствие нормативу",
    "amount_normative_integrity": "Корректность суммы",
    "amount_adequacy": "Адекватность суммы",
    "budget_pressure": "Бюджетное давление",
    "queue_position": "Позиция в очереди",
    "region_specialization": "Профильность региона",
    "region_direction_approval_rate": "Одобряемость направления в регионе",
    "akimat_approval_rate": "Одобрение акимата",
    "unit_count": "Кол-во единиц",
    "direction_approval_rate": "Одобряемость направления",
    "subsidy_type_approval_rate": "Одобряемость типа субсидии",
    "pasture_compliance": "Нагрузка на пастбища",
    "mortality_compliance": "Уровень падежа",
    "grazing_utilization": "Использование пастбищ",
    "criteria_complexity": "Регуляторная сложность",
    "direction_risk": "Биологический риск направления",
    "regional_pasture_capacity": "Ресурсная ёмкость пастбищ",
    "rule_score": "Оценка по правилам",
    "contrib_budget_pressure": "Вклад бюджетного давления",
    "contrib_region_direction_approval_rate": "Вклад одобряемости региона",
    "contrib_normative_match": "Вклад соответствия нормативу",
    "contrib_queue_position": "Вклад позиции в очереди",
    "contrib_akimat_approval_rate": "Вклад одобрения акимата",
    "contrib_amount_adequacy": "Вклад адекватности суммы",
    "contrib_region_specialization": "Вклад специализации региона",
    "contrib_amount_normative_integrity": "Вклад корректности расчёта",
    "contrib_direction_approval_rate": "Вклад одобряемости направления",
    "contrib_subsidy_type_approval_rate": "Вклад одобряемости типа субсидии",
    "contrib_unit_count": "Вклад количества единиц",
    "region_encoded": "Региональный код",
    "direction_encoded": "Код направления",
    "submit_month": "Месяц подачи",
    "log_amount": "Размер суммы (лог.)",
    "amount_per_unit": "Сумма на единицу",
    "normative_log": "Норматив (лог.)",
    "normative_original_match": "Совпадение исходного норматива",
    "normative_reference_gap": "Отклонение от эталонного норматива",
    "normative_reference_typicality": "Типичность норматива",
    "unit_count_log": "Кол-во единиц (лог.)",
    "unit_count_original_log": "Исходное кол-во единиц (лог.)",
    "submit_month_sin": "Сезонность подачи",
    "region_approval_rate": "Одобряемость в регионе",
    "region_direction_lift": "Эффект направления в регионе",
    "akimat_lift": "Эффект акимата",
    "direction_history_count_log": "История заявок по направлению",
    "subsidy_type_history_count_log": "История заявок по типу субсидии",
    "region_direction_history_count_log": "История заявок направления в регионе",
    "akimat_history_count_log": "История заявок акимата",
    "amount_to_normative_ratio": "Отношение суммы к нормативу",
    "amount_to_type_median_ratio": "Отношение суммы к медиане типа",
    "adequacy_x_direction_rate": "Адекватность × одобряемость направления",
    "adequacy_x_budget_pressure": "Адекватность × бюджетное давление",
    "criteria_complexity_x_subsidy_type_rate": "Рег. сложность × одобряемость типа",
    "direction_risk_x_mortality_compliance": "Биориск × уровень падежа",
    "rule_score_x_budget_pressure": "Оценка по правилам × бюджетное давление",
    "amount_log_x_rule_score": "Размер суммы × оценка по правилам",
    "rule_score_feature": "Оценка по правилам (фича)",
    "subsidy_type": "Тип субсидии",
    "submit_month_cos": "Сезонность подачи (косинус)",
    "amount_log": "Размер суммы (лог.)",
    "contrib_pasture_compliance": "Вклад нагрузки на пастбища",
    "contrib_mortality_compliance": "Вклад уровня падежа",
    "contrib_grazing_utilization": "Вклад использования пастбищ",
    "contrib_criteria_complexity": "Вклад регуляторной сложности",
    "contrib_direction_risk": "Вклад биориска направления",
    "contrib_regional_pasture_capacity": "Вклад ёмкости пастбищ",
}


def _ml_label(mf: dict) -> str:
    return ML_FACTOR_LABELS.get(mf["name"], mf.get("label", mf["name"]))


def _score_verdict(score: float) -> str:
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
        "Условия содержания": {
            "хорошо": "Пастбищная нагрузка, падёж и использование сезона — в пределах нормы.",
            "на среднем уровне": "Часть показателей условий содержания требует внимания.",
            "ниже ожиданий": "Показатели условий содержания существенно отклоняются от нормативов.",
        },
        "Регуляторная сложность": {
            "хорошо": "Направление имеет умеренные регуляторные требования и низкий биологический риск.",
            "на среднем уровне": "Средний уровень регуляторной и биологической нагрузки.",
            "ниже ожиданий": "Направление сопряжено со сложными критериями или повышенным биологическим риском.",
        },
    }

    return narratives.get(group_name, {}).get(strength, f"Оценка группы: {strength}.")


def _factor_level_from_value(name: str, value: float) -> str:
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

    score_color = "#2d6a4f" if score >= 60 else "#e9c46a" if score >= 40 else "#e63946"
    st.markdown(f"""
    <div class="scores-breakdown">
        <div class="score-card" style="max-width:180px;">
            <div class="score-card-label">Итоговый балл</div>
            <div class="score-card-value" style="color:{score_color}">{score:.1f}</div>
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

    decision_positive = detail.get("decision_predicted_positive")
    decision_threshold = detail.get("decision_threshold")

    if has_ml:
        rule_color = "#4cc9f0" if rule_score and rule_score >= 60 else "#e9c46a" if rule_score and rule_score >= 40 else "#e63946"
        ml_color = "#4cc9f0" if ml_score and ml_score >= 60 else "#e9c46a" if ml_score and ml_score >= 40 else "#e63946"
        prob_pct = f"{ml_probability:.0%}" if ml_probability else "—"
        prob_color = "#2d6a4f" if ml_probability and ml_probability >= 0.7 else "#e9c46a" if ml_probability and ml_probability >= 0.4 else "#e63946"

        if decision_positive is True:
            decision_badge = '<span class="risk-badge risk-low">Рекомендована</span>'
        elif decision_positive is False:
            decision_badge = '<span class="risk-badge risk-high">Не рекомендована</span>'
        else:
            decision_badge = ''

        threshold_text = f"порог: {decision_threshold / 100:.0%}" if decision_threshold is not None else ""

        st.markdown(f"""
        <div class="scores-breakdown">
            <div class="score-card">
                <div class="score-card-label">Оценка по правилам</div>
                <div class="score-card-value" style="color: {rule_color}">{rule_score:.1f}<span class="score-card-max">/100</span></div>
                <div class="score-card-desc">Нормативные критерии и статистика</div>
            </div>
            <div class="score-card">
                <div class="score-card-label">Оценка ML-модели</div>
                <div class="score-card-value" style="color: {ml_color}">{ml_score:.1f}<span class="score-card-max">/100</span></div>
                <div class="score-card-desc">Прогноз на основе исторических данных</div>
            </div>
            <div class="score-card">
                <div class="score-card-label">Сила заявки</div>
                <div class="score-card-value" style="color: {prob_color}">{prob_pct}</div>
                <div class="score-card-desc">{decision_badge} {threshold_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    history_rec = detail.get("history_recommendation")
    history_note = detail.get("history_note")
    if history_rec:
        advisory_icon = {"история поддерживает": "✅", "история предупреждает": "⚠️"}.get(
            history_rec, "ℹ️"
        )
        st.info(f"{advisory_icon} **{history_rec}** — {history_note or ''}")

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

    st.markdown('<p class="section-header">Из чего складывается оценка</p>', unsafe_allow_html=True)

    factors = detail.get("factors", [])
    ml_factors = detail.get("ml_factors", [])

    if has_ml and ml_factors:
        tab_rule, tab_ml = st.tabs(["Оценка по нормативным критериям", "Оценка на основе исторических данных"])
    else:
        tab_rule = st.container()
        tab_ml = None

    with tab_rule:
        if factors:
            factors_df = pd.DataFrame(factors)
            factors_df["max_contribution"] = factors_df["name"].map(
                lambda n: WEIGHTS.get(n, 0) * 100
            )
            factors_df = factors_df.sort_values("contribution", ascending=True)

            def get_rule_color(row):
                max_c = row["max_contribution"]
                if max_c <= 0: return "#e9c46a"
                pct = row["contribution"] / max_c
                if pct >= 0.7: return "#2d6a4f"
                elif pct >= 0.4: return "#e9c46a"
                else: return "#e63946"

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
                hovertemplate="%{y}: %{x:.1f} баллов<extra>Факт</extra>",
            ))
            fig.update_layout(
                barmode="overlay",
                height=max(450, len(factors_df) * 40),
                margin=dict(l=320, r=20, t=10, b=50),
                yaxis_title="",
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, title_text=""),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, width="stretch")

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
                        if fill_pct >= 70:
                            level_color = "#2d6a4f"
                        elif fill_pct >= 40:
                            level_color = "#e9c46a"
                        else:
                            level_color = "#e63946"
                        raw_value = fd.get("value")
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

    if tab_ml is not None:
        with tab_ml:
            if ml_factors:
                prob_pct = f"{ml_probability:.0%}" if ml_probability else "—"
                if decision_positive is True:
                    decision_text = "По результатам анализа исторических данных заявка <b>рекомендована к одобрению</b>."
                    decision_border = "#2d6a4f"
                elif decision_positive is False:
                    decision_text = "По результатам анализа исторических данных заявка <b>не рекомендована к одобрению</b>."
                    decision_border = "#e63946"
                else:
                    decision_text = "Система проанализировала заявку на основе тысяч ранее рассмотренных обращений."
                    decision_border = "#e9c46a"

                st.markdown(f"""
                <div style="background:#1a1a2e;border:1px solid #3a3a5a;border-left:4px solid {decision_border};
                            border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
                    <div style="font-size:1.05rem;color:#d0d0e0;line-height:1.6;">
                        {decision_text}
                        Оценка силы заявки: <b style="font-size:1.2rem;">{prob_pct}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                active_factors = [mf for mf in ml_factors if mf["contribution"] != 0]
                active_factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

                if active_factors:
                    st.markdown(
                        '<p style="font-size:1.1rem;font-weight:600;color:#d0d0e0;margin:1rem 0 0.5rem;">'
                        'Что повлияло на решение (от самого значимого к наименее):</p>',
                        unsafe_allow_html=True,
                    )

                    for mf in active_factors:
                        impact = mf["contribution"]
                        label = _ml_label(mf)
                        desc_text = ML_FACTOR_DESCRIPTIONS.get(mf["name"], "")

                        raw_val = mf.get("value")
                        if raw_val is not None and raw_val != "":
                            try:
                                num_val = float(raw_val)
                                if 0 <= num_val <= 1:
                                    val_display = f"{num_val:.0%}"
                                elif abs(num_val) > 1000:
                                    val_display = f"{num_val:,.0f}"
                                else:
                                    val_display = f"{num_val:.2f}"
                            except (ValueError, TypeError):
                                val_display = str(raw_val)
                        else:
                            val_display = None

                        if impact > 0:
                            level_color = "#2d6a4f"
                            arrow = "▲"
                        else:
                            level_color = "#e63946"
                            arrow = "▼"

                        sign_mf = "+" if impact > 0 else ""
                        bar_width = min(abs(impact) * 8, 100)
                        value_html = f' <span style="color:#8888aa;font-size:0.85rem;">= {val_display}</span>' if val_display else ""

                        st.markdown(f"""
                        <div class="factor-detail-card">
                            <div class="factor-row-header">
                                <span class="factor-row-label">{label}{value_html}</span>
                                <span class="factor-row-value" style="color:{level_color}">{arrow} {sign_mf}{impact:.1f} б.</span>
                            </div>
                            <div class="factor-row-bar-bg">
                                <div class="factor-row-bar-fill" style="width: {bar_width:.0f}%; background: {level_color}"></div>
                            </div>
                            <div class="factor-detail-what">{desc_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                neutral_ml = [mf for mf in ml_factors if mf["contribution"] == 0]
                if neutral_ml:
                    with st.expander(f"Показатели без влияния ({len(neutral_ml)})"):
                        for mf in neutral_ml:
                            label = _ml_label(mf)
                            desc_text = ML_FACTOR_DESCRIPTIONS.get(mf["name"], "")
                            st.markdown(f"**{label}** — {desc_text}" if desc_text else f"**{label}**")

            else:
                st.info("Нет данных для анализа на основе исторических заявок.")

    st.markdown('<p class="section-header">📜 Заключение для комиссии</p>', unsafe_allow_html=True)

    verdict = _score_verdict(score).replace("**", "")
    st.markdown(
        f'<div class="verdict-card"><div class="verdict-text">{verdict}</div></div>',
        unsafe_allow_html=True,
    )

    strengths_html = ""
    weaknesses_html = ""
    for group_name, group_factor_names in FACTOR_GROUPS.items():
        group_total = sum(f.get("contribution", 0) for f in factors if f["name"] in group_factor_names)
        group_max = sum(WEIGHTS.get(fn, 0) * 100 for fn in group_factor_names)
        group_pct = group_total / group_max if group_max > 0 else 0
        icon = GROUP_ICONS.get(group_name, "📊")
        narrative = _group_narrative(group_name, group_pct)
        pct_display = f"{group_pct:.0%}"

        if group_pct >= 0.7:
            strengths_html += (
                f'<div class="conclusion-item high">'
                f'<div class="conclusion-item-header">'
                f'<span>{icon} {group_name}</span>'
                f'<span class="conclusion-item-pct" style="color:#2d6a4f">{pct_display}</span>'
                f'</div>'
                f'<div class="conclusion-item-text">{narrative}</div>'
                f'</div>'
            )
        else:
            level_cls = "medium" if group_pct >= 0.4 else "low"
            color = "#e9c46a" if group_pct >= 0.4 else "#e63946"
            weaknesses_html += (
                f'<div class="conclusion-item {level_cls}">'
                f'<div class="conclusion-item-header">'
                f'<span>{icon} {group_name}</span>'
                f'<span class="conclusion-item-pct" style="color:{color}">{pct_display}</span>'
                f'</div>'
                f'<div class="conclusion-item-text">{narrative}</div>'
                f'</div>'
            )

    if not strengths_html:
        strengths_html = '<div class="conclusion-empty">Явных сильных сторон по показателям не выявлено</div>'
    if not weaknesses_html:
        weaknesses_html = '<div class="conclusion-empty">Значительных зон риска не выявлено</div>'

    col_str, col_weak = st.columns(2)
    with col_str:
        st.markdown(
            f'<div class="conclusion-column">'
            f'<div class="conclusion-title high">✅ Сильные стороны</div>'
            f'{strengths_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_weak:
        st.markdown(
            f'<div class="conclusion-column">'
            f'<div class="conclusion-title warning">⚠️ Зоны риска</div>'
            f'{weaknesses_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

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
