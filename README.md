# Subsidy Scoring System

**Кейс 2 — Decentrathon 5.0**  
AI/Data-driven система предварительной оценки заявок на субсидирование в животноводстве (МСХ РК).

Система ранжирует заявки по 11 факторам на основе Правил субсидирования (Приказ МСХ РК №108), объясняет каждый score и формирует shortlist для рассмотрения комиссией. **Финальное решение остаётся за человеком — система рекомендует, не решает.**

---

## Архитектура

```
Excel (subsidies.xlsx)
  └─▶ pipeline.py       — загрузка, очистка, замена нормативов
        └─▶ features.py     — feature engineering (11 признаков)
              └─▶ eligibility.py  — hard-fail проверки (дедлайн, поля, сумма)
                    └─▶ scoring.py     — rule-based score 0–100 + explainability
                          └─▶ modeling.py    — ML (neural network), калибровка, blend
                                └─▶ advisory.py    — историческая рекомендация (не влияет на score)
                                      └─▶ api.py         — FastAPI: /score /rank /explain /stats /health
                                                └─▶ app/          — Streamlit dashboard (5 страниц)
```

---

## Данные

Основной файл: `data/subsidies.xlsx` (лист `Page 1`).

| Параметр | Значение |
|----------|----------|
| Записей до очистки | 36 651 |
| Записей после очистки | 34 584 |
| Одобренных (`is_approved = 1`) | 31 675 |
| Отклонённых (`is_approved = 0`) | 2 909 |
| Регионов | 19 |
| Направлений | 9 (скотоводство, овцеводство, коневодство и др.) |
| Типов субсидий | 47 |
| Диапазон сумм | 300 — 1 230 000 000 тенге |

**Логика меток:**

| Класс | Статусы заявки | Действие |
|-------|---------------|----------|
| Positive (1) | Исполнена, Одобрена, Сформировано поручение, Получена | Включены |
| Negative (0) | Отклонена | Включены |
| Excluded | Отозвано | Исключены (инициатива заявителя) |
| Excluded | Нарушение дедлайна подачи | Исключены (по Правилам автоматически отклоняются) |

---

## Слои системы

### 1. Eligibility (жёсткие фильтры)

Заявка немедленно получает `disqualified = true` и `score = 0`, если:

- подана вне окна **20 января — 20 декабря**;
- сумма заявки отсутствует или ≤ 0;
- не заполнены обязательные поля: регион, направление, тип субсидии.

Заявки, прошедшие базовые проверки, получают статус `preliminarily_eligible` с флагом `manual_review_required = true`, так как ряд критериев (Приложение 2) не может быть проверен автоматически без интеграции с ГИСС, ИСЖ, ИБСПР, ЕАСУ, ИСЕГКН.

### 2. Rule-based score

Для допустимых заявок вычисляется score по 11 факторам, распределённым в 4 группы:

| # | Признак | Вес | Группа |
|---|---------|-----|--------|
| 1 | `normative_match` | 10% | Соответствие нормативам |
| 2 | `amount_normative_integrity` | 8% | Соответствие нормативам |
| 3 | `amount_adequacy` | 8% | Соответствие нормативам |
| 4 | `budget_pressure` | 16% | Бюджет и очередь |
| 5 | `queue_position` | 12% | Бюджет и очередь |
| 6 | `region_specialization` | 10% | Региональный контекст |
| 7 | `region_direction_approval_rate` | 12% | Региональный контекст |
| 8 | `akimat_approval_rate` | 7% | Региональный контекст |
| 9 | `unit_count` | 5% | Характеристики заявки |
| 10 | `direction_approval_rate` | 5% | Характеристики заявки |
| 11 | `subsidy_type_approval_rate` | 5% | Характеристики заявки |

**Формула:** `score = Σ(feature_i × weight_i × 100)`, где все признаки ∈ [0, 1].

Уровни риска: `низкий` (score ≥ 70), `средний` (45–69), `высокий` (< 45).

Все исторические approval rates вычисляются **только по прошлым заявкам** относительно даты текущей, исключая leakage.

### 3. ML score (neural network)

Модель обучается поверх rule-based признаков и дополнительных сигналов:

- `rule_score` и вклады `contrib_*` каждого из 11 факторов;
- нормативные и amount-based сигналы заявки;
- history-aware approval rates и исторические счётчики;
- process-context признаки `budget_pressure`, `queue_position`;
- сжатые региональные priors вместо сырого `region`.

Итоговый score строится как blend: `final_score = rule_score × w_rule + ml_score × w_ml`.  
Веса `w_rule` / `w_ml` и порог принятия решения подбираются на validation set (cost-sensitive: штраф за FN в 3× больше, чем за FP).  
Вероятности проходят калибровку — `identity`, `sigmoid` или `isotonic` выбираются автоматически.

Готовый bundle сохраняется в `models/artifacts/subsidy_model.joblib`.

Если модель не обучена, система работает в **rule-only** режиме.

### 4. Historical advisory (диагностика)

Отдельный слой, который **не влияет** на итоговый score, но возвращается API-ответом как контекст:

| Условие | Метка |
|---------|-------|
| approval rate ≥ 75%, ≥ 5 случаев | `история поддерживает` |
| approval rate ≤ 45%, ≥ 5 случаев | `история предупреждает` |
| 45–75% | `история нейтральна` |
| < 5 случаев | `недостаточно истории` |

Скоуп матчинга: сначала точное совпадение (регион, направление, тип), затем похожее (направление, тип), затем глобальный fallback.

---

## Explainability

API и дашборд возвращают:

- вклад каждого из 11 факторов с весом и значением;
- текстовые объяснения по группам факторов (высокий/средний/низкий уровень);
- ML feature effects для финального score;
- статус eligibility и флаг `manual_review_required`;
- historical advisory как отдельный блок.

---

## Структура проекта

```
subsidy-gov-scoring/
├── src/
│   ├── pipeline.py          # Загрузка xlsx, очистка, замена нормативов
│   ├── features.py          # Feature engineering (11 признаков)
│   ├── eligibility.py       # Hard-fail проверки
│   ├── scoring.py           # Rule-based score + explainability
│   ├── modeling.py          # ML: обучение, калибровка, blend, bundle
│   ├── advisory.py          # Историческая рекомендация
│   ├── normatives.py        # Работа со справочником нормативов
│   ├── normatives_dict.py   # Справочник нормативов (Приложение 1)
│   ├── schemas.py           # Pydantic-схемы запросов и ответов
│   └── api.py               # FastAPI endpoints
├── app/
│   ├── 0_🏛️_Обзор.py        # Сводная статистика, распределение рисков
│   ├── 1_📋_Шортлист.py      # Таблица ранжирования с фильтрами
│   ├── 2_🔎_Детали.py        # Карточка заявки + разбивка по факторам
│   ├── 3_⚖️_Сравнение.py     # Сравнение 2–3 заявок
│   ├── 4_📊_Аналитика.py     # Региональная аналитика
│   ├── shared.py            # Общие UI-утилиты, sidebar-фильтры
│   ├── api_client.py        # HTTP-клиент dashboard → API
│   └── style.css            # Тёмная тема
├── models/
│   ├── artifacts/           # subsidy_model.joblib (после train.py)
│   └── reports/             # training_metrics.json, test_predictions.csv
├── data/
│   └── subsidies.xlsx       # Исходный датасет
├── output/
│   └── shortlist.csv        # Shortlist после запуска main.py
├── main.py                  # CLI entry point + запуск сервера
├── train.py                 # Обучение ML-модели
├── predict.py               # Batch prediction
├── docker-compose.yml       # API (8000) + Dashboard (8501)
├── Dockerfile.api
├── Dockerfile.dashboard
└── requirements.txt
```

---

## Быстрый старт

### Вариант 1: локально

```bash
# Создать окружение
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Обучить модель
.venv/bin/python train.py
# → models/artifacts/subsidy_model.joblib
# → models/reports/training_metrics.json

# Запустить API-сервер
.venv/bin/python main.py --serve
# → http://localhost:8000/docs

# Запустить дашборд (в отдельном терминале)
.venv/bin/streamlit run app/0_🏛️_Обзор.py
# → http://localhost:8501

# Batch prediction
.venv/bin/python predict.py

# CLI shortlist (top-10 без сервера)
.venv/bin/python main.py
```

### Вариант 2: Docker Compose

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

> Перед запуском контейнеров нужно обучить модель (`train.py`), иначе API стартует в rule-only режиме.

---

## API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `POST` | `/score` | Предварительная оценка одной заявки |
| `POST` | `/rank` | Ранжирование с фильтрами (регион, направление, уровень риска) |
| `GET` | `/explain/{app_id}` | Детальное объяснение по заявке |
| `GET` | `/stats` | Сводная аналитика по всему реестру |
| `GET` | `/applications` | Пагинированный список заявок |
| `GET` | `/health` | Статус API, параметры модели, runtime-метрики |

Поле `normative` в запросах не передаётся — система определяет его автоматически из справочника по `subsidy_type`.

---

## Ограничения

- Справочник нормативов (Приложение 1) не покрывает все варианты `subsidy_type` из реестра — неизвестные типы помечаются флагом `manual_review_required`.
- Критерии Приложения 2 (ГИСС, ИСЖ, ИБСПР, ЕАСУ, ИСЕГКН, приложенные документы) не могут быть проверены без внешних интеграций.
- Модель обучена на исторических решениях и отражает поведение прошлой системы, а не «идеальную справедливость».
- Система предназначена для pre-screening и decision-support — не для автономного юридически значимого решения.

---

## Зависимости

```
pandas>=2.0        numpy>=1.24        openpyxl>=3.1
fastapi>=0.110     uvicorn[standard]>=0.29
pydantic>=2.0      scikit-learn>=1.4
catboost>=1.2      joblib>=1.3
httpx>=0.27        streamlit>=1.28    plotly>=5.15
```
