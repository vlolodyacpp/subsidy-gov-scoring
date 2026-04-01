# Subsidy Scoring System

Кейс 2 Decentrathon 5.0: AI/Data-driven система предварительной оценки заявок на субсидирование в животноводстве.

## Что делает проект

Проект читает реестр заявок из `xlsx`, очищает данные, извлекает признаки, считает rule-based score, применяет ML-модель и отдаёт:

- предварительный `score` 0-100;
- explainability по факторам;
- `shortlist` для комиссии;
- признаки eligibility-проверки.

Система является инструментом поддержки решения. Она не заменяет комиссию, МИО и установленный Правилами порядок рассмотрения заявок.

## Важное ограничение по соответствию Правилам

Проект не является самостоятельной системой назначения и выплаты субсидий.

Что покрывается автоматически:

- окно подачи `20 января - 20 декабря`;
- базовая целостность заявки;
- локальная сверка с нормативами из приложения 1;
- предварительный аналитический score;
- explainability.

Что пока не покрывается автоматически и требует ручной или интеграционной проверки:

- критерии приложения 2, зависящие от ГИСС;
- проверки через ИСЖ, ИБСПР, ЕАСУ, ИСЕГКН;
- приложенные документы и их полнота;
- встречные обязательства;
- сохранность поголовья;
- прочие юридически значимые проверки из Правил.

То есть результат проекта нужно интерпретировать как `pre-screening` и `decision-support`, а не как автоматическое юридически значимое решение.

## Текущая архитектура

```text
Excel registry
  -> pipeline.py
  -> features.py
  -> eligibility.py
  -> scoring.py
  -> modeling.py
  -> api.py
```

### Слои

- `src/pipeline.py`
  Чтение Excel, очистка, парсинг дат, построение целевой переменной.

- `src/normatives.py` и `src/normatives_dict.py`
  Локальный справочник нормативов из приложения 1 и базовые проверки по сроку подачи.

- `src/features.py`
  Feature engineering для rule-based и ML-слоя.

- `src/eligibility.py`
  Предварительная eligibility-проверка:
  - hard fail по дедлайну и базовым некорректным данным;
  - manual review flag для критериев приложения 2, которые нельзя честно проверить без внешних систем.

- `src/scoring.py`
  Rule-based score и explainability.

- `src/modeling.py`
  Подготовка датасета, обучение dual-branch модели, калибровка, подбор порога, сохранение bundle.

- `src/api.py`
  FastAPI-слой с endpoint-ами для скоринга, explainability и аналитики.

## Данные

Основной датасет:

- `data/subsidies.xlsx`

Текущее чтение ожидает лист `Page 1`.

После очистки:

- исключаются заявки со статусом `Отозвано`;
- positive-класс строится по статусам:
  - `Исполнена`
  - `Одобрена`
  - `Сформировано поручение`
  - `Получена`
- negative-класс:
  - `Отклонена`

## Скоринг

### 1. Eligibility

Сначала заявка проходит предварительную eligibility-проверку.

Если заявка:

- подана вне окна `20 января - 20 декабря`;
- имеет некорректную сумму;
- не содержит базовых обязательных полей,

она получает:

- `disqualified = true`
- `eligibility_status = failed`
- `score = 0`

Такие заявки не попадают в ML-инференс.

### 2. Rule-based score

Для допустимых заявок считается rule-based score по 11 факторам:

- `normative_match`
- `amount_normative_integrity`
- `amount_adequacy`
- `budget_pressure`
- `queue_position`
- `region_specialization`
- `region_direction_approval_rate`
- `akimat_approval_rate`
- `unit_count`
- `direction_approval_rate`
- `subsidy_type_approval_rate`

Дедлайн вынесен из score и работает как отдельный eligibility-фильтр.

### 3. ML score

ML-слой использует dual-branch primary model:

- `context branch` на `RandomForest`
  Видит ограниченный контекст заявки, включая `region`, `direction`, `subsidy_type`.
- `core branch` на `LogisticRegression`
  Видит ту же заявку, но в region-neutral представлении:
  `region` зануляется, а `region_specialization` переводится в нейтральное значение.

Итоговая вероятность получается смешиванием обеих веток. Это сделано специально, чтобы:

- не выбрасывать полезный региональный контекст полностью;
- но и не позволять `region` доминировать над качеством самой заявки.

Готовый production bundle сохраняется в:

- `models/artifacts/subsidy_model.joblib`

### 4. Финальный score

Для допустимых заявок итоговый `score` сейчас строится от primary dual-branch ML:

`final_score = ml_score`

`rule_score` остаётся в системе как диагностический и explainability-слой, а history advisory возвращается отдельно и не управляет финальным баллом напрямую.

## Explainability

API возвращает:

- factor contributions;
- текстовое объяснение;
- статус предварительной eligibility-проверки;
- флаг `manual_review_required`.

Если критерии приложения 2 не могут быть подтверждены автоматически, заявка помечается как:

- `eligibility_status = preliminarily_eligible`
- `manual_review_required = true`

Это сделано специально, чтобы не создавать ложное ощущение полной юридической проверки.

## API

Запуск сервера:

```bash
.venv/bin/python main.py --serve
```

Swagger UI:

- `http://127.0.0.1:8000/docs`

Основные endpoint-ы:

- `POST /score`
  Предварительная оценка одной заявки.

- `POST /rank`
  Ранжирование заявок по score.

- `GET /explain/{app_id}`
  Подробное объяснение по заявке.

- `GET /stats`
  Сводная аналитика.

- `GET /applications`
  Пагинированный список заявок.

- `GET /health`
  Статус API и ML-модели, включая dual-branch конфигурацию.

## Установка и запуск

Из корня проекта:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Обучение модели:

```bash
.venv/bin/python train.py
```

После обучения появятся:

- `models/artifacts/subsidy_model.joblib`
- `models/reports/training_metrics.json`
- `models/reports/test_predictions.csv`

Batch prediction:

```bash
.venv/bin/python predict.py
```

API:

```bash
.venv/bin/python main.py --serve
```

Важно:

- API ожидает, что `models/artifacts/subsidy_model.joblib` уже существует;
- если модель не обучена, сервер не стартует и попросит сначала выполнить `.venv/bin/python train.py`.

## Что изменилось в v3.2

- primary ML переведён на dual-branch архитектуру;
- влияние `region` снижено: одна ветка модели принципиально не видит `region`;
- выбор порога стал более recall-friendly;
- historical advisory полностью отделён от финального decision score;
- `/health` теперь показывает состав веток модели и их веса.

## Что лежит в репозитории сейчас

```text
src/
  api.py
  eligibility.py
  features.py
  modeling.py
  normatives.py
  normatives_dict.py
  pipeline.py
  schemas.py
  scoring.py
models/
  artifacts/
  reports/
main.py
train.py
predict.py
requirements.txt
```

В текущем репозитории нет Streamlit-dashboard и Docker-конфигурации, поэтому старые описания такого стека больше неактуальны.

## Ограничения

- Локальный справочник нормативов ещё не покрывает все варианты `subsidy_type` из реестра.
- Часть критериев Правил не может быть автоматически проверена без внешних интеграций.
- Модель учится на исторических решениях, поэтому отражает поведение прошлой системы, а не "идеальную справедливость".
- Проект подходит для аналитики и приоритизации, но не должен использоваться как единственный источник решения.

## Ближайшие доработки

- закрыть оставшиеся `subsidy_type` в локальном справочнике нормативов;
- добавить интеграции или заглушки под проверки приложения 2;
- расширить README примерами запросов и ответов API;
- добавить отдельный compliance-report по покрытию требований Правил.
