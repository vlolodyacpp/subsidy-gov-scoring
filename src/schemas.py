# pydantic-схемы для валидации запросов и ответов API
from pydantic import BaseModel, Field
from typing import Optional


# модели запросов

class ScoreRequest(BaseModel):
    # запрос на скоринг одной заявки, норматив берётся из справочника автоматически
    region: str = Field(..., description="Регион подачи заявки")
    direction: str = Field(..., description="Направление (скотоводство, овцеводство и т.д.)")
    subsidy_type: str = Field(..., description="Тип субсидии")
    district: str = Field("", description="Район")
    akimat: str = Field("", description="Акимат")
    amount: float = Field(..., description="Запрашиваемая сумма субсидии (тенге)")
    submit_month: int = Field(6, ge=1, le=12, description="Месяц подачи заявки (1-12)")
    submit_day: int = Field(15, ge=1, le=31, description="День подачи заявки (1-31)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "region": "СКО",
                    "direction": "Субсидирование в мясном и мясо-молочном скотоводстве",
                    "subsidy_type": "Заявка на получение субсидий за приобретение племенных быков-производителей мясных и мясо-молочных пород",
                    "district": "г. Петропавловск",
                    "akimat": "Акимат СКО",
                    "amount": 780000,
                    "submit_month": 3,
                    "submit_day": 15,
                }
            ]
        }
    }


class RankRequest(BaseModel):
    # запрос на ранжирование заявок с фильтрами
    region: Optional[str] = Field(None, description="Фильтр по региону")
    direction: Optional[str] = Field(None, description="Фильтр по направлению")
    subsidy_type: Optional[str] = Field(None, description="Фильтр по типу субсидии")
    min_score: Optional[float] = Field(None, ge=0, le=100, description="Минимальный балл")
    max_score: Optional[float] = Field(None, ge=0, le=100, description="Максимальный балл")
    risk_level: Optional[str] = Field(None, description="Фильтр по уровню риска: Низкий, Средний, Высокий")
    top_n: int = Field(50, ge=1, le=1000, description="Количество заявок в результате")


# модели ответов

class FactorDetail(BaseModel):
    # детали одного фактора скоринга
    name: str = Field(..., description="Системное имя фактора")
    label: str = Field(..., description="Человекочитаемое название")
    value: Optional[str | float] = Field(None, description="Наблюдаемое значение признака")
    contribution: float = Field(..., description="Оценка вклада в итоговый балл")
    level: str = Field(..., description="Уровень: высокий / средний / низкий")


class ScoreResponse(BaseModel):
    # результат скоринга одной заявки
    score: float = Field(..., description="Итоговый балл 0-100")
    risk_level: str = Field(..., description="Уровень риска")
    rule_score: Optional[float] = Field(None, description="Rule-based score")
    ml_score: Optional[float] = Field(None, description="ML score 0-100")
    ml_probability: Optional[float] = Field(None, description="Оценка итоговой силы заявки по primary ML")
    history_match_source: str = Field("global", description="Какой исторический срез использован для advisory")
    history_match_count: int = Field(0, description="Сколько похожих исторических заявок найдено")
    history_approval_rate: Optional[float] = Field(None, description="Историческая одобряемость похожих заявок")
    history_advisory_score: Optional[float] = Field(None, description="Исторический advisory score 0-100")
    history_recommendation: Optional[str] = Field(None, description="Текстовая историческая рекомендация")
    history_note: Optional[str] = Field(None, description="Пояснение по historical advisory")
    disqualified: bool = Field(False, description="Флаг мгновенной дисквалификации")
    disqualification_reason: Optional[str] = Field(None, description="Причина дисквалификации")
    eligibility_status: str = Field("preliminarily_eligible", description="Статус eligibility-проверки")
    manual_review_required: bool = Field(True, description="Нужна ли ручная проверка критериев приложения 2")
    eligibility_note: Optional[str] = Field(None, description="Пояснение по предварительной eligibility-проверке")
    normative_reference_found: bool = Field(True, description="Найден ли эталонный норматив в локальном справочнике")
    scoring_engine: str = Field("merit-ml-advisory-v4.0", description="Использованный движок скоринга")
    model_name: Optional[str] = Field(None, description="Имя загруженной ML-модели")
    factors: list[FactorDetail] = Field(..., description="Детализация факторов")
    explanation: list[str] = Field(..., description="Текстовые объяснения")


class ApplicationBrief(BaseModel):
    # краткая информация о заявке в списке
    app_number: str
    region: str
    district: str
    direction: str
    subsidy_type: str
    amount: float
    status: str
    score: float
    risk_level: str
    rule_score: Optional[float] = None
    ml_score: Optional[float] = None
    ml_probability: Optional[float] = None
    history_match_source: str = "global"
    history_match_count: int = 0
    history_approval_rate: Optional[float] = None
    history_advisory_score: Optional[float] = None
    history_recommendation: Optional[str] = None
    history_note: Optional[str] = None
    disqualified: bool = False
    disqualification_reason: Optional[str] = None
    eligibility_status: str = "preliminarily_eligible"
    manual_review_required: bool = True
    eligibility_note: Optional[str] = None
    normative_reference_found: bool = True
    scoring_engine: str = "merit-ml-advisory-v4.0"
    model_name: Optional[str] = None
    top_factor: str


class RankResponse(BaseModel):
    # результат ранжирования
    total_filtered: int = Field(..., description="Всего заявок после фильтрации")
    returned: int = Field(..., description="Возвращено заявок")
    scoring_engine: str = Field("merit-ml-advisory-v4.0", description="Использованный движок скоринга")
    model_name: Optional[str] = Field(None, description="Имя загруженной ML-модели")
    applications: list[ApplicationBrief]


class ExplainResponse(BaseModel):
    # детальное объяснение скора заявки
    app_number: str
    region: str
    direction: str
    subsidy_type: str
    amount: float
    status: str
    normative: Optional[float] = Field(None, description="Норматив из заявки")
    ref_normative: Optional[float] = Field(None, description="Эталонный норматив из справочника")
    score: float
    risk_level: str
    rule_score: Optional[float] = Field(None, description="Rule-based score")
    ml_score: Optional[float] = Field(None, description="ML score 0-100")
    ml_probability: Optional[float] = Field(None, description="Оценка итоговой силы заявки по primary ML")
    history_match_source: str = Field("global", description="Какой исторический срез использован для advisory")
    history_match_count: int = Field(0, description="Сколько похожих исторических заявок найдено")
    history_approval_rate: Optional[float] = Field(None, description="Историческая одобряемость похожих заявок")
    history_advisory_score: Optional[float] = Field(None, description="Исторический advisory score 0-100")
    history_recommendation: Optional[str] = Field(None, description="Текстовая историческая рекомендация")
    history_note: Optional[str] = Field(None, description="Пояснение по historical advisory")
    disqualified: bool = Field(False, description="Флаг мгновенной дисквалификации")
    disqualification_reason: Optional[str] = Field(None, description="Причина дисквалификации")
    eligibility_status: str = Field("preliminarily_eligible", description="Статус eligibility-проверки")
    manual_review_required: bool = Field(True, description="Нужна ли ручная проверка критериев приложения 2")
    eligibility_note: Optional[str] = Field(None, description="Пояснение по preliminary eligibility")
    normative_reference_found: bool = Field(True, description="Найден ли эталонный норматив в локальном справочнике")
    scoring_engine: str = Field("merit-ml-advisory-v4.0", description="Использованный движок скоринга")
    model_name: Optional[str] = Field(None, description="Имя загруженной ML-модели")
    factors: list[FactorDetail]
    ml_factors: list[FactorDetail] = Field(default_factory=list, description="ML feature effects (top impacts)")
    explanation: list[str]


class RiskDistribution(BaseModel):
    # распределение по рискам
    low: int = Field(0, alias="Низкий")
    medium: int = Field(0, alias="Средний")
    high: int = Field(0, alias="Высокий")

    model_config = {"populate_by_name": True}


class RegionStat(BaseModel):
    # статистика по региону
    region: str
    count: int
    avg_score: float
    approval_rate: float


class StatsResponse(BaseModel):
    # агрегированная статистика по всем заявкам
    total_records: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float
    risk_distribution: dict[str, int]
    scoring_engine: str = "merit-ml-advisory-v4.0"
    model_name: Optional[str] = None
    top_regions: list[RegionStat]


class HealthResponse(BaseModel):
    # проверка работоспособности
    status: str = "ok"
    version: str = "4.0.0"
    records_loaded: int = 0
    scoring_engine: str = "merit-ml-advisory-v4.0"
    model_loaded: bool = False
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    started_at: Optional[str] = None
    model_created_at: Optional[str] = None
    calibration_method: Optional[str] = None
    decision_threshold: Optional[float] = None
    blend_rule_weight: Optional[float] = None
    blend_ml_weight: Optional[float] = None
    test_roc_auc: Optional[float] = None
    validation_roc_auc: Optional[float] = None
    region_sensitivity_mean_delta: Optional[float] = None
    score_requests_total: int = 0
    rank_requests_total: int = 0
    explain_requests_total: int = 0
    avg_score_latency_ms: Optional[float] = None
    avg_rank_latency_ms: Optional[float] = None
    avg_explain_latency_ms: Optional[float] = None
    dataset_name: Optional[str] = None


class PaginatedApplications(BaseModel):
    # список заявок с пагинацией
    total: int
    page: int
    per_page: int
    scoring_engine: str = "merit-ml-advisory-v4.0"
    model_name: Optional[str] = None
    applications: list[ApplicationBrief]
