# pydantic-схемы для валидации запросов и ответов API

from pydantic import BaseModel, Field
from typing import Optional


# ── модели запросов ───────────────────────────────────────────

class ScoreRequest(BaseModel):
    # запрос на скоринг одной заявки по сырым данным
    region: str = Field(..., description="Регион подачи заявки")
    direction: str = Field(..., description="Направление (скотоводство, овцеводство и т.д.)")
    subsidy_type: str = Field(..., description="Тип субсидии")
    district: str = Field("", description="Район")
    amount: float = Field(..., description="Запрашиваемая сумма субсидии (тенге)")
    normative: float = Field(0, description="Норматив субсидирования (тенге)")
    submit_month: int = Field(6, ge=1, le=12, description="Месяц подачи заявки (1-12)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "region": "СКО",
                    "direction": "Субсидирование в скотоводстве",
                    "subsidy_type": "На возмещение части затрат на ведение селекционной и племенной работы",
                    "district": "г. Петропавловск",
                    "amount": 400000,
                    "normative": 5000,
                    "submit_month": 3,
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


# ── модели ответов ────────────────────────────────────────────

class FactorDetail(BaseModel):
    # детали одного фактора скоринга
    name: str = Field(..., description="Системное имя фактора")
    label: str = Field(..., description="Человекочитаемое название")
    value: float = Field(..., description="Значение признака (0-1)")
    contribution: float = Field(..., description="Вклад в итоговый балл")
    level: str = Field(..., description="Уровень: высокий / средний / низкий")


class ScoreResponse(BaseModel):
    # результат скоринга одной заявки
    score: float = Field(..., description="Итоговый балл 0-100")
    risk_level: str = Field(..., description="Уровень риска")
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
    score: float
    risk_level: str
    top_factor: str


class RankResponse(BaseModel):
    # результат ранжирования
    total_filtered: int = Field(..., description="Всего заявок после фильтрации")
    returned: int = Field(..., description="Возвращено заявок")
    applications: list[ApplicationBrief]


class ExplainResponse(BaseModel):
    # детальное объяснение скора заявки
    app_number: str
    region: str
    direction: str
    subsidy_type: str
    amount: float
    status: str
    score: float
    risk_level: str
    factors: list[FactorDetail]
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
    top_regions: list[RegionStat]


class HealthResponse(BaseModel):
    # проверка работоспособности
    status: str = "ok"
    version: str = "0.2.0"
    records_loaded: int = 0
    scoring_engine: str = "rule-based-v1"


class PaginatedApplications(BaseModel):
    # список заявок с пагинацией
    total: int
    page: int
    per_page: int
    applications: list[ApplicationBrief]
