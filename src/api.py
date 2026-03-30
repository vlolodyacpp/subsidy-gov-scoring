# REST API для системы скоринга субсидий

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline
from src.features import build_feature_tables, extract_features, extract_features_batch
from src.scoring import (
    score_single,
    score_batch,
    get_score_distribution,
    WEIGHTS,
    FACTOR_LABELS,
)
from src.schemas import (
    ScoreRequest,
    ScoreResponse,
    RankRequest,
    RankResponse,
    ExplainResponse,
    StatsResponse,
    HealthResponse,
    FactorDetail,
    ApplicationBrief,
    RegionStat,
    PaginatedApplications,
)

logger = logging.getLogger("subsidy_api")
logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/subsidies.xlsx"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # загрузка данных и расчёт скоров при старте сервера
    logger.info("Загрузка данных из %s ...", DATA_PATH)
    df = run_pipeline(DATA_PATH)

    logger.info("Feature engineering ...")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)

    logger.info("Batch скоринг ...")
    scores = score_batch(features)

    # сохраняем в state
    app.state.df = df
    app.state.tables = tables
    app.state.features = features
    app.state.scores = scores

    logger.info(
        "Сервер готов: %d записей, средний балл %.1f",
        len(df),
        scores["score"].mean(),
    )
    yield
    logger.info("Сервер остановлен.")


app = FastAPI(
    title="Subsidy Scoring API",
    description=(
        "AI-система ранжирования заявок на субсидирование "
        "сельхозпроизводителей Казахстана. "
        "Рассчитывает score 0–100 и объясняет каждый результат."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_factor_details(features_dict: dict, scoring_result) -> list[FactorDetail]:
    # преобразует ScoringResult + features в список FactorDetail
    details = []
    sorted_factors = sorted(
        scoring_result.factors.items(), key=lambda x: x[1], reverse=True
    )
    for name, contribution in sorted_factors:
        value = features_dict.get(name, 0)
        if contribution >= 15:
            level = "высокий"
        elif contribution >= 8:
            level = "средний"
        else:
            level = "низкий"
        details.append(
            FactorDetail(
                name=name,
                label=FACTOR_LABELS.get(name, name),
                value=round(value, 4),
                contribution=contribution,
                level=level,
            )
        )
    return details


def _make_app_brief(row: pd.Series, score_row: pd.Series) -> ApplicationBrief:
    # создаёт ApplicationBrief из строк df и scores
    return ApplicationBrief(
        app_number=str(row.get("app_number", "")),
        region=str(row.get("region", "")),
        district=str(row.get("district", "")),
        direction=str(row.get("direction", "")),
        subsidy_type=str(row.get("subsidy_type", "")),
        amount=float(row.get("amount", 0)),
        score=float(score_row.get("score", 0)),
        risk_level=str(score_row.get("risk_level", "")),
        top_factor=str(score_row.get("top_factor_label", "")),
    )


# ── endpoints ─────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Системные"])
async def health_check():
    # проверка работоспособности API
    return HealthResponse(
        status="ok",
        records_loaded=len(app.state.df),
    )


@app.post("/score", response_model=ScoreResponse, tags=["Скоринг"])
async def score_application(request: ScoreRequest):
    # скоринг одной заявки — принимает сырые данные, вычисляет признаки, возвращает балл
    row = pd.Series(
        {
            "region": request.region,
            "direction": request.direction,
            "subsidy_type": request.subsidy_type,
            "district": request.district,
            "amount": request.amount,
            "normative": request.normative,
            "submit_month": request.submit_month,
        }
    )

    tables = app.state.tables
    features_dict = extract_features(row, tables)
    result = score_single(features_dict)

    return ScoreResponse(
        score=result.score,
        risk_level=result.risk_level,
        factors=_build_factor_details(features_dict, result),
        explanation=result.explanation,
    )


@app.post("/rank", response_model=RankResponse, tags=["Скоринг"])
async def rank_applications(request: RankRequest):
    # ранжирование заявок с фильтрами, возвращает топ-N отсортированных
    df = app.state.df
    scores = app.state.scores

    # объединяем для фильтрации
    combined = pd.concat([df, scores], axis=1)

    # применяем фильтры
    mask = pd.Series(True, index=combined.index)

    if request.region:
        mask &= combined["region"] == request.region
    if request.direction:
        mask &= combined["direction"] == request.direction
    if request.subsidy_type:
        mask &= combined["subsidy_type"] == request.subsidy_type
    if request.min_score is not None:
        mask &= combined["score"] >= request.min_score
    if request.max_score is not None:
        mask &= combined["score"] <= request.max_score
    if request.risk_level:
        mask &= combined["risk_level"] == request.risk_level

    filtered = combined[mask].sort_values("score", ascending=False)
    total_filtered = len(filtered)
    top = filtered.head(request.top_n)

    applications = []
    for idx, row in top.iterrows():
        applications.append(
            ApplicationBrief(
                app_number=str(row.get("app_number", "")),
                region=str(row.get("region", "")),
                district=str(row.get("district", "")),
                direction=str(row.get("direction", "")),
                subsidy_type=str(row.get("subsidy_type", "")),
                amount=float(row.get("amount", 0)),
                score=float(row.get("score", 0)),
                risk_level=str(row.get("risk_level", "")),
                top_factor=str(row.get("top_factor_label", "")),
            )
        )

    return RankResponse(
        total_filtered=total_filtered,
        returned=len(applications),
        applications=applications,
    )


@app.get("/explain/{app_id}", response_model=ExplainResponse, tags=["Скоринг"])
async def explain_score(app_id: str):
    # детальное объяснение скора — ищет заявку по номеру, возвращает факторы
    df = app.state.df
    tables = app.state.tables

    match = df[df["app_number"].astype(str) == app_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Заявка {app_id} не найдена")

    row = match.iloc[0]
    features_dict = extract_features(row, tables)
    result = score_single(features_dict)

    return ExplainResponse(
        app_number=str(row["app_number"]),
        region=str(row["region"]),
        direction=str(row["direction"]),
        subsidy_type=str(row["subsidy_type"]),
        amount=float(row["amount"]),
        status=str(row["status"]),
        score=result.score,
        risk_level=result.risk_level,
        factors=_build_factor_details(features_dict, result),
        explanation=result.explanation,
    )


@app.get("/stats", response_model=StatsResponse, tags=["Аналитика"])
async def get_stats():
    # агрегированная статистика — распределение баллов, риски, топ регионов
    df = app.state.df
    scores = app.state.scores

    dist = get_score_distribution(scores)

    # статистика по регионам
    combined = pd.concat([df[["region"]], scores[["score"]]], axis=1)
    region_stats = (
        combined.groupby("region")
        .agg(count=("score", "size"), avg_score=("score", "mean"))
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    # добавляем approval_rate из справочных таблиц
    ar = app.state.tables["approval_rates"]["region"]
    top_regions = []
    for _, r in region_stats.head(10).iterrows():
        approval_rate = ar.get(r["region"], {}).get("approval_rate", 0)
        top_regions.append(
            RegionStat(
                region=r["region"],
                count=int(r["count"]),
                avg_score=round(float(r["avg_score"]), 1),
                approval_rate=round(float(approval_rate), 4),
            )
        )

    # приводим к dict[str, int]
    risk_raw = dist["risk_distribution"]
    risk_dict = {str(k): int(v) for k, v in risk_raw.items()}

    return StatsResponse(
        total_records=len(df),
        mean_score=dist["mean"],
        median_score=dist["median"],
        std_score=dist["std"],
        min_score=dist["min"],
        max_score=dist["max"],
        risk_distribution=risk_dict,
        top_regions=top_regions,
    )


@app.get("/applications", response_model=PaginatedApplications, tags=["Заявки"])
async def list_applications(
    page: int = Query(1, ge=1, description="Номер страницы"),
    per_page: int = Query(20, ge=1, le=100, description="Заявок на странице"),
    region: str | None = Query(None, description="Фильтр по региону"),
    direction: str | None = Query(None, description="Фильтр по направлению"),
    sort_by: str = Query("score", description="Поле сортировки: score, amount"),
    sort_order: str = Query("desc", description="Порядок: asc / desc"),
):
    # список заявок с пагинацией и фильтрацией
    df = app.state.df
    scores = app.state.scores

    combined = pd.concat([df, scores], axis=1)

    mask = pd.Series(True, index=combined.index)
    if region:
        mask &= combined["region"] == region
    if direction:
        mask &= combined["direction"] == direction

    filtered = combined[mask]

    ascending = sort_order.lower() == "asc"
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]

    applications = []
    for idx, row in page_data.iterrows():
        applications.append(
            ApplicationBrief(
                app_number=str(row.get("app_number", "")),
                region=str(row.get("region", "")),
                district=str(row.get("district", "")),
                direction=str(row.get("direction", "")),
                subsidy_type=str(row.get("subsidy_type", "")),
                amount=float(row.get("amount", 0)),
                score=float(row.get("score", 0)),
                risk_level=str(row.get("risk_level", "")),
                top_factor=str(row.get("top_factor_label", "")),
            )
        )

    return PaginatedApplications(
        total=total,
        page=page,
        per_page=per_page,
        applications=applications,
    )


@app.get("/applications/{app_id}", response_model=ApplicationBrief, tags=["Заявки"])
async def get_application(app_id: str):
    # получить информацию о конкретной заявке по номеру
    df = app.state.df
    scores = app.state.scores

    match = df[df["app_number"].astype(str) == app_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Заявка {app_id} не найдена")

    idx = match.index[0]
    row = df.loc[idx]
    score_row = scores.loc[idx]

    return _make_app_brief(row, score_row)


@app.get("/regions", tags=["Справочники"])
async def list_regions():
    # список всех регионов с количеством заявок и средним баллом
    df = app.state.df
    scores = app.state.scores

    combined = pd.concat([df[["region"]], scores[["score"]]], axis=1)
    stats = (
        combined.groupby("region")
        .agg(count=("score", "size"), avg_score=("score", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    return [
        {
            "region": row["region"],
            "count": int(row["count"]),
            "avg_score": round(float(row["avg_score"]), 1),
        }
        for _, row in stats.iterrows()
    ]


@app.get("/directions", tags=["Справочники"])
async def list_directions():
    # список всех направлений субсидирования с количеством заявок
    df = app.state.df
    scores = app.state.scores

    combined = pd.concat([df[["direction"]], scores[["score"]]], axis=1)
    stats = (
        combined.groupby("direction")
        .agg(count=("score", "size"), avg_score=("score", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    return [
        {
            "direction": row["direction"],
            "count": int(row["count"]),
            "avg_score": round(float(row["avg_score"]), 1),
        }
        for _, row in stats.iterrows()
    ]
