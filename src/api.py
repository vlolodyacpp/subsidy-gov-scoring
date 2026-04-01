import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline
from src.features import build_feature_tables, extract_features, extract_features_batch
from src.normatives import get_normative_for_type, build_normative_lookup
from src.scoring import (
    score_single,
    score_batch,
    get_score_distribution,
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

    logger.info("Feature engineering v2 ...")
    tables = build_feature_tables(df)
    features = extract_features_batch(df, tables)

    logger.info("Batch скоринг v2 ...")
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
    title="Subsidy Scoring API v2",
    description=(
        "AI-система ранжирования заявок на субсидирование "
        "сельхозпроизводителей Казахстана. "
        "12 факторов на основе Правил субсидирования (Приказ МСХ РК №108). "
        "Рассчитывает score 0–100 и объясняет каждый результат."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_combined() -> pd.DataFrame:
    return pd.concat([app.state.df, app.state.scores], axis=1)


def _apply_filters(
    combined: pd.DataFrame,
    region: str | None = None,
    direction: str | None = None,
    subsidy_type: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    risk_level: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=combined.index)
    if region:
        mask &= combined["region"] == region
    if direction:
        mask &= combined["direction"] == direction
    if subsidy_type:
        mask &= combined["subsidy_type"] == subsidy_type
    if min_score is not None:
        mask &= combined["score"] >= min_score
    if max_score is not None:
        mask &= combined["score"] <= max_score
    if risk_level:
        mask &= combined["risk_level"] == risk_level
    return combined[mask]


def _make_app_brief(row: pd.Series) -> ApplicationBrief:
    return ApplicationBrief(
        app_number=str(row.get("app_number", "")),
        region=str(row.get("region", "")),
        district=str(row.get("district", "")),
        direction=str(row.get("direction", "")),
        subsidy_type=str(row.get("subsidy_type", "")),
        amount=float(row.get("amount", 0)),
        status=str(row.get("status", "")),
        score=float(row.get("score", 0)),
        risk_level=str(row.get("risk_level", "")),
        top_factor=str(row.get("top_factor_label", "")),
    )


def _build_factor_details(features_dict: dict, scoring_result) -> list[FactorDetail]:
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


def _group_stats(column: str) -> list[dict]:
    combined = _get_combined()
    stats = (
        combined.groupby(column)["score"]
        .agg(count="size", avg_score="mean")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return [
        {
            column: row[column],
            "count": int(row["count"]),
            "avg_score": round(float(row["avg_score"]), 1),
        }
        for _, row in stats.iterrows()
    ]


# endpoints 


@app.get("/health", response_model=HealthResponse, tags=["Системные"])
async def health_check():
    return HealthResponse(
        status="ok",
        records_loaded=len(app.state.df),
    )


@app.post("/score", response_model=ScoreResponse, tags=["Скоринг"])
async def score_application(request: ScoreRequest):
    # скоринг одной заявки, норматив подставляется из справочника автоматически
    norm_lookup = app.state.tables["normative_lookup"]
    ref_norm = get_normative_for_type(request.subsidy_type, norm_lookup)

    # строим submit_date из месяца и дня
    try:
        submit_date = datetime(2025, request.submit_month, request.submit_day)
    except ValueError:
        submit_date = datetime(2025, request.submit_month, 15)

    row = pd.Series(
        {
            "region": request.region,
            "direction": request.direction,
            "subsidy_type": request.subsidy_type,
            "district": request.district,
            "akimat": request.akimat,
            "amount": request.amount,
            "normative": ref_norm or 0,
            "submit_date": submit_date,
            "submit_month": request.submit_month,
        }
    )

    features_dict = extract_features(row, app.state.tables)
    result = score_single(features_dict)

    return ScoreResponse(
        score=result.score,
        risk_level=result.risk_level,
        factors=_build_factor_details(features_dict, result),
        explanation=result.explanation,
    )


@app.post("/rank", response_model=RankResponse, tags=["Скоринг"])
async def rank_applications(request: RankRequest):
    combined = _get_combined()
    filtered = _apply_filters(
        combined,
        region=request.region,
        direction=request.direction,
        subsidy_type=request.subsidy_type,
        min_score=request.min_score,
        max_score=request.max_score,
        risk_level=request.risk_level,
    ).sort_values("score", ascending=False)

    total_filtered = len(filtered)
    top = filtered.head(request.top_n)

    return RankResponse(
        total_filtered=total_filtered,
        returned=len(top),
        applications=[_make_app_brief(row) for _, row in top.iterrows()],
    )


@app.get("/explain/{app_id}", response_model=ExplainResponse, tags=["Скоринг"])
async def explain_score(app_id: str):
    df = app.state.df
    match = df[df["app_number"].astype(str) == app_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Заявка {app_id} не найдена")

    row = match.iloc[0]
    idx = match.index[0]

    # используем предрассчитанные batch-фичи для консистентности с шортлистом
    features_dict = app.state.features.loc[idx].to_dict()
    result = score_single(features_dict)

    # нормативы: из заявки и эталонный
    norm_lookup = app.state.tables["normative_lookup"]
    ref_norm = get_normative_for_type(str(row["subsidy_type"]), norm_lookup)

    return ExplainResponse(
        app_number=str(row["app_number"]),
        region=str(row["region"]),
        direction=str(row["direction"]),
        subsidy_type=str(row["subsidy_type"]),
        amount=float(row["amount"]),
        status=str(row["status"]),
        normative=float(row["normative"]) if pd.notna(row.get("normative")) else None,
        ref_normative=float(ref_norm) if ref_norm else None,
        score=result.score,
        risk_level=result.risk_level,
        factors=_build_factor_details(features_dict, result),
        explanation=result.explanation,
    )


@app.get("/stats", response_model=StatsResponse, tags=["Аналитика"])
async def get_stats(
    region: str | None = Query(None, description="Фильтр по региону"),
    direction: str | None = Query(None, description="Фильтр по направлению"),
    subsidy_type: str | None = Query(None, description="Фильтр по типу субсидии"),
    risk_level: str | None = Query(None, description="Фильтр по уровню риска"),
    min_score: float | None = Query(None, ge=0, le=100, description="Минимальный балл"),
    max_score: float | None = Query(None, ge=0, le=100, description="Максимальный балл"),
):
    combined = _get_combined()
    filtered = _apply_filters(
        combined, region=region, direction=direction,
        subsidy_type=subsidy_type, risk_level=risk_level,
        min_score=min_score, max_score=max_score,
    )
    filtered_scores = app.state.scores.loc[filtered.index]

    dist = get_score_distribution(filtered_scores)

    # статистика по регионам
    region_stats = (
        filtered.groupby("region")["score"]
        .agg(count="size", avg_score="mean")
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    ar = app.state.tables["approval_rates"]["region"]
    top_regions = [
        RegionStat(
            region=r["region"],
            count=int(r["count"]),
            avg_score=round(float(r["avg_score"]), 1),
            approval_rate=round(float(ar.get(r["region"], {}).get("approval_rate", 0)), 4),
        )
        for _, r in region_stats.head(10).iterrows()
    ]

    risk_dict = {str(k): int(v) for k, v in dist["risk_distribution"].items()}

    return StatsResponse(
        total_records=len(filtered),
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
    min_score: float | None = Query(None, ge=0, le=100, description="Минимальный балл"),
    max_score: float | None = Query(None, ge=0, le=100, description="Максимальный балл"),
    sort_by: str = Query("score", description="Поле сортировки: score, amount"),
    sort_order: str = Query("desc", description="Порядок: asc / desc"),
):
    combined = _get_combined()
    filtered = _apply_filters(
        combined, region=region, direction=direction,
        min_score=min_score, max_score=max_score,
    )

    ascending = sort_order.lower() == "asc"
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    total = len(filtered)
    start = (page - 1) * per_page
    page_data = filtered.iloc[start:start + per_page]

    return PaginatedApplications(
        total=total,
        page=page,
        per_page=per_page,
        applications=[_make_app_brief(row) for _, row in page_data.iterrows()],
    )


@app.get("/applications/{app_id}", response_model=ApplicationBrief, tags=["Заявки"])
async def get_application(app_id: str):
    df = app.state.df
    match = df[df["app_number"].astype(str) == app_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Заявка {app_id} не найдена")

    idx = match.index[0]
    combined = _get_combined()
    return _make_app_brief(combined.loc[idx])


@app.get("/factor-stats", tags=["Аналитика"])
async def get_factor_stats(
    region: str | None = Query(None),
    direction: str | None = Query(None),
    subsidy_type: str | None = Query(None),
    min_score: float | None = Query(None, ge=0, le=100),
    max_score: float | None = Query(None, ge=0, le=100),
):
    """Средние значения каждого фактора (для графиков распределения)."""
    combined = _get_combined()
    filtered = _apply_filters(
        combined, region=region, direction=direction,
        subsidy_type=subsidy_type,
        min_score=min_score, max_score=max_score,
    )
    features = app.state.features.loc[filtered.index]

    result = {}
    for factor in FACTOR_LABELS:
        if factor in features.columns:
            col = features[factor]
            std_val = col.std()
            result[factor] = {
                "label": FACTOR_LABELS[factor],
                "mean": round(float(col.mean()), 4),
                "median": round(float(col.median()), 4),
                "std": round(float(std_val), 4) if pd.notna(std_val) else 0.0,
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
                "values": col.round(3).tolist(),
            }
    return result


@app.get("/region-factors", tags=["Аналитика"])
async def get_region_factors(
    direction: str | None = Query(None),
    subsidy_type: str | None = Query(None),
):
    """Средние значения факторов в разрезе регионов."""
    combined = _get_combined()
    filtered = _apply_filters(combined, direction=direction, subsidy_type=subsidy_type)
    features = app.state.features.loc[filtered.index]

    # добавляем регион к features
    feat_with_region = features.copy()
    feat_with_region["region"] = filtered["region"]

    factor_cols = [c for c in FACTOR_LABELS if c in features.columns]
    grouped = feat_with_region.groupby("region")[factor_cols].mean()

    result = []
    for region, row in grouped.iterrows():
        factors = {}
        for col in factor_cols:
            factors[col] = {
                "label": FACTOR_LABELS[col],
                "mean": round(float(row[col]), 4),
            }
        result.append({
            "region": region,
            "count": int((filtered["region"] == region).sum()),
            "factors": factors,
        })

    return sorted(result, key=lambda x: x["count"], reverse=True)


@app.get("/timeline", tags=["Аналитика"])
async def get_timeline(
    region: str | None = Query(None),
    direction: str | None = Query(None),
):
    """Динамика бюджетного давления и очереди по месяцам."""
    combined = _get_combined()
    filtered = _apply_filters(combined, region=region, direction=direction)
    features = app.state.features.loc[filtered.index]

    df_timeline = pd.DataFrame({
        "submit_month": filtered["submit_month"],
        "budget_pressure": features["budget_pressure"] if "budget_pressure" in features.columns else 0,
        "queue_position": features["queue_position"] if "queue_position" in features.columns else 0,
        "score": filtered["score"],
    })

    monthly = df_timeline.groupby("submit_month").agg(
        count=("score", "size"),
        avg_score=("score", "mean"),
        avg_budget_pressure=("budget_pressure", "mean"),
        avg_queue_position=("queue_position", "mean"),
    ).reset_index()

    monthly = monthly.sort_values("submit_month")

    return [
        {
            "month": int(row["submit_month"]),
            "count": int(row["count"]),
            "avg_score": round(float(row["avg_score"]), 1),
            "avg_budget_pressure": round(float(row["avg_budget_pressure"]), 4),
            "avg_queue_position": round(float(row["avg_queue_position"]), 4),
        }
        for _, row in monthly.iterrows()
    ]


@app.get("/regions", tags=["Справочники"])
async def list_regions():
    return _group_stats("region")


@app.get("/directions", tags=["Справочники"])
async def list_directions():
    return _group_stats("direction")
