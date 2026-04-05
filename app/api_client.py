import os
import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 30.0


def _get(endpoint: str, params: dict | None = None) -> dict | list:
    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        resp = client.get(endpoint, params=params)
        resp.raise_for_status()
        return resp.json()


def _post(endpoint: str, json_body: dict) -> dict:
    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        resp = client.post(endpoint, json=json_body)
        resp.raise_for_status()
        return resp.json()


# справочники 

@st.cache_data(ttl=300)
def get_regions() -> list[dict]:
    return _get("/regions")


@st.cache_data(ttl=300)
def get_directions() -> list[dict]:
    return _get("/directions")


@st.cache_data(ttl=300)
def get_subsidy_types(direction: str | None = None) -> list[dict]:
    params = {"direction": direction} if direction else {}
    return _get("/subsidy-types", params=params)


@st.cache_data(ttl=300)
def get_districts(region: str | None = None) -> list[dict]:
    params = {"region": region} if region else {}
    return _get("/districts", params=params)


@st.cache_data(ttl=300)
def get_akimats(region: str | None = None) -> list[dict]:
    params = {"region": region} if region else {}
    return _get("/akimats", params=params)


def score_new_application(data: dict) -> dict:
    return _post("/score", data)


# статистика

@st.cache_data(ttl=60)
def get_stats(
    region: str | None = None,
    direction: str | None = None,
    subsidy_type: str | None = None,
    risk_level: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> dict:
    params = {}
    if region:
        params["region"] = region
    if direction:
        params["direction"] = direction
    if subsidy_type:
        params["subsidy_type"] = subsidy_type
    if risk_level:
        params["risk_level"] = risk_level
    if min_score is not None:
        params["min_score"] = min_score
    if max_score is not None:
        params["max_score"] = max_score
    return _get("/stats", params=params)


# ранжирование

@st.cache_data(ttl=60)
def rank_applications(
    region: str | None = None,
    direction: str | None = None,
    subsidy_type: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    risk_level: str | None = None,
    top_n: int = 50,
) -> dict:
    body = {"top_n": top_n}
    if region:
        body["region"] = region
    if direction:
        body["direction"] = direction
    if subsidy_type:
        body["subsidy_type"] = subsidy_type
    if min_score is not None:
        body["min_score"] = min_score
    if max_score is not None:
        body["max_score"] = max_score
    if risk_level:
        body["risk_level"] = risk_level
    return _post("/rank", body)


# детали заявки

def get_explanation(app_id: str) -> dict:
    return _get(f"/explain/{app_id}")


@st.cache_data(ttl=60)
def get_factor_stats(
    region: str | None = None,
    direction: str | None = None,
    subsidy_type: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> dict:
    params = {}
    if region:
        params["region"] = region
    if direction:
        params["direction"] = direction
    if subsidy_type:
        params["subsidy_type"] = subsidy_type
    if min_score is not None:
        params["min_score"] = min_score
    if max_score is not None:
        params["max_score"] = max_score
    return _get("/factor-stats", params=params)


@st.cache_data(ttl=60)
def get_region_factors(
    direction: str | None = None,
    subsidy_type: str | None = None,
) -> list[dict]:
    params = {}
    if direction:
        params["direction"] = direction
    if subsidy_type:
        params["subsidy_type"] = subsidy_type
    return _get("/region-factors", params=params)


@st.cache_data(ttl=60)
def get_timeline(
    region: str | None = None,
    direction: str | None = None,
) -> list[dict]:
    params = {}
    if region:
        params["region"] = region
    if direction:
        params["direction"] = direction
    return _get("/timeline", params=params)


# загрузка датасета

def upload_dataset(file_bytes: bytes, filename: str) -> dict:
    with httpx.Client(base_url=API_URL, timeout=120.0) as client:
        resp = client.post(
            "/upload-dataset",
            files={"file": (filename, file_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        resp.raise_for_status()
        return resp.json()


def get_retrain_status() -> dict:
    return _get("/retrain-status")


# проверка доступности API

def check_health() -> dict | None:
    try:
        return _get("/health")
    except (httpx.ConnectError, httpx.HTTPError):
        return None
