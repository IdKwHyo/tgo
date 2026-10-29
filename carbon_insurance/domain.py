from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assessment_today() -> str:
    """TGO assessment dates follow Thailand civil time; audit timestamps retain UTC offsets."""
    return datetime.now(ZoneInfo("Asia/Bangkok")).date().isoformat()


def digest(value: bytes | str | Any) -> str:
    if not isinstance(value, (bytes, str)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def require_tgo_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == "tgo.or.th" or host.endswith(".tgo.or.th")):
        raise ValueError(f"Only official HTTPS TGO sources are permitted: {url}")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Credentials and nonstandard ports are not permitted in TGO URLs")
    return url


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path = Path("data")
    timeout_seconds: float = Field(45, gt=0)
    retries: int = Field(2, ge=0, le=5)
    max_workers: int = Field(3, ge=1, le=5)
    max_projects: int | None = Field(None, gt=0)
    max_project_pdfs: int = Field(12, ge=0)
    sample_project_ids: list[str] = ["449", "667", "718", "8"]
    catboost_min_projects: int = Field(40, ge=20)
    catboost_min_test_projects: int = Field(10, ge=5)
    lstm_min_observations: int = Field(60, ge=40)
    lstm_lookback: int = Field(6, ge=2)
    baseline_min_observations: int = Field(8, ge=6)
    minimum_relative_improvement: float = Field(0.05, ge=0, lt=1)
    stale_after_days: int = Field(90, gt=0)
    gemini_model: str = "gemini-3.8-flash"
    retrieval_top_k: int = Field(8, ge=1, le=30)

    @classmethod
    def load(cls, path: str | None = None, data_dir: str | None = None) -> "Settings":
        values = yaml.safe_load(Path(path).read_text()) if path else {}
        values = values or {}
        if data_dir:
            values["data_dir"] = data_dir
        result = cls(**values)
        result.data_dir = result.data_dir.resolve()
        return result


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    project_id: str  # registry URL identifier, NOT the registration number
    registration_number: str | None = None
    program: str
    group: str
    category: Literal["FOR&AGR"] = "FOR&AGR"
    name: str
    methodology: str | None = None
    methodology_codes: list[str] = []
    status: str | None = None
    size: str | None = None
    location: str | None = None
    developer: str | None = None
    owner: str | None = None
    validator: str | None = None
    registration_date: str | None = None
    fiscal_year_be: int | None = None
    crediting_start: str | None = None
    crediting_end: str | None = None
    stated_duration: str | None = None
    expected_annual_tonnes: float | None = Field(None, ge=0)
    source_id: str
    source_url: str
    raw_fields: dict[str, Any] = {}
    warnings: list[str] = []


class FactorReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factor: str
    support: Literal["SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"]
    insurance_relevance: Literal["RELEVANT", "NOT_RELEVANT", "UNCERTAIN"]
    evidence_chunk_ids: list[str]
    reason: str


class QualityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["ACCEPT", "REJECT", "INVESTIGATE"]
    factors: list[FactorReview]
    statistical_credibility: Literal["SUPPORTED", "INSUFFICIENT", "FAILED"]
    material_model_disagreement: bool
    eligibility_violation: bool
    additional_evidence: list[str]
    reason: str
