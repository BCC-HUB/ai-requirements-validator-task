"""Модели данных: результаты анализа (для отчёта и JSON) и схемы ответов модели."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reqvalidator.criteria import normalize_criterion, normalize_severity

Severity = Literal["critical", "major", "minor"]
Status = Literal["pass", "warn", "fail", "na"]


# ============================================================================ результаты анализа
class Evidence(BaseModel):
    quote: str
    line_start: int | None = None
    line_end: int | None = None
    location: str | None = None
    section: str | None = None
    verified: bool = False
    score: float = 0.0


class Requirement(BaseModel):
    id: str
    doc_id: str | None = None
    section: str = ""
    kind: str = "functional"
    summary: str = ""
    evidence: Evidence


class Excluded(BaseModel):
    line_start: int
    line_end: int
    location: str = ""
    section: str = ""
    reason: str = ""


class LintHit(BaseModel):
    rule: str
    title: str
    criterion: str
    line: int
    location: str
    fragment: str
    message: str


class Finding(BaseModel):
    id: str = ""
    scope: Literal["requirement", "set"]
    req_ids: list[str] = Field(default_factory=list)
    criterion: str
    severity: Severity
    defect_type: str = "other"
    title: str
    description: str
    recommendation: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["confirmed", "unverified", "rejected"] = "confirmed"
    review_note: str | None = None
    stage: str = ""


class RequirementCheck(BaseModel):
    req_id: str
    statuses: dict[str, Status]
    finding_ids: list[str] = Field(default_factory=list)
    rewrite: str | None = None


class SetCheck(BaseModel):
    criterion: str
    status: Status
    summary: str = ""
    finding_ids: list[str] = Field(default_factory=list)


class CoverageRow(BaseModel):
    req_id: str
    covered_by: list[str] = Field(default_factory=list)
    note: str = ""


class Priority(BaseModel):
    title: str
    action: str = ""
    finding_ids: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    development: Literal["ready", "conditional", "not_ready"]
    testing: Literal["ready", "conditional", "not_ready"]
    summary: str = ""
    development_rationale: str = ""
    testing_rationale: str = ""
    priorities: list[Priority] = Field(default_factory=list)


class CallInfo(BaseModel):
    stage: str
    label: str = ""
    duration_s: float = 0.0
    attempts: int = 0
    cached: bool = False
    ok: bool = True
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class RunInfo(BaseModel):
    document_title: str
    document_path: str
    document_format: str
    document_sha256: str
    document_lines: int
    document_pages: int | None = None
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    provider: str
    model: str
    models_reported: list[str] = Field(default_factory=list)
    agent_version: str
    agent_commit: str | None = None
    agent_dirty: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    llm_cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: list[CallInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Stats(BaseModel):
    requirements: int = 0
    with_doc_id: int = 0
    with_findings: int = 0
    without_findings: int = 0
    findings_total: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_criterion: dict[str, int] = Field(default_factory=dict)
    individual_findings: int = 0
    set_findings: int = 0
    unverified: int = 0
    rejected: int = 0
    coverage_ratio: float | None = None


class AnalysisResult(BaseModel):
    run: RunInfo
    requirements: list[Requirement]
    excluded: list[Excluded] = Field(default_factory=list)
    checks: list[RequirementCheck] = Field(default_factory=list)
    set_checks: list[SetCheck] = Field(default_factory=list)
    coverage: list[CoverageRow] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    lint: list[LintHit] = Field(default_factory=list)
    stats: Stats = Field(default_factory=Stats)
    verdict: Verdict | None = None
    set_overview: str = ""

    def finding(self, fid: str) -> Finding | None:
        return next((f for f in self.findings if f.id == fid), None)

    def requirement(self, rid: str) -> Requirement | None:
        return next((r for r in self.requirements if r.id == rid), None)


# ============================================================================ схемы ответов модели
class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


def _to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip().lstrip("Ll").split("-")[0].split("–")[0])
    except ValueError:
        return None


class SegItem(_Lenient):
    doc_id: str | None = None
    section: str | None = None
    kind: str = "functional"
    line_start: int | None = None
    line_end: int | None = None
    quote: str
    summary: str = ""

    _ints = field_validator("line_start", "line_end", mode="before")(lambda v: _to_int(v))

    @field_validator("doc_id", mode="before")
    @classmethod
    def _doc_id(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return None if s.lower() in ("", "-", "—", "none", "null", "нет", "n/a") else s


class SegExcluded(_Lenient):
    line_start: int | None = None
    line_end: int | None = None
    reason: str = ""

    _ints = field_validator("line_start", "line_end", mode="before")(lambda v: _to_int(v))


class SegResponse(_Lenient):
    document_title: str | None = None
    requirements: list[SegItem]
    excluded: list[SegExcluded] = Field(default_factory=list)


class LLMEvidence(_Lenient):
    quote: str
    line: int | None = None

    _ints = field_validator("line", mode="before")(lambda v: _to_int(v))

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"quote": v}
        return v


class LLMFinding(_Lenient):
    criterion: str | None = None
    severity: Severity = "major"
    title: str
    description: str = ""
    quote: str | None = None
    line: int | None = None
    related: list[LLMEvidence] = Field(default_factory=list)
    related_req_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""

    _ints = field_validator("line", mode="before")(lambda v: _to_int(v))

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: Any) -> str:
        return normalize_severity(str(v) if v is not None else None)

    @field_validator("criterion", mode="before")
    @classmethod
    def _crit(cls, v: Any) -> str | None:
        # Неизвестный код не роняет весь ответ: такое замечание будет отброшено с предупреждением.
        return normalize_criterion(str(v or ""))


class ReqCheckItem(_Lenient):
    req_id: str
    checklist: dict[str, str] = Field(default_factory=dict)
    findings: list[LLMFinding] = Field(default_factory=list)
    rewrite: str | None = None

    @field_validator("checklist", mode="before")
    @classmethod
    def _norm_checklist(cls, v: Any) -> dict[str, str]:
        out: dict[str, str] = {}
        if isinstance(v, dict):
            for k, val in v.items():
                code = normalize_criterion(str(k))
                if code:
                    out[code] = str(val).strip().lower()
        return out


class ReqCheckResponse(_Lenient):
    results: list[ReqCheckItem]


class SetAssessmentLLM(_Lenient):
    criterion: str
    status: str = "warn"
    summary: str = ""

    @field_validator("criterion", mode="before")
    @classmethod
    def _crit(cls, v: Any) -> str:
        code = normalize_criterion(str(v or ""))
        if code is None:
            raise ValueError(f"неизвестный критерий {v!r}")
        return code


class SetFindingLLM(_Lenient):
    criterion: str | None = None
    severity: Severity = "major"
    title: str
    description: str = ""
    evidence: list[LLMEvidence] = Field(default_factory=list)
    req_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""

    _sev = field_validator("severity", mode="before")(lambda v: normalize_severity(str(v) if v else None))

    @field_validator("criterion", mode="before")
    @classmethod
    def _crit(cls, v: Any) -> str | None:
        return normalize_criterion(str(v or ""))


class CoverageLLM(_Lenient):
    req_id: str
    covered_by: list[str] = Field(default_factory=list)
    note: str = ""


class SetResponse(_Lenient):
    assessments: list[SetAssessmentLLM] = Field(default_factory=list)
    findings: list[SetFindingLLM] = Field(default_factory=list)
    coverage: list[CoverageLLM] = Field(default_factory=list)
    overview: str = ""


class ReviewDecision(_Lenient):
    id: str
    decision: Literal["keep", "drop", "merge"] = "keep"
    severity: Severity | None = None
    merge_into: str | None = None
    reason: str = ""

    @field_validator("decision", mode="before")
    @classmethod
    def _dec(cls, v: Any) -> str:
        low = str(v or "keep").strip().lower()
        if low.startswith(("drop", "rej", "удал", "откл", "false")):
            return "drop"
        if low.startswith(("merge", "dup", "объед", "дубл")):
            return "merge"
        return "keep"

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: Any) -> str | None:
        return normalize_severity(str(v)) if v else None


class ReviewResponse(_Lenient):
    decisions: list[ReviewDecision] = Field(default_factory=list)


class PriorityLLM(_Lenient):
    title: str
    action: str = ""
    finding_ids: list[str] = Field(default_factory=list)


class VerdictLLM(_Lenient):
    summary: str
    development_rationale: str = ""
    testing_rationale: str = ""
    priorities: list[PriorityLLM] = Field(default_factory=list)
