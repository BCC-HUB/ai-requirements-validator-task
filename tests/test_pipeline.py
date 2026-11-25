import pytest

from reqvalidator.llm import LLMAbortError, LLMFatalError, LLMSession
from reqvalidator.loaders import load_document
from reqvalidator.pipeline import Pipeline, PipelineOptions, clean_text
from reqvalidator.version import AgentVersion
from tests.fakes import FakeClient

VERSION = AgentVersion("1.0.0", commit="abc123")


def _run(fixtures, client=None, **opts):
    client = client or FakeClient()
    doc = load_document(fixtures / "sample_requirements.md")
    options = PipelineOptions(**{"batch_size": 2, "concurrency": 2, "review": True, **opts})
    return Pipeline(LLMSession(client, sleep=lambda s: None), client.settings, options).run(
        doc, VERSION
    ), client


def test_end_to_end_segmentation_and_grounding(fixtures):
    res, _ = _run(fixtures)
    ids = [r.id for r in res.requirements]
    assert ids == [f"ТР-{i:02d}" for i in range(1, 7)]
    assert [r.doc_id for r in res.requirements][:2] == ["REQ-1", "REQ-2"]
    # Цитата, которой нет в документе, не превратилась в требование.
    assert not any("нет в документе" in r.evidence.quote for r in res.requirements)
    assert any("ненайденной цитатой" in w for w in res.run.warnings)
    # Строка 19 была упущена моделью и добавлена на этапе уточнения нарезки.
    assert any(r.evidence.line_start == 19 for r in res.requirements)
    for r in res.requirements:
        assert r.evidence.verified and r.evidence.location.startswith("строка")


def test_findings_are_grounded_numbered_and_reviewed(fixtures):
    res, _ = _run(fixtures)
    confirmed = [f for f in res.findings if f.status == "confirmed"]
    assert [f.id for f in confirmed] == [f"Д-{i:02d}" for i in range(1, len(confirmed) + 1)]
    for f in confirmed:
        assert f.evidence and all(e.verified for e in f.evidence)
        assert all(e.location for e in f.evidence)
    # Замечание с неизвестным кодом критерия отброшено, а не уронило пакет.
    assert any("неизвестным критерием" in w for w in res.run.warnings)
    assert not any(f.title == "Замечание с неизвестным критерием" for f in res.findings)
    # Английский синоним критерия («single») распознан как И5.
    assert any(f.criterion == "И5" and f.title == "Два требования в одном пункте" for f in res.findings)
    assert all(f.criterion[0] in "ИН" for f in res.findings)
    # Выдуманная цитата в замечании к набору — «не подтверждено».
    assert any(f.status == "unverified" and f.title == "Выдуманная цитата" for f in res.findings)
    # Самопроверка: одно отклонено, одно объединено, у первого повышена серьёзность.
    rejected = [f for f in res.findings if f.status == "rejected"]
    assert any("Отклонено" in (f.review_note or "") for f in rejected)
    merged = [f for f in rejected if "Объединено с замечанием Д-" in (f.review_note or "")]
    assert merged, [f.review_note for f in rejected]
    first = confirmed[0]
    assert first.severity == "critical"


def test_statuses_stats_and_verdict(fixtures):
    res, _ = _run(fixtures)
    checks = {c.req_id: c for c in res.checks}
    assert checks["ТР-01"].statuses["И3"] == "fail"
    assert checks["ТР-01"].statuses["И5"] in ("warn", "fail")
    assert checks["ТР-02"].statuses == {f"И{i}": "pass" for i in range(1, 10)}
    assert checks["ТР-02"].rewrite is None
    st = res.stats
    assert st.requirements == 6 and st.findings_total == len(
        [f for f in res.findings if f.status == "confirmed"]
    )
    assert st.coverage_ratio == pytest.approx(1 / 3, abs=0.01)
    assert res.verdict.development == "not_ready"
    # Несуществующий ID из ответа модели отфильтрован.
    assert all(i.startswith("Д-") and i != "Д-999" for p in res.verdict.priorities for i in p.finding_ids)
    set_codes = [s.criterion for s in res.set_checks]
    assert set_codes == [f"Н{i}" for i in range(1, 9)]
    assert {c.req_id for c in res.coverage} <= {r.id for r in res.requirements}


def test_truncated_batches_are_split(fixtures):
    res, client = _run(fixtures, FakeClient(truncate_batches_over=1), batch_size=4)
    assert all(any(v != "na" for v in c.statuses.values()) for c in res.checks)
    check_calls = [p for stage, p in client.calls if stage == "check"]
    assert len(check_calls) > 2


def test_invalid_json_is_repaired(fixtures):
    res, client = _run(fixtures, FakeClient(broken_first={"segment", "verdict"}))
    assert len(res.requirements) == 6
    assert sum(1 for stage, _ in client.calls if stage == "segment") == 2
    assert res.verdict.summary


def test_segmentation_failure_falls_back_to_heuristics(fixtures):
    class NoSegment(FakeClient):
        def complete(self, system, prompt, *, json_mode=True, stage=""):
            if stage == "segment":
                raise LLMFatalError("сбой")
            return super().complete(system, prompt, json_mode=json_mode, stage=stage)

    res, _ = _run(fixtures, NoSegment())
    assert len(res.requirements) >= 6
    assert any("эвристическая нарезка" in w for w in res.run.warnings)


def test_review_can_be_disabled(fixtures):
    res, client = _run(fixtures, review=False)
    assert not any(stage == "review" for stage, _ in client.calls)
    assert not any(f.status == "rejected" for f in res.findings)


def test_readiness_rules():
    from reqvalidator.models import Finding, Stats

    def f(sev, dtype):
        return Finding(
            scope="set", criterion="Н1", severity=sev, defect_type=dtype, title="t", description="d"
        )

    assert Pipeline.readiness(Stats(coverage_ratio=1.0), []) == ("ready", "ready")
    assert Pipeline.readiness(Stats(coverage_ratio=1.0), [f("minor", "structure")]) == ("ready", "ready")
    assert Pipeline.readiness(Stats(coverage_ratio=1.0), [f("major", "ambiguous")]) == (
        "conditional",
        "conditional",
    )
    assert Pipeline.readiness(Stats(coverage_ratio=1.0), [f("critical", "incomplete")]) == (
        "not_ready",
        "conditional",
    )
    assert Pipeline.readiness(Stats(coverage_ratio=0.9), [f("critical", "contradiction")]) == (
        "not_ready",
        "not_ready",
    )
    assert Pipeline.readiness(Stats(coverage_ratio=0.3), []) == ("ready", "not_ready")


def test_clean_text_strips_markdown():
    assert clean_text("| **REST type** | `GET` |") == "REST type | GET"
    assert clean_text("* **Ф-1.** Текст") == "Ф-1. Текст"
    assert clean_text("### Заголовок") == "Заголовок"


def test_merged_finding_keeps_requirement_status_and_links_target():
    from reqvalidator.models import Evidence, Finding, ReqCheckItem, Requirement

    req = Requirement(id="ТР-01", evidence=Evidence(quote="q", line_start=1, line_end=1, verified=True))
    merged = Finding(
        id="О-01",
        scope="requirement",
        req_ids=["ТР-01"],
        criterion="И8",
        severity="major",
        title="t",
        description="d",
        status="rejected",
        merged_into="Д-05",
    )
    target = Finding(
        id="Д-05",
        scope="set",
        req_ids=["ТР-01", "ТР-07"],
        criterion="Н2",
        severity="critical",
        title="t",
        description="d",
    )
    item = ReqCheckItem(req_id="ТР-01", checklist={"И8": "fail"}, rewrite="Новая формулировка")
    (check,) = Pipeline._requirement_statuses([req], {"ТР-01": item}, [merged, target])
    assert check.statuses["И8"] == "fail"
    assert check.finding_ids == ["Д-05"]
    assert check.rewrite == "Новая формулировка"


class _FailingStage(FakeClient):
    def __init__(self, stages, exc, **kw):
        super().__init__(**kw)
        self.fail_stages, self.exc = set(stages), exc

    def complete(self, system, prompt, *, json_mode=True, stage=""):
        if stage in self.fail_stages:
            raise self.exc
        return super().complete(system, prompt, json_mode=json_mode, stage=stage)


def test_account_level_error_aborts_the_whole_run(fixtures):
    client = _FailingStage({"check"}, LLMAbortError("на счёте провайдера недостаточно средств (HTTP 402)"))
    with pytest.raises(LLMAbortError, match="недостаточно средств"):
        _run(fixtures, client)


def test_mostly_failed_checks_do_not_produce_a_false_all_clear(fixtures):
    client = _FailingStage({"check"}, LLMFatalError("модель вернула мусор"))
    with pytest.raises(LLMFatalError, match="отчёт не сформирован"):
        _run(fixtures, client)


def test_failed_set_stage_marks_run_incomplete_and_blocks_ready(fixtures):
    res, _ = _run(fixtures, _FailingStage({"set-structure"}, LLMFatalError("сбой")))
    assert any("structure" in x for x in res.verdict.incomplete)
    from reqvalidator.report import render_markdown

    assert "Проверка выполнена не полностью" in render_markdown(res)


def test_incomplete_run_is_never_ready():
    from reqvalidator.models import Stats

    assert Pipeline.readiness(Stats(coverage_ratio=1.0), [], incomplete=True) == (
        "conditional",
        "conditional",
    )
