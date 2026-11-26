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


def test_segmentation_guards_against_weak_model_output(fixtures):
    """Придуманные ID, вложенные дубли и повторы исключений (так отвечала локальная модель 3B)."""
    from reqvalidator.models import SegExcluded, SegItem
    from reqvalidator.pipeline import _literal_doc_id

    doc = load_document(fixtures / "sample_requirements.md")
    pipe = Pipeline(LLMSession(FakeClient(), sleep=lambda s: None), FakeClient().settings)
    pipe.warnings = []
    items = [
        SegItem(
            doc_id="REQ-1",
            line_start=7,
            quote="REQ-1. Сотрудник бронирует переговорную на интервал от 15 "
            "минут до 4 часов. Бронирование подтверждается быстро.",
        ),
        SegItem(doc_id=None, line_start=7, quote="Бронирование подтверждается быстро."),  # фрагмент REQ-1
        SegItem(
            doc_id="REQ-4",
            line_start=24,
            line_end=26,  # такого ID в документе нет
            quote="Исходные условия | Действие | Ожидаемый результат | Свободная переговорная | "
            "Забронировать на 30 минут | Бронь создана",
        ),
        SegItem(
            doc_id=None,
            line_start=26,
            quote="Свободная переговорная | Забронировать на 30 минут | Бронь создана",
        ),
    ]
    reqs = pipe._materialize(doc, items)
    assert [(r.doc_id, r.evidence.line_start) for r in reqs] == [("REQ-1", 7), (None, 26)]
    assert any("REQ-4" in w for w in pipe.warnings)
    excluded = pipe._excluded(doc, [SegExcluded(line_start=34, line_end=36, reason="x")] * 5)
    assert len(excluded) == 1

    assert _literal_doc_id("Ф-1", "Ф-1. Текст", "ф-1. текст")
    assert _literal_doc_id("Логика работы метода, п. 4", "4. Заменить настройку", "...")
    assert _literal_doc_id("Статус 401", "**401.** Ответ в случае", "...")
    assert not _literal_doc_id("Статус 401", "4010 строк", "...")
    assert not _literal_doc_id("REQ-4", "Критерии приемки", "req-1 req-2")
    assert _literal_doc_id("Статус 404", "Для чужой карты возвращается 404", "...", "Для чужой\n**404**\n{}")


def test_over_excluded_requirements_are_reconsidered(fixtures):
    """Модель 7B объявила НФТ и критерии приёмки «метаданными» — агент отправляет их на пересмотр."""

    class OverExcluding(FakeClient):
        def _segment(self, prompt, stage):
            data = super()._segment(prompt, stage)
            data["requirements"] = data["requirements"][:2]  # только REQ-1 и REQ-2
            data["excluded"] = [{"line_start": 10, "line_end": 36, "reason": "Метаданные задачи"}]
            return data

        def _coverage(self, prompt, stage):
            assert "похожи на требования" in prompt
            assert "L12|" in prompt and "L26|" in prompt
            return {
                "requirements": [
                    {
                        "doc_id": None,
                        "kind": "nonfunctional",
                        "line_start": 12,
                        "quote": "Сервис должен быть удобным и надёжным и т.д.",
                        "summary": "НФТ",
                    },
                    {
                        "doc_id": None,
                        "kind": "acceptance",
                        "line_start": 26,
                        "quote": "Свободная переговорная | Забронировать на 30 минут | Бронь создана",
                        "summary": "КП",
                    },
                ],
                "excluded": [],
            }

    res, _ = _run(fixtures, OverExcluding())
    lines = {r.evidence.line_start for r in res.requirements}
    assert {7, 8, 12, 26} <= lines
    for e in res.excluded:  # исключения больше не перекрывают требования, причина сохранена
        assert not any(e.line_start <= n <= e.line_end for n in lines)
    assert any(e.reason == "Метаданные задачи" for e in res.excluded)


def test_nested_exclusions_are_collapsed(fixtures):
    from reqvalidator.models import SegExcluded
    from reqvalidator.pipeline import _trim_excluded

    doc = load_document(fixtures / "sample_requirements.md")
    pipe = Pipeline(LLMSession(FakeClient(), sleep=lambda s: None), FakeClient().settings)
    raw = [
        SegExcluded(line_start=15, line_end=21, reason="API"),
        SegExcluded(line_start=18, line_end=19, reason="x"),
    ]
    out = _trim_excluded(doc, pipe._excluded(doc, raw), [])
    assert [(e.line_start, e.line_end, e.reason) for e in out] == [(15, 21, "API")]


def test_status_label_above_code_block_counts_as_document_id(tmp_path):
    from reqvalidator.models import SegItem

    path = tmp_path / "api.md"
    path.write_text(
        '# API\n\n## Статусы\n\n**404**\n```json\n{"message":"Карта не найдена"}\n```\n', encoding="utf-8"
    )
    doc = load_document(path)
    pipe = Pipeline(LLMSession(FakeClient(), sleep=lambda s: None), FakeClient().settings)
    pipe.warnings = []
    (req,) = pipe._materialize(
        doc, [SegItem(doc_id="Статус 404", line_start=7, quote='{"message":"Карта не найдена"}')]
    )
    assert req.doc_id == "Статус 404" and not pipe.warnings
