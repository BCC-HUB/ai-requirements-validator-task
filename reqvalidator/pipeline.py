"""Конвейер проверки: нарезка → проверка требований → проверка набора → самопроверка → итог."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from reqvalidator import prompts
from reqvalidator.config import Settings
from reqvalidator.criteria import (
    BY_CODE,
    CRITERION_DEFECT_TYPE,
    INDIVIDUAL,
    SET,
    SEVERITY_ORDER,
)
from reqvalidator.document import Document, normalize
from reqvalidator.lint import run_lint
from reqvalidator.llm import LLMError, LLMFatalError, LLMSession, LLMTruncatedError
from reqvalidator.models import (
    AnalysisResult,
    CallInfo,
    CoverageRow,
    Evidence,
    Excluded,
    Finding,
    LintHit,
    Priority,
    ReqCheckItem,
    ReqCheckResponse,
    Requirement,
    RequirementCheck,
    ReviewResponse,
    RunInfo,
    SegItem,
    SegResponse,
    SetCheck,
    SetResponse,
    Stats,
    Verdict,
    VerdictLLM,
)
from reqvalidator.version import AgentVersion

log = logging.getLogger("reqvalidator")

_SET_TO_IND = {"Н1": "И4", "Н2": "И8", "Н3": "И6", "Н4": "И3", "Н5": "И7", "Н6": "И9", "Н7": "И9", "Н8": "И9"}
_IND_TO_SET = {
    "И1": "Н1",
    "И2": "Н7",
    "И3": "Н4",
    "И4": "Н1",
    "И5": "Н7",
    "И6": "Н3",
    "И7": "Н5",
    "И8": "Н2",
    "И9": "Н8",
}
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_MD_MARKUP = re.compile(r"\*\*|__|`")
_COVERAGE_KINDS = {"functional", "api", "data", "integration", "nonfunctional", "ui", "process"}


@dataclass
class PipelineOptions:
    batch_size: int = 6
    concurrency: int = 1
    review: bool = True


def clean_text(text: str) -> str:
    """Убрать Markdown-разметку из строки документа для показа в отчёте."""
    out = []
    for line in text.splitlines():
        s = _MD_MARKUP.sub("", line).strip()
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"^[*-]\s+", "", s)
        if s.startswith("|") and s.endswith("|"):
            s = " | ".join(c.strip() for c in s.strip("|").split("|"))
        out.append(s)
    return re.sub(r"\s+", " ", " ".join(x for x in out if x)).strip()


class Pipeline:
    def __init__(self, session: LLMSession, settings: Settings, options: PipelineOptions | None = None):
        self.session = session
        self.settings = settings
        self.opt = options or PipelineOptions(
            batch_size=settings.batch_size, concurrency=settings.concurrency, review=settings.review
        )
        self.warnings: list[str] = []
        self._temp_ids: dict[str, Finding] = {}

    # ================================================================== запуск
    def run(self, doc: Document, version: AgentVersion) -> AnalysisResult:
        t0 = time.monotonic()
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.warnings = list(doc.warnings)
        self.incomplete: list[str] = []
        numbered = doc.numbered_text()
        if len(numbered) > 250_000:
            self.warnings.append(
                "Документ очень большой: часть контекста может не поместиться в окно модели."
            )

        log.info("[1/7] Статические проверки текста")
        lint = run_lint(doc)

        log.info("[2/7] Нарезка требований (LLM)")
        requirements, excluded = self._segment(doc, numbered)
        log.info("      выделено требований: %d, исключено фрагментов: %d", len(requirements), len(excluded))

        log.info("[3/7] Проверка каждого требования (LLM, пакетами по %d)", self.opt.batch_size)
        checklists, findings = self._check_requirements(doc, numbered, requirements, lint)

        log.info("[4/7] Проверка набора требований целиком (LLM)")
        assessments, set_findings, coverage, overview = self._check_set(doc, numbered, requirements, lint)
        findings += set_findings

        if self.opt.review and findings:
            log.info("[5/7] Самопроверка замечаний (LLM)")
            self._review(doc, numbered, requirements, findings)
        else:
            log.info("[5/7] Самопроверка замечаний пропущена")

        log.info("[6/7] Сводка и статусы")
        self._assign_ids(findings, requirements)
        checks = self._requirement_statuses(requirements, checklists, findings)
        set_checks = self._set_statuses(assessments, findings)
        stats = self._stats(requirements, findings, coverage, checks)

        log.info("[7/7] Итоговое заключение (LLM)")
        verdict = self._verdict(doc, requirements, findings, stats, coverage)

        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        run = RunInfo(
            document_title=doc.title,
            document_path=str(doc.path),
            document_format=doc.format_name,
            document_sha256=doc.sha256,
            document_lines=doc.line_count,
            document_pages=doc.pages,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=round(time.monotonic() - t0, 1),
            provider=self.settings.provider,
            model=self.settings.model_label,
            models_reported=list(self.session.models_seen),
            agent_version=version.label,
            agent_commit=version.commit,
            agent_dirty=version.dirty,
            settings={
                "temperature": self.settings.temperature,
                "endpoint": urlparse(self.settings.base_url).netloc if self.settings.base_url else None,
                "effort": self.settings.effort,
                "max_output_tokens": self.settings.max_output_tokens,
                "batch_size": self.opt.batch_size,
                "concurrency": self.opt.concurrency,
                "review": self.opt.review,
            },
            llm_calls=self.session.total_calls,
            llm_cached_calls=self.session.cached_calls,
            input_tokens=self.session.input_tokens,
            output_tokens=self.session.output_tokens,
            calls=[
                CallInfo(
                    stage=r.stage,
                    label=r.label,
                    duration_s=round(r.duration_s, 1),
                    attempts=r.attempts,
                    cached=r.cached,
                    ok=r.ok,
                    input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens,
                    error=r.error,
                )
                for r in self.session.records
            ],
            warnings=list(dict.fromkeys(self.warnings + self.session.client.switches)),
        )
        return AnalysisResult(
            run=run,
            requirements=requirements,
            excluded=excluded,
            checks=checks,
            set_checks=set_checks,
            coverage=coverage,
            findings=findings,
            lint=lint,
            stats=stats,
            verdict=verdict,
            set_overview=overview,
        )

    # ================================================================== общие помощники
    def _ground(
        self,
        doc: Document,
        quote: str | None,
        near: int | None = None,
        fallback: tuple[int, int] | None = None,
    ) -> Evidence:
        q = (quote or "").strip().strip("«»\"' ")
        match = doc.locate(q, near=near) if q else None
        if match:
            start, end = match.line_start, match.line_end
            shown = clean_text(q)
            if not match.exact:
                window = clean_text(doc.text_of(start, end))
                if len(window) <= max(len(shown) * 2.5, 160):
                    shown = window
            return Evidence(
                quote=shown,
                line_start=start,
                line_end=end,
                location=doc.location_label(start, end),
                section=doc.section_of(start),
                verified=True,
                score=match.score,
            )
        if fallback:
            start, end = fallback
            return Evidence(
                quote=clean_text(doc.text_of(start, end)),
                line_start=start,
                line_end=end,
                location=doc.location_label(start, end),
                section=doc.section_of(start),
                verified=True,
                score=0.0,
            )
        return Evidence(quote=clean_text(q), verified=False)

    def _map(self, fn, items: list, workers: int | None = None) -> list:
        workers = max(1, min(workers or self.opt.concurrency, len(items) or 1))
        if workers == 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    @staticmethod
    def _rid_normalizer(requirements: list[Requirement]):
        by_num = {}
        for r in requirements:
            m = re.search(r"(\d+)", r.id)
            if m:
                by_num[int(m.group(1))] = r.id
        valid = {r.id for r in requirements}

        def norm(value: str | None) -> str | None:
            if not value:
                return None
            v = value.strip()
            if v in valid:
                return v
            m = re.search(r"(\d+)", v)
            if m and re.match(r"^\s*(ТР|TP|TR|T|Т|R|REQ)?\s*-?\s*\d+\s*$", v, re.I):
                return by_num.get(int(m.group(1)))
            return None

        return norm

    @staticmethod
    def _catalog(requirements: list[Requirement]) -> str:
        return "\n".join(
            f"{r.id} [{r.doc_id or r.locator or '—'}] ({r.evidence.location}) {r.kind}: {r.summary}"
            for r in requirements
        )

    @staticmethod
    def _hints(lint: list[LintHit], ranges: list[tuple[int, int]] | None = None) -> str:
        rows = []
        for h in lint:
            if ranges is None or any(a <= h.line <= b for a, b in ranges):
                rows.append(f"L{h.line} [{h.rule}, {h.criterion}] «{h.fragment[:80]}» — {h.title}")
        return "\n".join(rows)

    # ================================================================== 2. нарезка
    def _segment(self, doc: Document, numbered: str) -> tuple[list[Requirement], list[Excluded]]:
        try:
            resp = self.session.ask_json(
                "segment", prompts.SYSTEM, prompts.segment_prompt(numbered), SegResponse, label="документ"
            )
            items, excluded_raw = list(resp.requirements), list(resp.excluded)
        except LLMError as exc:
            self.warnings.append(f"Нарезка моделью не удалась ({exc}); применена эвристическая нарезка.")
            self.incomplete.append("нарезка выполнена эвристикой без модели")
            log.warning("Нарезка моделью не удалась: %s — использую эвристику", exc)
            items, excluded_raw = _heuristic_segmentation(doc), []

        reqs = self._materialize(doc, items)
        excluded = self._excluded(doc, excluded_raw)

        uncovered = _uncovered_lines(doc, reqs, excluded)
        # Слабые модели исключают настоящие требования («это метаданные»); такие строки пересматриваются.
        reconsider = _suspicious_exclusions(doc, excluded)
        todo = len(uncovered) + len(reconsider)
        if todo and todo <= 150 and items:
            log.info(
                "      строк вне нарезки: %d, исключённых строк на пересмотр: %d — уточняю у модели",
                len(uncovered),
                len(reconsider),
            )
            try:
                extra = self.session.ask_json(
                    "segment-coverage",
                    prompts.SYSTEM,
                    prompts.coverage_repair_prompt(
                        numbered,
                        [(n, doc.line(n).text) for n in uncovered],
                        "\n".join(f"{r.id} {r.evidence.location}: {r.summary}" for r in reqs),
                        [(n, doc.line(n).text) for n in reconsider],
                    ),
                    SegResponse,
                    label=f"{todo} строк",
                )
                reqs = self._materialize(doc, items + list(extra.requirements))
                excluded = self._excluded(doc, excluded_raw + list(extra.excluded))
            except LLMError as exc:
                self.warnings.append(f"Уточнение нарезки не выполнено: {exc}")
        excluded = _trim_excluded(doc, excluded, reqs)
        # Всё, что так и не попало в нарезку, явно фиксируем в отчёте.
        for start, end in _ranges(_uncovered_lines(doc, reqs, excluded)):
            excluded.append(
                Excluded(
                    line_start=start,
                    line_end=end,
                    location=doc.location_label(start, end),
                    section=doc.section_of(start),
                    reason="Не отнесено к требованиям при нарезке",
                )
            )
        excluded.sort(key=lambda e: e.line_start)
        if not reqs:
            raise LLMError("В документе не выделено ни одного требования")
        return reqs, excluded

    def _materialize(self, doc: Document, items: list[SegItem]) -> list[Requirement]:
        doc_norm = normalize(doc.text)
        rows: list[tuple[int, int, SegItem, Evidence]] = []
        seen: set[tuple[int, int, str]] = set()
        for it in items:
            near = it.line_start
            fallback = None
            if it.line_start:
                fallback = (doc.clamp(it.line_start), doc.clamp(it.line_end or it.line_start))
            ev = self._ground(doc, it.quote, near=near, fallback=fallback)
            if not ev.verified or ev.line_start is None:
                self.warnings.append(f"Пропущено требование с ненайденной цитатой: «{it.quote[:80]}»")
                continue
            key = (ev.line_start, ev.line_end or ev.line_start, normalize(ev.quote)[:60])
            if key in seen:
                continue
            seen.add(key)
            if it.doc_id and not _literal_doc_id(it.doc_id, ev.quote, doc_norm):
                # Модель придумала идентификатор, которого нет в документе: в отчёт его не переносим.
                self.warnings.append(f"Идентификатор «{it.doc_id}» не найден в документе и не использован")
                it = it.model_copy(update={"doc_id": None})
            rows.append((ev.line_start, ev.line_end or ev.line_start, it, ev))
        rows = _drop_contained(rows)
        rows.sort(key=lambda r: (r[0], r[1]))
        width = 3 if len(rows) > 99 else 2
        reqs = []
        for i, (_, _, it, ev) in enumerate(rows, 1):
            kind = (it.kind or "other").strip().lower()
            reqs.append(
                Requirement(
                    id=f"ТР-{i:0{width}d}",
                    doc_id=it.doc_id,
                    locator=(it.locator or "").strip() or None,
                    section=(it.section or ev.section or "").strip(),
                    kind=kind,
                    summary=(it.summary or ev.quote[:80]).strip(),
                    evidence=ev,
                )
            )
        return reqs

    @staticmethod
    def _excluded(doc: Document, raw) -> list[Excluded]:
        out = []
        seen: set[tuple[int, int]] = set()
        for e in raw:
            if not e.line_start:
                continue
            start = doc.clamp(e.line_start)
            end = doc.clamp(e.line_end or e.line_start)
            if end < start:
                start, end = end, start
            if (start, end) in seen:
                continue
            seen.add((start, end))
            out.append(
                Excluded(
                    line_start=start,
                    line_end=end,
                    location=doc.location_label(start, end),
                    section=doc.section_of(start),
                    reason=e.reason or "Не требование",
                )
            )
        return out

    # ================================================================== 3. проверка требований
    def _check_requirements(
        self, doc: Document, numbered: str, reqs: list[Requirement], lint: list[LintHit]
    ) -> tuple[dict[str, ReqCheckItem], list[Finding]]:
        catalog = self._catalog(reqs)
        size = max(1, self.opt.batch_size)
        batches = [reqs[i : i + size] for i in range(0, len(reqs), size)]
        done = {"n": 0}

        def run_batch(batch: list[Requirement]) -> dict[str, ReqCheckItem]:
            result = self._check_batch(doc, numbered, batch, catalog, lint, reqs)
            done["n"] += 1
            log.info("      пакет %d/%d: %s–%s", done["n"], len(batches), batch[0].id, batch[-1].id)
            return result

        merged: dict[str, ReqCheckItem] = {}
        for part in self._map(run_batch, batches):
            merged.update(part)
        by_id = {r.id: r for r in reqs}
        norm = self._rid_normalizer(reqs)
        findings: list[Finding] = []
        for rid, item in merged.items():
            req = by_id[rid]
            rng = (req.evidence.line_start or 1, req.evidence.line_end or req.evidence.line_start or 1)
            for f in item.findings:
                if f.criterion is None:
                    self.warnings.append(
                        f"Отброшено замечание к {rid} с неизвестным критерием: «{f.title[:60]}»"
                    )
                    continue
                criterion = _SET_TO_IND.get(f.criterion, f.criterion)
                primary = self._ground(doc, f.quote, near=f.line or rng[0], fallback=rng)
                related = [self._ground(doc, e.quote, near=e.line) for e in f.related]
                related = [e for e in related if e.verified and e.line_start != primary.line_start]
                extra_ids = [x for x in (norm(r) for r in f.related_req_ids) if x and x != rid]
                findings.append(
                    Finding(
                        scope="requirement",
                        req_ids=[rid, *dict.fromkeys(extra_ids)],
                        criterion=criterion,
                        severity=f.severity,
                        defect_type=CRITERION_DEFECT_TYPE.get(criterion, "other"),
                        title=f.title.strip(),
                        description=f.description.strip(),
                        recommendation=f.recommendation.strip(),
                        evidence=[primary, *related],
                        stage="check",
                    )
                )
        missing = [r.id for r in reqs if r.id not in merged]
        if missing:
            self.warnings.append("Не получен результат проверки для: " + ", ".join(missing))
            self.incomplete.append(f"не проверено требований: {len(missing)} из {len(reqs)}")
        if len(missing) * 2 > len(reqs):
            # Отчёт по большинству непроверенных требований выглядел бы как «замечаний нет» — это ложный вывод.
            raise LLMFatalError(
                f"модель не вернула результаты проверки для {len(missing)} из {len(reqs)} требований; "
                "отчёт не сформирован, чтобы не выдать ложный вывод. Подробности — в журнале выше."
            )
        return merged, findings

    def _check_batch(
        self,
        doc: Document,
        numbered: str,
        batch: list[Requirement],
        catalog: str,
        lint: list[LintHit],
        all_reqs: list[Requirement],
    ) -> dict[str, ReqCheckItem]:
        ranges = [(r.evidence.line_start or 0, r.evidence.line_end or 0) for r in batch]
        payload = [
            {
                "id": r.id,
                "doc_id": r.doc_id,
                "kind": r.kind,
                "section": r.section,
                "line_start": r.evidence.line_start,
                "line_end": r.evidence.line_end,
                "quote": r.evidence.quote,
            }
            for r in batch
        ]
        prompt = prompts.check_prompt(numbered, payload, catalog, self._hints(lint, ranges))
        label = f"{batch[0].id}–{batch[-1].id}" if len(batch) > 1 else batch[0].id
        try:
            resp = self.session.ask_json("check", prompts.SYSTEM, prompt, ReqCheckResponse, label=label)
        except LLMTruncatedError:
            if len(batch) == 1:
                self.warnings.append(f"Ответ модели по {batch[0].id} обрезан — требование не проверено")
                return {}
            mid = len(batch) // 2
            log.info("      ответ обрезан — делю пакет %s пополам", label)
            out = self._check_batch(doc, numbered, batch[:mid], catalog, lint, all_reqs)
            out.update(self._check_batch(doc, numbered, batch[mid:], catalog, lint, all_reqs))
            return out
        except LLMError as exc:
            if len(batch) > 1:
                log.warning("Пакет %s не обработан (%s) — проверяю требования по одному", label, exc)
                out = {}
                for r in batch:
                    out.update(self._check_batch(doc, numbered, [r], catalog, lint, all_reqs))
                return out
            self.warnings.append(f"Проверка {label} не выполнена: {exc}")
            return {}
        norm = self._rid_normalizer(all_reqs)
        wanted = {r.id for r in batch}
        out: dict[str, ReqCheckItem] = {}
        for item in resp.results:
            rid = norm(item.req_id)
            if rid in wanted and rid not in out:
                out[rid] = item
        missing = [r for r in batch if r.id not in out]
        if missing and len(batch) > 1:
            for r in missing:
                out.update(self._check_batch(doc, numbered, [r], catalog, lint, all_reqs))
        return out

    # ================================================================== 4. проверка набора
    def _check_set(self, doc: Document, numbered: str, reqs: list[Requirement], lint: list[LintHit]):
        catalog = self._catalog(reqs)
        kinds = Counter(r.kind for r in reqs)
        with_id = sum(1 for r in reqs if r.doc_id)
        stats = (
            f"Выделено требований: {len(reqs)}; с идентификатором в документе: {with_id}; "
            f"без идентификатора (присвоен агентом): {len(reqs) - with_id}.\n"
            "По типам: " + ", ".join(f"{k} — {v}" for k, v in kinds.most_common()) + "."
        )
        hints = self._hints(lint)
        norm = self._rid_normalizer(reqs)

        def run_stage(stage: str):
            try:
                return stage, self.session.ask_json(
                    f"set-{stage}",
                    prompts.SYSTEM,
                    prompts.set_prompt(stage, numbered, catalog, stats, hints),
                    SetResponse,
                    label=stage,
                )
            except LLMError as exc:
                self.warnings.append(f"Проверка набора ({stage}) не выполнена: {exc}")
                self.incomplete.append(f"не выполнен проход проверки набора «{stage}»")
                return stage, None

        assessments: dict[str, tuple[str, str]] = {}
        findings: list[Finding] = []
        coverage: list[CoverageRow] = []
        overview = ""
        acceptance = {r.id for r in reqs if r.kind == "acceptance"}
        for stage, resp in self._map(run_stage, list(prompts.SET_STAGES)):
            if resp is None:
                continue
            if resp.overview and not overview:
                overview = resp.overview.strip()
            for a in resp.assessments:
                code = _IND_TO_SET.get(a.criterion, a.criterion)
                assessments.setdefault(code, (a.status.strip().lower(), a.summary.strip()))
            for f in resp.findings:
                if f.criterion is None:
                    self.warnings.append(
                        f"Отброшено замечание к набору с неизвестным критерием: «{f.title[:60]}»"
                    )
                    continue
                code = _IND_TO_SET.get(f.criterion, f.criterion)
                evidence = [self._ground(doc, e.quote, near=e.line) for e in f.evidence]
                verified = [e for e in evidence if e.verified]
                req_ids = list(dict.fromkeys(x for x in (norm(r) for r in f.req_ids) if x))
                if not verified and req_ids:
                    first = next(r for r in reqs if r.id == req_ids[0])
                    verified = [first.evidence]
                findings.append(
                    Finding(
                        scope="set",
                        req_ids=req_ids,
                        criterion=code,
                        severity=f.severity,
                        defect_type=CRITERION_DEFECT_TYPE.get(code, "other"),
                        title=f.title.strip(),
                        description=f.description.strip(),
                        recommendation=f.recommendation.strip(),
                        evidence=verified or evidence,
                        status="confirmed" if verified else "unverified",
                        review_note=None if verified else "Цитаты не найдены в документе",
                        stage=f"set-{stage}",
                    )
                )
            for row in resp.coverage:
                rid = norm(row.req_id)
                if not rid or any(c.req_id == rid for c in coverage):
                    continue
                covered = [x for x in (norm(c) for c in row.covered_by) if x and x in acceptance]
                coverage.append(CoverageRow(req_id=rid, covered_by=covered, note=row.note.strip()))
        order = {r.id: i for i, r in enumerate(reqs)}
        coverage.sort(key=lambda c: order.get(c.req_id, 0))
        return assessments, findings, coverage, overview

    # ================================================================== 5. самопроверка
    def _review(self, doc: Document, numbered: str, reqs: list[Requirement], findings: list[Finding]) -> None:
        active = [f for f in findings if f.status == "confirmed"]
        if not active:
            return
        temp = {f"F{i:03d}": f for i, f in enumerate(active, 1)}
        by_req = {r.id: r for r in reqs}

        def line(tid: str, f: Finding) -> str:
            req = (
                ", ".join(
                    f"{rid} [{by_req[rid].doc_id or by_req[rid].locator or '—'}]"
                    for rid in f.req_ids[:3]
                    if rid in by_req
                )
                or "набор"
            )
            quotes = "; ".join(f"«{e.quote[:160]}» ({e.location or '?'})" for e in f.evidence[:3])
            return (
                f"{tid} | {req} | {f.criterion} | {f.severity} | {f.title} — {f.description[:400]} | "
                f"рекомендация: {f.recommendation[:200]} | цитаты: {quotes}"
            )

        ids = list(temp)
        # Одним запросом, чтобы рецензент видел дубликаты между требованиями и уровнями проверки;
        # делим только очень большие списки.
        chunks = [ids[i : i + 300] for i in range(0, len(ids), 300)]

        def run_chunk(chunk: list[str]):
            block = "\n".join(line(t, temp[t]) for t in chunk)
            try:
                return self.session.ask_json(
                    "review",
                    prompts.SYSTEM,
                    prompts.review_prompt(numbered, block, len(chunk)),
                    ReviewResponse,
                    label=f"{chunk[0]}–{chunk[-1]}",
                )
            except LLMError as exc:
                self.warnings.append(f"Самопроверка части замечаний не выполнена: {exc}")
                return None

        decisions = {}
        for resp in self._map(run_chunk, chunks):
            if resp is None:
                continue
            for d in resp.decisions:
                key = d.id.strip().upper().replace("Ф", "F")
                if key in temp:
                    decisions[key] = d
        # Сначала исключения, затем объединения — чтобы не объединять с исключённым.
        for tid, d in decisions.items():
            if d.decision == "drop":
                temp[tid].status = "rejected"
                temp[tid].review_note = "Отклонено при самопроверке: " + (d.reason or "ложное срабатывание")
        for tid, d in decisions.items():
            f = temp[tid]
            if d.decision == "merge":
                target_id = (d.merge_into or "").strip().upper()
                target = temp.get(target_id)
                if target is None or target is f or target.status != "confirmed":
                    continue
                f.status = "rejected"
                f.merged_into = target_id
                f.review_note = f"Объединено с замечанием {target_id}: " + (d.reason or "дубликат")
                for ev in f.evidence:
                    if all(ev.line_start != e.line_start for e in target.evidence) and ev.verified:
                        target.evidence.append(ev)
                for rid in f.req_ids:
                    if rid not in target.req_ids:
                        target.req_ids.append(rid)
            if f.status == "confirmed" and d.severity and d.severity != f.severity:
                f.review_note = (
                    f"Серьёзность изменена при самопроверке: {f.severity} → {d.severity}. {d.reason}".strip()
                )
                f.severity = d.severity
        # Ссылки «объединено с F012» заменим на итоговые ID после нумерации.
        self._temp_ids = temp

    # ================================================================== 6. нумерация и статусы
    def _assign_ids(self, findings: list[Finding], reqs: list[Requirement]) -> None:
        order = {r.id: i for i, r in enumerate(reqs)}
        crit_order = {c.code: i for i, c in enumerate(INDIVIDUAL + SET)}

        def key(f: Finding):
            if f.scope == "requirement":
                return (
                    0,
                    order.get(f.req_ids[0], 10_000) if f.req_ids else 10_000,
                    SEVERITY_ORDER[f.severity],
                    crit_order.get(f.criterion, 99),
                )
            return (1, crit_order.get(f.criterion, 99), SEVERITY_ORDER[f.severity], 0)

        confirmed = sorted((f for f in findings if f.status == "confirmed"), key=key)
        others = sorted((f for f in findings if f.status != "confirmed"), key=key)
        width = 3 if len(confirmed) > 99 else 2
        for i, f in enumerate(confirmed, 1):
            f.id = f"Д-{i:0{width}d}"
        for i, f in enumerate(others, 1):
            f.id = f"О-{i:02d}"
        temp = self._temp_ids
        for f in findings:
            if f.merged_into in temp:
                f.merged_into = temp[f.merged_into].id
            if f.review_note and "замечанием F" in f.review_note:
                m = re.search(r"замечанием (F\d+)", f.review_note)
                if m and m.group(1) in temp:
                    f.review_note = f.review_note.replace(m.group(1), temp[m.group(1)].id)
        findings.sort(key=lambda f: (f.status != "confirmed", key(f)))

    @staticmethod
    def _requirement_statuses(
        reqs: list[Requirement], checklists: dict[str, ReqCheckItem], findings: list[Finding]
    ) -> list[RequirementCheck]:
        checks = []
        by_id = {f.id: f for f in findings}
        for r in reqs:
            item = checklists.get(r.id)
            own = [f for f in findings if f.scope == "requirement" and f.req_ids and f.req_ids[0] == r.id]
            # Замечание, объединённое с другим, по-прежнему означает нарушение критерия у этого требования:
            # статус берём с серьёзностью итогового замечания, ссылку ведём на него.
            effective: list[tuple[str, Finding]] = []
            for f in own:
                if f.status == "confirmed":
                    effective.append((f.criterion, f))
                elif (
                    f.merged_into and by_id.get(f.merged_into) and by_id[f.merged_into].status == "confirmed"
                ):
                    effective.append((f.criterion, by_id[f.merged_into]))
            statuses: dict[str, str] = {}
            for c in INDIVIDUAL:
                conf = [t for code, t in effective if code == c.code]
                rejected = [f for f in own if f.criterion == c.code and f.status != "confirmed"]
                mark = (item.checklist.get(c.code, "") if item else "").lower()
                if conf:
                    worst = min(SEVERITY_ORDER[f.severity] for f in conf)
                    statuses[c.code] = "fail" if worst <= SEVERITY_ORDER["major"] else "warn"
                elif item is None or mark.startswith("n"):
                    statuses[c.code] = "na"
                elif mark.startswith("f") and not rejected:
                    statuses[c.code] = "warn"
                else:
                    statuses[c.code] = "pass"
            finding_ids = list(dict.fromkeys(t.id for _, t in effective))
            checks.append(
                RequirementCheck(
                    req_id=r.id,
                    statuses=statuses,
                    finding_ids=finding_ids,
                    rewrite=(item.rewrite.strip() if item and item.rewrite else None)
                    if finding_ids
                    else None,
                )
            )
        return checks

    @staticmethod
    def _set_statuses(assessments: dict[str, tuple[str, str]], findings: list[Finding]) -> list[SetCheck]:
        out = []
        for c in SET:
            conf = [
                f for f in findings if f.scope == "set" and f.criterion == c.code and f.status == "confirmed"
            ]
            llm_status, summary = assessments.get(c.code, ("", ""))
            if conf:
                worst = min(SEVERITY_ORDER[f.severity] for f in conf)
                status = "fail" if worst <= SEVERITY_ORDER["major"] else "warn"
            elif llm_status.startswith("f") or llm_status.startswith("w"):
                status = "warn"
            elif llm_status.startswith("o") or llm_status.startswith("p"):
                status = "pass"
            else:
                status = "na" if not summary else "pass"
            out.append(
                SetCheck(criterion=c.code, status=status, summary=summary, finding_ids=[f.id for f in conf])
            )
        return out

    @staticmethod
    def _stats(
        reqs: list[Requirement],
        findings: list[Finding],
        coverage: list[CoverageRow],
        checks: list[RequirementCheck],
    ) -> Stats:
        conf = [f for f in findings if f.status == "confirmed"]
        with_findings = {c.req_id for c in checks if c.finding_ids}
        considered = [
            c for c in coverage if any(r.id == c.req_id and r.kind in _COVERAGE_KINDS for r in reqs)
        ]
        ratio = (sum(1 for c in considered if c.covered_by) / len(considered)) if considered else None
        return Stats(
            requirements=len(reqs),
            with_doc_id=sum(1 for r in reqs if r.doc_id),
            with_findings=len(with_findings),
            without_findings=len(reqs) - len(with_findings),
            findings_total=len(conf),
            by_severity={s: sum(1 for f in conf if f.severity == s) for s in ("critical", "major", "minor")},
            by_criterion=dict(Counter(f.criterion for f in conf)),
            individual_findings=sum(1 for f in conf if f.scope == "requirement"),
            set_findings=sum(1 for f in conf if f.scope == "set"),
            unverified=sum(1 for f in findings if f.status == "unverified"),
            rejected=sum(1 for f in findings if f.status == "rejected"),
            coverage_ratio=round(ratio, 3) if ratio is not None else None,
        )

    # ================================================================== 7. итог
    @staticmethod
    def readiness(stats: Stats, findings: list[Finding], incomplete: bool = False) -> tuple[str, str]:
        """Правила готовности (детерминированные, не зависят от формулировок модели).

        Если часть проверки не выполнена, вывод «можно начинать» не выдаётся.
        """
        conf = [f for f in findings if f.status == "confirmed"]
        crit = [f for f in conf if f.severity == "critical"]
        major = [f for f in conf if f.severity == "major"]
        dev = "not_ready" if crit else ("conditional" if major else "ready")
        test_types = {"unverifiable", "ambiguous", "contradiction"}
        test_crit = [f for f in crit if f.defect_type in test_types]
        test_major = [f for f in major if f.defect_type in test_types]
        coverage = stats.coverage_ratio
        if test_crit or (coverage is not None and coverage < 0.5):
            test = "not_ready"
        elif test_major or crit or (coverage is not None and coverage < 1.0):
            test = "conditional"
        else:
            test = "ready"
        if incomplete:
            dev = "conditional" if dev == "ready" else dev
            test = "conditional" if test == "ready" else test
        return dev, test

    def _verdict(
        self,
        doc: Document,
        reqs: list[Requirement],
        findings: list[Finding],
        stats: Stats,
        coverage: list[CoverageRow],
    ) -> Verdict:
        dev, test = self.readiness(stats, findings, incomplete=bool(self.incomplete))
        conf = [f for f in findings if f.status == "confirmed"]
        important = [f for f in conf if f.severity in ("critical", "major")]
        important.sort(key=lambda f: SEVERITY_ORDER[f.severity])
        by_req = {r.id: r for r in reqs}

        def where(f: Finding) -> str:
            refs = [
                f"{rid} [{by_req[rid].doc_id or by_req[rid].locator or '—'}]"
                for rid in f.req_ids[:3]
                if rid in by_req
            ]
            return ", ".join(refs) or "набор"

        block = "\n".join(
            f"{f.id} | {where(f)} | {f.criterion} {BY_CODE[f.criterion].name} | {f.severity} | "
            f"{f.title}: {f.description[:300]}"
            for f in important[:80]
        )
        cov = f"{stats.coverage_ratio:.0%}" if stats.coverage_ratio is not None else "не оценено"
        stats_text = (
            f"Требований: {stats.requirements}; замечаний: {stats.findings_total} (критичных "
            f"{stats.by_severity.get('critical', 0)}, существенных {stats.by_severity.get('major', 0)}, "
            f"незначительных {stats.by_severity.get('minor', 0)}); требований без замечаний: "
            f"{stats.without_findings}; покрытие критериями приёмки: {cov}."
        )
        valid_ids = {f.id for f in conf}
        try:
            resp = self.session.ask_json(
                "verdict",
                prompts.SYSTEM,
                prompts.verdict_prompt(doc.title, stats_text, dev, test, block or "нет"),
                VerdictLLM,
                label="итог",
            )
            priorities = [
                Priority(
                    title=p.title.strip(),
                    action=p.action.strip(),
                    finding_ids=[i for i in (x.strip() for x in p.finding_ids) if i in valid_ids],
                )
                for p in resp.priorities
            ]
            return Verdict(
                development=dev,
                testing=test,
                summary=resp.summary.strip(),
                development_rationale=resp.development_rationale.strip(),
                testing_rationale=resp.testing_rationale.strip(),
                priorities=priorities,
                incomplete=list(self.incomplete),
            )
        except LLMError as exc:
            self.warnings.append(f"Итоговое заключение сформировано без модели: {exc}")
            return Verdict(
                development=dev,
                testing=test,
                summary=stats_text,
                incomplete=list(self.incomplete),
                priorities=[
                    Priority(title=f.title, action=f.recommendation, finding_ids=[f.id])
                    for f in important[:7]
                ],
            )


# ====================================================================== вспомогательное
_ID_TAIL = re.compile(r"([A-Za-zА-Яа-яЁё]{0,8}-?\d+(?:\.\d+)*)\s*\.?\s*$")


def _literal_doc_id(doc_id: str, quote: str, doc_norm: str) -> bool:
    """Идентификатор считается настоящим, если он есть в тексте документа или им начинается цитата.

    Второй случай — составные подписи вида «Логика работы метода, п. 4» или «Статус 401»,
    где в документе написано «4. …» или «401. …».
    """
    nid = normalize(doc_id).strip(" .")
    if nid and nid in doc_norm:
        return True
    m = _ID_TAIL.search(doc_id.strip())
    if not m:
        return False
    tail = normalize(m.group(1))
    return bool(tail) and re.match(rf"^{re.escape(tail)}(?![\w-])", normalize(quote)) is not None


_REQ_WORDS = re.compile(
    r"(?<![\w-])(?:должн\w*|обязан\w*|необходимо|не допускается|запрещ\w*|shall|must)(?![\w-])", re.IGNORECASE
)
_REQ_SECTIONS = (
    "функциональн",
    "критери",
    "api",
    "логик",
    "правил",
    "параметр",
    "статус",
    "интеграц",
    "данн",
    "ответ",
)


def _suspicious_exclusions(doc: Document, excluded: list[Excluded]) -> list[int]:
    """Исключённые моделью строки, похожие на требования: слова обязательности или раздел требований."""
    content = set(_content_lines(doc))
    out: set[int] = set()
    for e in excluded:
        for n in range(e.line_start, e.line_end + 1):
            if n not in content:
                continue
            if n < doc.line_count and _TABLE_SEP.match(doc.line(n + 1).text.strip()):
                continue  # строка заголовка таблицы
            section = doc.section_of(n).split(" › ")[-1].lower()  # ближайший заголовок, не вся ветка
            if _REQ_WORDS.search(doc.line(n).text) or any(k in section for k in _REQ_SECTIONS):
                out.add(n)
    return sorted(out)


def _trim_excluded(doc: Document, excluded: list[Excluded], reqs: list[Requirement]) -> list[Excluded]:
    """Убрать из исключений строки, вошедшие в требования; остаток диапазона сохраняет причину."""
    covered: set[int] = set()
    for r in reqs:
        a, b = r.evidence.line_start or 0, r.evidence.line_end or r.evidence.line_start or 0
        covered.update(range(a, b + 1))
    content = set(_content_lines(doc))
    out: list[Excluded] = []
    seen: set[tuple[int, int]] = set()
    for e in excluded:
        lines = [n for n in range(e.line_start, e.line_end + 1) if n not in covered]
        if len(lines) == e.line_end - e.line_start + 1:
            if (e.line_start, e.line_end) not in seen:
                seen.add((e.line_start, e.line_end))
                out.append(e)
            continue
        for start, end in _ranges([n for n in lines if n in content]):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            out.append(
                e.model_copy(
                    update={
                        "line_start": start,
                        "line_end": end,
                        "location": doc.location_label(start, end),
                        "section": doc.section_of(start),
                    }
                )
            )
    return out


def _drop_contained(rows: list) -> list:
    """Убрать дубли нарезки: если цитата одного требования целиком входит в цитату другого на тех же строках,
    остаётся пункт с идентификатором документа, а без него — более точный (вложенный) фрагмент."""
    drop: set[int] = set()
    norms = [normalize(r[3].quote) for r in rows]
    for i, (a_start, a_end, a_item, _) in enumerate(rows):
        for j, (b_start, b_end, _b_item, _) in enumerate(rows):
            if i == j or i in drop or j in drop:
                continue
            if b_end < a_start or a_end < b_start or norms[j] not in norms[i]:
                continue
            if norms[i] == norms[j]:
                drop.add(max(i, j))
            elif a_item.doc_id:
                drop.add(j)  # пункт с идентификатором поглощает свой фрагмент
            else:
                drop.add(i)  # без идентификатора оставляем более точный фрагмент
    return [r for k, r in enumerate(rows) if k not in drop]


def _content_lines(doc: Document) -> list[int]:
    out = []
    in_code = False
    for ln in doc.lines:
        s = ln.text.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if not s or ln.is_heading or _TABLE_SEP.match(s) or s in ("---", "***", "___"):
            continue
        out.append(ln.no)
    return out


def _uncovered_lines(doc: Document, reqs: list[Requirement], excluded: list[Excluded]) -> list[int]:
    covered: set[int] = set()
    for r in reqs:
        a, b = r.evidence.line_start or 0, r.evidence.line_end or r.evidence.line_start or 0
        covered.update(range(a, b + 1))
    for e in excluded:
        covered.update(range(e.line_start, e.line_end + 1))
    return [n for n in _content_lines(doc) if n not in covered]


def _ranges(nums: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def _heuristic_segmentation(doc: Document) -> list[SegItem]:
    """Запасная нарезка без модели: каждый непустой абзац, пункт списка или строка таблицы."""
    items = []
    for n in _content_lines(doc):
        text = clean_text(doc.line(n).text)
        if len(text) < 8:
            continue
        m = re.match(r"^([A-Za-zА-Яа-яЁё]{1,5}-\d+)[.)]?\s", text)
        items.append(
            SegItem(
                doc_id=m.group(1) if m else None,
                section=doc.section_of(n),
                kind="other",
                line_start=n,
                line_end=n,
                quote=text,
                summary=text[:80],
            )
        )
    return items
