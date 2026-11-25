"""Детерминированный «LLM» для тестов: отвечает по этапу конвейера, разбирая промпт."""

from __future__ import annotations

import json
import re

from reqvalidator.config import Settings
from reqvalidator.llm.base import LLMClient, LLMResponse, LLMTruncatedError


def fake_settings(**kw) -> Settings:
    base = dict(provider="gemini", model="fake-model", api_key="test", cache_dir=None, min_interval=0.0)
    base.update(kw)
    return Settings(**base)


class FakeClient(LLMClient):
    """Имитация модели для учебного документа tests/fixtures/sample_requirements.md."""

    provider = "fake"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        truncate_batches_over: int | None = None,
        broken_first: set[str] | None = None,
    ):
        super().__init__(settings or fake_settings())
        self.calls: list[tuple[str, str]] = []
        self.truncate_batches_over = truncate_batches_over
        self.broken_first = set(broken_first or ())

    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        self.calls.append((stage, prompt))
        if stage in self.broken_first:
            self.broken_first.discard(stage)
            return LLMResponse(text="это не JSON", model="fake-model")
        handler = {
            "segment": self._segment,
            "segment-coverage": self._coverage,
            "check": self._check,
            "review": self._review,
            "verdict": self._verdict,
        }.get(stage)
        if handler is None and stage.startswith("set-"):
            handler = self._set
        data = handler(prompt, stage) if handler else {}
        return LLMResponse(
            text=json.dumps(data, ensure_ascii=False), model="fake-model", input_tokens=100, output_tokens=50
        )

    # ------------------------------------------------------------------ этапы
    def _segment(self, prompt: str, stage: str) -> dict:
        return {
            "document_title": "Сервис бронирования переговорных",
            "requirements": [
                {
                    "doc_id": "REQ-1",
                    "section": "Функциональные требования",
                    "kind": "functional",
                    "line_start": 7,
                    "line_end": 7,
                    "quote": "REQ-1. Сотрудник бронирует переговорную на интервал от 15 минут до 4 "
                    "часов. Бронирование подтверждается быстро.",
                    "summary": "Бронирование переговорной",
                },
                {
                    "doc_id": "REQ-2",
                    "section": "Функциональные требования",
                    "kind": "functional",
                    "line_start": 8,
                    "line_end": 8,
                    "quote": "REQ-2. Система отправляет напоминание за 10 минут до начала встречи.",
                    "summary": "Напоминание о встрече",
                },
                {
                    "doc_id": None,
                    "section": "Нефункциональные требования",
                    "kind": "nonfunctional",
                    "line_start": 12,
                    "line_end": 12,
                    "quote": "Сервис должен быть удобным и надёжным и т.д.",
                    "summary": "Удобство и надёжность",
                },
                {
                    "doc_id": "REST type",
                    "section": "API",
                    "kind": "api",
                    "line_start": 18,
                    "line_end": 18,
                    "quote": "REST type | GET",
                    "summary": "HTTP-метод GET",
                },
                {
                    "doc_id": "Критерий приёмки 1",
                    "section": "Критерии приемки",
                    "kind": "acceptance",
                    "line_start": 26,
                    "line_end": 26,
                    "quote": "Свободная переговорная | Забронировать на 30 минут | Бронь создана",
                    "summary": "Успешное бронирование",
                },
                # Цитата, которой нет в документе, — агент должен её отбросить.
                {
                    "doc_id": None,
                    "section": "?",
                    "kind": "other",
                    "quote": "Этого текста нет в документе вообще",
                },
            ],
            "excluded": [
                {"line_start": 3, "line_end": 3, "reason": "Пояснение о документе"},
                {"line_start": 34, "line_end": 36, "reason": "Блок согласования"},
            ],
        }

    def _coverage(self, prompt: str, stage: str) -> dict:
        reqs = []
        if "L19|" in prompt:
            reqs.append(
                {
                    "doc_id": "Description",
                    "section": "API",
                    "kind": "api",
                    "line_start": 19,
                    "line_end": 19,
                    "quote": "Description | Сохранение бронирования",
                    "summary": "Назначение метода",
                }
            )
        excluded = [
            {"line_start": n, "line_end": n, "reason": "Служебная строка"}
            for n in (16, 20, 24, 30)
            if f"L{n}|" in prompt
        ]
        return {"requirements": reqs, "excluded": excluded}

    def _check(self, prompt: str, stage: str) -> dict:
        m = re.search(r"требование из пакета \(([^)]*)\)", prompt)
        ids = [x.strip() for x in m.group(1).split(",")] if m else []
        if self.truncate_batches_over and len(ids) > self.truncate_batches_over:
            raise LLMTruncatedError("обрезано (тест)")
        results = []
        for rid in ids:
            block = prompt.split(f"\n{rid} | ", 1)[1].split("\n\n", 1)[0] if f"\n{rid} | " in prompt else ""
            findings = []
            checklist = {f"И{i}": "ok" for i in range(1, 10)}
            if "быстро" in block:
                checklist["И3"] = checklist["И5"] = "fail"
                findings += [
                    {
                        "criterion": "И3",
                        "severity": "major",
                        "title": "Размытое «быстро»",
                        "description": "Не задано время подтверждения.",
                        "quote": "Бронирование подтверждается быстро",
                        "line": 7,
                        "recommendation": "Указать время подтверждения, например «не более 2 с».",
                    },
                    {
                        "criterion": "single",
                        "severity": "minor",
                        "title": "Два требования в одном пункте",
                        "description": "Интервал и подтверждение — разные требования.",
                        "quote": "REQ-1.",
                        "line": 7,
                        "recommendation": "Разделить.",
                    },
                    {
                        "criterion": "zzz-неизвестный",
                        "severity": "minor",
                        "title": "Замечание с неизвестным критерием",
                        "quote": "REQ-1.",
                    },
                ]
            if "удобным" in block:
                checklist["И7"] = "fail"
                findings.append(
                    {
                        "criterion": "И7",
                        "severity": "critical",
                        "title": "Непроверяемое «удобным»",
                        "description": "Нет измеримого критерия.",
                        "quote": "удобным и надёжным",
                        "line": 12,
                        "recommendation": "Задать метрики удобства и доступности.",
                    }
                )
            if "GET" in block:
                checklist["И8"] = "fail"
                findings.append(
                    {
                        "criterion": "И8",
                        "severity": "major",
                        "title": "GET для сохранения",
                        "description": "GET не должен менять состояние.",
                        "quote": "GET",
                        "line": 18,
                        "related": [{"quote": "Сохранение бронирования", "line": 19}],
                        "recommendation": "Использовать POST или PUT.",
                    }
                )
            results.append(
                {
                    "req_id": rid,
                    "checklist": checklist,
                    "findings": findings,
                    "rewrite": "Улучшенная формулировка" if findings else None,
                }
            )
        return {"results": results}

    def _set(self, prompt: str, stage: str) -> dict:
        if stage == "set-consistency":
            return {
                "overview": "Документ описывает бронирование переговорных.",
                "assessments": [
                    {"criterion": "Н2", "status": "warn", "summary": "Есть расхождение"},
                    {"criterion": "Н3", "status": "ok", "summary": "Реализуемо"},
                    {"criterion": "Н4", "status": "ok", "summary": "Понятно"},
                ],
                "findings": [
                    {
                        "criterion": "Н2",
                        "severity": "major",
                        "title": "Метод не соответствует описанию",
                        "description": "GET и «Сохранение».",
                        "evidence": [
                            {"quote": "GET", "line": 18},
                            {"quote": "Сохранение бронирования", "line": 19},
                        ],
                        "req_ids": ["ТР-04", "ТР-05"],
                        "recommendation": "Согласовать метод.",
                    },
                    {
                        "criterion": "Н2",
                        "severity": "minor",
                        "title": "Выдуманная цитата",
                        "evidence": [{"quote": "совершенно отсутствующий фрагмент текста"}],
                    },
                ],
            }
        if stage == "set-completeness":
            return {
                "assessments": [
                    {"criterion": "Н1", "status": "fail", "summary": "Нет раздела тестирования"},
                    {"criterion": "Н5", "status": "warn", "summary": "Мало критериев приёмки"},
                ],
                "findings": [
                    {
                        "criterion": "Н1",
                        "severity": "critical",
                        "title": "Раздел тестирования пуст",
                        "description": "Вместо требований — TBD.",
                        "evidence": [{"quote": "TBD", "line": 30}],
                        "req_ids": [],
                    }
                ],
                "coverage": [
                    {"req_id": "ТР-01", "covered_by": ["ТР-06"]},
                    {"req_id": "ТР-02", "covered_by": [], "note": "Напоминание не проверяется"},
                    {"req_id": "ТР-03", "covered_by": [], "note": "Нет метрик"},
                    {"req_id": "ТР-404", "covered_by": ["ТР-06"]},
                ],
            }
        return {
            "assessments": [
                {"criterion": "Н6", "status": "fail", "summary": "Нет приоритетов"},
                {"criterion": "Н7", "status": "ok", "summary": "Структура понятна"},
                {"criterion": "Н8", "status": "warn", "summary": "Не у всех требований есть ID"},
            ],
            "findings": [
                {
                    "criterion": "Н6",
                    "severity": "minor",
                    "title": "Нет приоритетов требований",
                    "evidence": [{"quote": "Функциональные требования", "line": 5}],
                }
            ],
        }

    def _review(self, prompt: str, stage: str) -> dict:
        ids = re.findall(r"^(F\d{3}) \|", prompt, flags=re.M)
        decisions = [{"id": i, "decision": "keep"} for i in ids]
        # Последнее замечание объявляем дубликатом первого, предпоследнее — ложным.
        if len(ids) >= 3:
            decisions[-1] = {"id": ids[-1], "decision": "merge", "merge_into": ids[0], "reason": "дубликат"}
            decisions[-2] = {"id": ids[-2], "decision": "drop", "reason": "ложное срабатывание"}
        if ids:
            decisions[0]["severity"] = "critical"
        return {"decisions": decisions}

    def _verdict(self, prompt: str, stage: str) -> dict:
        ids = re.findall(r"^(Д-\d+) \|", prompt, flags=re.M)
        return {
            "summary": "Документ не готов: есть критичные дефекты.",
            "development_rationale": "Есть критичные дефекты.",
            "testing_rationale": "Нет метрик.",
            "priorities": [
                {
                    "title": "Устранить критичные дефекты",
                    "action": "Исправить формулировки",
                    "finding_ids": [*ids[:2], "Д-999"],
                }
            ],
        }
