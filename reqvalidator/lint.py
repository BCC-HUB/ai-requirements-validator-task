"""Детерминированные проверки текста (без LLM).

Правила опираются на языковые критерии ISO/IEC/IEEE 29148 п. 5.2.7 (размытые и
субъективные слова, открытые перечни, лазейки) и на типовые дефекты оформления
(заглушки, пустые разделы и таблицы, смешение единиц, GET для изменения данных).
Результаты попадают в отчёт отдельным приложением и передаются модели как
подсказки: модель решает, является ли сигнал дефектом в контексте.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reqvalidator.document import Document
from reqvalidator.models import LintHit


@dataclass(frozen=True)
class _Rule:
    code: str
    title: str
    criterion: str
    pattern: re.Pattern
    message: str


def _rx(words: list[str]) -> re.Pattern:
    return re.compile(r"(?<![\w-])(?:" + "|".join(words) + r")(?![\w-])", re.IGNORECASE)


_VAGUE = _rx(
    [
        r"быстр(?:о|ый|ая|ое|ые|ого)?",
        r"оперативн\w*",
        r"своевременн\w*",
        r"незамедлительн\w*",
        r"мгновенн\w*",
        r"немедленн\w*",
        r"в\s+момент",
        r"в\s+кратчайш\w*",
        r"как\s+можно\s+\w+",
        r"удобн\w*",
        r"интуитивн\w*",
        r"оптимальн\w*",
        r"эффективн\w*",
        r"надёжн\w*",
        r"надежн\w*",
        r"гибк\w*",
        r"современн\w*",
        r"достаточн\w*",
        r"адекватн\w*",
        r"разумн\w*",
        r"приемлем\w*",
        r"качественн\w*",
        r"максимально",
        r"минимально",
        r"должным\s+образом",
        r"надлежащ\w*",
        r"корректно",
        r"по\s+возможности",
        r"если\s+возможно",
        r"при\s+необходимости",
        r"по\s+мере\s+необходимости",
        r"в\s+случае\s+необходимости",
        r"как\s+правило",
        r"обычно",
        r"в\s+основном",
        r"некотор\w+",
        r"соответствующ\w+",
        r"user-friendly",
        r"fast",
        r"asap",
    ]
)
_OPEN_LIST = _rx(
    [
        r"и\s+т\.\s?д\.?",
        r"и\s+т\.\s?п\.?",
        r"и\s+др\.",
        r"и\s+прочее",
        r"и\s+так\s+далее",
        r"и\s+далее",
        r"etc\.?",
    ]
)
_AND_OR = re.compile(r"\bи/или\b", re.IGNORECASE)
_PLACEHOLDER = _rx(
    [
        r"TBD",
        r"TBA",
        r"TODO",
        r"FIXME",
        r"XXX",
        r"уточнить",
        r"уточняется",
        r"будет\s+определ\w*",
        r"будет\s+уточн\w*",
        r"заполняется",
        r"заполняются",
        r"не\s+заполнен\w*",
        r"не\s+проставлен\w*",
        r"lorem\s+ipsum",
    ]
)
_UNFILLED_CHOICE = re.compile(r"\bДа\s*/\s*Нет\b(?:\s*/\s*Не\s+требуется)?", re.IGNORECASE)
_DASH_CELL = re.compile(r"^[\s—–\-]*$")
_STATE_CHANGE = re.compile(
    r"сохран|измен|удал|созда|обнов|замен|запис|save|update|delete|create|remove|set[A-Z]", re.IGNORECASE
)
_GET = re.compile(r"(?<![\w-])GET(?![\w-])")
_MINOR_MAJOR_UNITS = (
    (re.compile(r"тиын", re.I), re.compile(r"тенге|₸|KZT", re.I), "тиыны и тенге"),
    (re.compile(r"копе[ейк]", re.I), re.compile(r"рубл|₽|RUB", re.I), "копейки и рубли"),
    (re.compile(r"цент", re.I), re.compile(r"доллар|USD|\$", re.I), "центы и доллары"),
)

_RULES: tuple[_Rule, ...] = (
    _Rule(
        "LNG-VAGUE",
        "Размытая или субъективная формулировка",
        "И3",
        _VAGUE,
        "Слово «{m}» допускает разные толкования (ISO/IEC/IEEE 29148, п. 5.2.7): замените измеримым "
        "значением или точным условием.",
    ),
    _Rule(
        "LNG-OPEN",
        "Открытый перечень",
        "И3",
        _OPEN_LIST,
        "Перечень «{m}» не закрыт: невозможно определить полный набор значений и проверить требование.",
    ),
    _Rule(
        "LNG-ANDOR",
        "Конструкция «и/или»",
        "И3",
        _AND_OR,
        "«и/или» допускает два толкования — укажите, допустимы ли оба варианта одновременно.",
    ),
    _Rule(
        "DOC-TBD",
        "Заглушка или незаполненный текст",
        "Н1",
        _PLACEHOLDER,
        "Фрагмент «{m}» означает, что содержание не определено; требование нельзя реализовать и проверить.",
    ),
)


def _is_table_separator(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", text)) and "-" in text


def _cells(text: str) -> list[str] | None:
    s = text.strip()
    if not (s.startswith("|") and s.endswith("|")) or _is_table_separator(s):
        return None
    return [c.strip() for c in s.strip("|").split("|")]


def run_lint(doc: Document) -> list[LintHit]:
    hits: list[LintHit] = []
    seen: set[tuple[str, int, str]] = set()

    def add(rule: str, title: str, criterion: str, line: int, fragment: str, message: str) -> None:
        key = (rule, line, fragment.lower())
        if key in seen:
            return
        seen.add(key)
        hits.append(
            LintHit(
                rule=rule,
                title=title,
                criterion=criterion,
                line=line,
                location=doc.location_label(line),
                fragment=fragment,
                message=message,
            )
        )

    in_code = False
    for ln in doc.lines:
        text = ln.text
        stripped = text.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or ln.is_heading or not stripped:
            continue
        for rule in _RULES:
            for m in rule.pattern.finditer(text):
                frag = m.group(0)
                add(rule.code, rule.title, rule.criterion, ln.no, frag, rule.message.format(m=frag))
        cells = _cells(text)
        if cells is not None:
            if cells and all(not c for c in cells):
                add(
                    "DOC-EMPTYROW",
                    "Пустая строка таблицы",
                    "Н1",
                    ln.no,
                    stripped,
                    "Строка таблицы не заполнена: обязательные сведения (ответственные, подписи, значения) отсутствуют.",
                )
            elif (
                len(cells) >= 2
                and cells[0]
                and all(_DASH_CELL.match(c) for c in cells[1:])
                and any(cells[1:])
            ):
                add(
                    "DOC-DASH",
                    "Значение не указано («—»)",
                    "Н1",
                    ln.no,
                    stripped,
                    f"Для «{cells[0]}» вместо значения стоит прочерк — сведения отсутствуют.",
                )
            else:
                for c in cells[1:]:
                    if c and _DASH_CELL.match(c) is None and re.fullmatch(r".*:\s*[—–-]\s*\.?", c):
                        add(
                            "DOC-DASH",
                            "Значение не указано («—»)",
                            "Н1",
                            ln.no,
                            c,
                            f"Во фрагменте «{c}» значение заменено прочерком.",
                        )
            choices = [c for c in cells[1:] if _UNFILLED_CHOICE.fullmatch(c)]
            if len(choices) >= 1:
                add(
                    "DOC-CHOICE",
                    "Не сделан выбор варианта",
                    "Н1",
                    ln.no,
                    choices[0],
                    "Ячейка содержит варианты шаблона («Да / Нет») без выбранного значения: пункт не заполнен.",
                )

    _empty_sections(doc, add)
    _http_get_for_changes(doc, add)
    _mixed_units(doc, add)
    hits.sort(key=lambda h: (h.line, h.rule))
    return _group_consecutive(doc, hits, "DOC-CHOICE")


def _group_consecutive(doc: Document, hits: list[LintHit], rule: str) -> list[LintHit]:
    """Схлопнуть однотипные сигналы на подряд идущих строках (например, незаполненный чек-лист)."""
    out: list[LintHit] = []
    run: list[LintHit] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            first, last = run[0], run[-1]
            out.append(
                first.model_copy(
                    update={
                        "location": doc.location_label(first.line, last.line),
                        "message": f"{first.message} Таких строк подряд: {len(run)}.",
                    }
                )
            )
        run.clear()

    for h in hits:
        if h.rule == rule and (not run or h.line - run[-1].line <= 1):
            run.append(h)
            continue
        if h.rule == rule:
            flush()
            run.append(h)
            continue
        out.append(h)
    flush()
    out.sort(key=lambda h: (h.line, h.rule))
    return out


def _empty_sections(doc: Document, add) -> None:
    lines = doc.lines
    for i, ln in enumerate(lines):
        if not ln.is_heading:
            continue
        body: list[str] = []
        for nxt in lines[i + 1 :]:
            if nxt.is_heading and nxt.heading_level <= ln.heading_level:
                break
            if nxt.is_heading:
                body.append("#")  # есть подраздел — раздел не пустой
                break
            if nxt.text.strip():
                body.append(nxt.text.strip())
        if not body:
            title = ln.text.lstrip("# ").strip()
            add("DOC-EMPTY", "Пустой раздел", "Н1", ln.no, title, f"Раздел «{title}» не содержит текста.")


def _http_get_for_changes(doc: Document, add) -> None:
    by_section: dict[str, list[int]] = {}
    for ln in doc.lines:
        by_section.setdefault(doc.section_of(ln.no).split(" › ")[0], []).append(ln.no)
    for nos in by_section.values():
        text = "\n".join(doc.line(n).text for n in nos)
        if not _GET.search(text) or not _STATE_CHANGE.search(text):
            continue
        for n in nos:
            line = doc.line(n).text
            if _GET.search(line) and ("curl" not in line.lower() or "-G" in line):
                add(
                    "API-GET",
                    "GET для операции изменения данных",
                    "И8",
                    n,
                    line.strip()[:120],
                    "Метод GET должен быть безопасным и идемпотентным (RFC 9110, п. 9.2.1), а в разделе описано "
                    "сохранение/изменение данных; кроме того, параметры GET попадают в URL и журналы.",
                )
                break


def _mixed_units(doc: Document, add) -> None:
    text = doc.text
    for minor, major, label in _MINOR_MAJOR_UNITS:
        if minor.search(text) and major.search(text):
            for ln in doc.lines:
                m = minor.search(ln.text)
                if m:
                    add(
                        "DATA-UNITS",
                        "Смешение единиц измерения",
                        "Н2",
                        ln.no,
                        m.group(0),
                        f"В документе одновременно используются {label}: сравнение сумм в разных единицах "
                        "требует явного правила пересчёта и округления.",
                    )
