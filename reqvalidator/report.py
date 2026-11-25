"""Отчёт о проверке на русском языке: Markdown (основной формат) и JSON (машиночитаемая копия)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from reqvalidator.criteria import BY_CODE, DEFECT_TYPES, INDIVIDUAL, SET, SEVERITIES
from reqvalidator.models import AnalysisResult, Finding

STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "—"}
STATUS_TEXT = {"pass": "соответствует", "warn": "есть замечания", "fail": "нарушение", "na": "не применимо"}
SEV_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}
READY_TEXT = {
    "ready": ("✅", "можно начинать"),
    "conditional": ("⚠️", "можно начинать с оговорками"),
    "not_ready": ("❌", "нельзя начинать"),
}
KIND_RU = {
    "goal": "цель",
    "functional": "функциональное",
    "ui": "интерфейс",
    "process": "процесс",
    "architecture": "архитектура",
    "nonfunctional": "нефункциональное",
    "data": "данные",
    "integration": "интеграция",
    "api": "API",
    "acceptance": "критерий приёмки",
    "other": "прочее",
}
PROVIDER_RU = {
    "gemini": "Google Gemini API",
    "anthropic": "Anthropic Claude API",
    "openai": "OpenAI-совместимый API",
    "ollama": "Ollama (локально)",
    "claude-cli": "Claude Code CLI",
}
CRITICAL_GROUPS = ("contradiction", "unverifiable", "incomplete")


# ---------------------------------------------------------------------------- утилиты
def plural(n: int, forms: tuple[str, str, str]) -> str:
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 14:
        form = forms[2]
    elif n_abs % 10 == 1:
        form = forms[0]
    elif 2 <= n_abs % 10 <= 4:
        form = forms[1]
    else:
        form = forms[2]
    return f"{n} {form}"


REQ_FORMS = ("требование", "требования", "требований")
FIND_FORMS = ("замечание", "замечания", "замечаний")


def cell(text: object) -> str:
    """Безопасное содержимое ячейки Markdown-таблицы."""
    s = "" if text is None else str(text)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", "<br>").strip()


def short(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def fmt_dt(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    offset = dt.strftime("%z")
    tz = f"UTC{offset[:3]}:{offset[3:]}" if offset else ""
    return dt.strftime("%d.%m.%Y %H:%M:%S") + (f" ({tz})" if tz else "")


def fmt_duration(seconds: float) -> str:
    seconds = round(seconds)
    m, s = divmod(seconds, 60)
    return f"{m} мин {s:02d} с" if m else f"{s} с"


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def sev_label(sev: str) -> str:
    return f"{SEV_ICON[sev]} {SEVERITIES[sev][0].lower()}"


_TRANSLIT = str.maketrans({"Т": "t", "Р": "r", "Д": "d", "О": "o", "т": "t", "р": "r", "д": "d", "о": "o"})


def anchor(identifier: str) -> str:
    """Якорь для ID требования/замечания: «ТР-05» → «tr-05» (работает в GitHub и любом HTML)."""
    return identifier.translate(_TRANSLIT).lower()


def link(identifier: str) -> str:
    return f"[{identifier}](#{anchor(identifier)})"


def links(ids: list[str]) -> str:
    return ", ".join(link(i) for i in ids)


SECTIONS = (
    ("passport", "1. Паспорт прогона"),
    ("method", "2. Методика"),
    ("segmentation", "3. Нарезка требований"),
    ("individual", "4. Проверка каждого требования"),
    ("set", "5. Проверка всего набора"),
    ("summary", "6. Сводка"),
    ("verdict", "7. Итог"),
    ("appendix", "Приложения"),
)


def crit_label(code: str) -> str:
    c = BY_CODE.get(code)
    return f"{code} {c.name}" if c else code


# ---------------------------------------------------------------------------- Markdown
def render_markdown(res: AnalysisResult) -> str:
    out: list[str] = []
    w = out.append
    run = res.run
    st = res.stats
    v = res.verdict
    req_by_id = {r.id: r for r in res.requirements}
    confirmed = [f for f in res.findings if f.status == "confirmed"]
    by_id = {f.id: f for f in res.findings}

    def ref(rid: str) -> str:
        r = req_by_id.get(rid)
        return f"{link(rid)} ({r.doc_id})" if r and r.doc_id else link(rid)

    def place(r) -> str:
        return r.section + (f" · {r.locator}" if r.locator else "")

    def section(key: str, title: str) -> None:
        w(f'<a id="{key}"></a>')
        w("")
        w(title)
        w("")

    def evidence_lines(f: Finding) -> list[str]:
        rows = []
        for e in f.evidence:
            loc = e.location or "место не найдено"
            note = "" if e.verified else " — цитата не найдена в документе"
            rows.append(f"«{short(e.quote, 260)}» — {loc}{note}")
        return rows

    def render_finding(f: Finding, heading_level: int = 0) -> None:
        reqs = ", ".join(ref(x) for x in f.req_ids) if f.req_ids else "набор в целом"
        head = (
            f'<a id="{anchor(f.id)}"></a>**{f.id}** · {sev_label(f.severity)} · {crit_label(f.criterion)} — '
            f"**{f.title}**"
        )
        w(head + "  ")
        if f.scope == "set":
            w(f"*Затронутые требования:* {reqs}  ")
        if f.description:
            w(f.description + "  ")
        ev = evidence_lines(f)
        if len(ev) == 1:
            w(f"*Где:* {ev[0]}  ")
        elif ev:
            w("*Где:*")
            w("")
            for e in ev:
                w(f"- {e}")
            w("")
        if f.recommendation:
            w(f"*Рекомендация:* {f.recommendation}  ")
        if f.review_note:
            w(f"*Самопроверка:* {f.review_note}  ")
        w("")

    # ------------------------------------------------------------ заголовок
    w(f"# Отчёт о проверке требований: «{run.document_title}»")
    w("")
    if v:
        dev_icon, dev_text = READY_TEXT[v.development]
        test_icon, test_text = READY_TEXT[v.testing]
        crit = st.by_severity.get("critical", 0)
        w(
            f"> **Итог:** разработку — {dev_icon} {dev_text}; тестирование — {test_icon} {test_text}. "
            f"Выделено {plural(st.requirements, REQ_FORMS)}, подтверждено {plural(st.findings_total, FIND_FORMS)}, "
            f"из них критичных — {crit}. Подробности — в разделе 7."
        )
        w("")
    w(
        "Отчёт сформирован автоматически агентом reqvalidator по результату прогона языковой модели. "
        "Каждое замечание привязано к месту документа: цитаты сверены с исходным текстом агентом, "
        "а номера строк и страниц вычислены по файлу, а не взяты из ответа модели."
    )
    w("")
    w("**Содержание:** " + " · ".join(f"[{title}](#{key})" for key, title in SECTIONS))
    w("")

    # ------------------------------------------------------------ 1. паспорт
    section("passport", "## 1. Паспорт прогона")
    w("| Параметр | Значение |")
    w("| :--- | :--- |")
    w(f"| Название документа | «{cell(run.document_title)}» |")
    size = f"{run.document_lines} строк" + (f", {run.document_pages} стр." if run.document_pages else "")
    w(f"| Файл | `{cell(run.document_path)}` ({cell(run.document_format)}, {size}) |")
    w(f"| SHA-256 файла | `{run.document_sha256}` |")
    w(f"| Дата и время прогона | {fmt_dt(run.started_at)} |")
    w(f"| Длительность прогона | {fmt_duration(run.duration_s)} |")
    provider = PROVIDER_RU.get(run.provider, run.provider)
    reported = ", ".join(f"`{m}`" for m in run.models_reported)
    model_line = f"`{cell(run.model)}` (провайдер: {provider})"
    if reported and run.models_reported != [run.model]:
        model_line += f"; модель по ответу API: {reported}"
    w(f"| Модель | {model_line} |")
    params = []
    s = run.settings
    if s.get("effort"):
        params.append(f"effort: {s['effort']}")
    params.append(
        f"temperature: {s['temperature'] if s.get('temperature') is not None else 'по умолчанию провайдера'}"
    )
    params.append(f"требований в пакете: {s.get('batch_size')}")
    params.append("самопроверка: " + ("включена" if s.get("review") else "выключена"))
    w(f"| Параметры прогона | {cell('; '.join(params))} |")
    w(f"| Версия агента | reqvalidator {cell(run.agent_version)} |")
    w("| Критерии | ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6, 5.2.7; IEEE Std 830-1998 п. 4.3 |")
    calls = f"{run.llm_calls}" + (f" (+{run.llm_cached_calls} из кэша)" if run.llm_cached_calls else "")
    tokens = f"вход {fmt_int(run.input_tokens)}, выход {fmt_int(run.output_tokens)}"
    w(f"| Обращений к модели | {calls}; токенов: {tokens} |")
    if run.warnings:
        w(f"| Предупреждения прогона | {cell('; '.join(run.warnings))} |")
    w("")

    # ------------------------------------------------------------ 2. методика
    section("method", "## 2. Методика проверки")
    w("Проверка статическая — ревью текста требований до разработки. Этапы прогона:")
    w("")
    w("1. **Чтение документа** и нумерация строк; для PDF — извлечение текста, таблиц и заголовков.")
    w(
        "2. **Статические проверки** без модели: размытые слова и открытые перечни (ISO/IEC/IEEE 29148, п. 5.2.7), "
        "заглушки, незаполненные поля, смешение единиц, HTTP-семантика (приложение А)."
    )
    w(
        "3. **Нарезка** документа моделью на отдельные требования; строки, не попавшие в нарезку, "
        "агент отправляет модели повторно, остаток фиксирует явно (раздел 3)."
    )
    w(
        "4. **Проверка каждого требования** по критериям И1–И9 с учётом всего документа как контекста (раздел 4)."
    )
    w(
        "5. **Проверка набора целиком** — три прохода: согласованность, полнота и покрытие критериями приёмки, "
        "структура и прослеживаемость (раздел 5)."
    )
    w(
        "6. **Сверка цитат**: каждая цитата модели ищется в тексте документа; место (строка, страница) вычисляет агент."
    )
    w(
        "7. **Самопроверка**: модель-рецензент отсеивает ложные срабатывания и дубликаты и калибрует серьёзность "
        "(приложение Б)."
    )
    w(
        "8. **Итог**: готовность к разработке и тестированию определяется правилами агента по числу и типу "
        "подтверждённых дефектов; модель формулирует обоснование и приоритеты исправлений (раздел 7)."
    )
    w("")
    w("| Код | Критерий | Источник | Что проверяется |")
    w("| :--- | :--- | :--- | :--- |")
    for c in INDIVIDUAL + SET:
        w(f"| {c.code} | {cell(c.name)} | {cell(c.source)} | {cell(c.question)} |")
    w("")
    w("Шкала серьёзности:")
    w("")
    for k, (name, desc) in SEVERITIES.items():
        w(f"- {SEV_ICON[k]} **{name.lower()}** — {desc[0].lower() + desc[1:]}")
    w("")
    w(
        "Обозначения в матрицах: ✅ соответствует; ⚠️ незначительное замечание; ❌ нарушение "
        "(существенное или критичное); — критерий не применим."
    )
    w("")

    # ------------------------------------------------------------ 3. нарезка
    section("segmentation", "## 3. Как агент нарезал требования")
    no_id = st.requirements - st.with_doc_id
    w(
        f"Агент выделил {plural(st.requirements, REQ_FORMS)}. Собственный идентификатор в документе есть у "
        f"{st.with_doc_id}; остальным {no_id} агент присвоил идентификаторы ТР-xx и сохранил дословную цитату."
    )
    w("")
    w("| ID в отчёте | ID в документе | Раздел | Место | Тип | Краткая формулировка | Цитата из документа |")
    w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in res.requirements:
        w(
            f"| {link(r.id)} | {cell(r.doc_id or '—')} | {cell(place(r))} | {cell(r.evidence.location)} | "
            f"{KIND_RU.get(r.kind, r.kind)} | {cell(r.summary)} | «{cell(r.evidence.quote)}» |"
        )
    w("")
    if res.excluded:
        w("Фрагменты документа, не отнесённые к требованиям (оцениваются на уровне набора):")
        w("")
        w("| Место | Раздел | Причина |")
        w("| :--- | :--- | :--- |")
        for e in res.excluded:
            w(f"| {cell(e.location)} | {cell(e.section)} | {cell(e.reason)} |")
        w("")

    # ------------------------------------------------------------ 4. каждое требование
    section("individual", "## 4. Проверка каждого требования")
    w("### 4.1. Матрица соответствия критериям")
    w("")
    w("| ID | ID в док. | " + " | ".join(c.code for c in INDIVIDUAL) + " | Замечания |")
    w("| :--- | :--- | " + " | ".join(":---:" for _ in INDIVIDUAL) + " | :--- |")
    checks = {c.req_id: c for c in res.checks}
    for r in res.requirements:
        chk = checks.get(r.id)
        marks = [STATUS_ICON[chk.statuses.get(c.code, "na")] if chk else "—" for c in INDIVIDUAL]
        own = [by_id[i] for i in (chk.finding_ids if chk else []) if i in by_id]
        sev = Counter(f.severity for f in own)
        summary = (
            " ".join(f"{SEV_ICON[k]}{sev[k]}" for k in ("critical", "major", "minor") if sev.get(k))
            if own
            else "нет"
        )
        w(f"| {link(r.id)} | {cell(r.doc_id or '—')} | " + " | ".join(marks) + f" | {summary} |")
    w("")
    w("Критерии: " + "; ".join(f"{c.code} — {c.name}" for c in INDIVIDUAL) + ".")
    w("")
    w("### 4.2. Результаты по требованиям")
    w("")
    for r in res.requirements:
        chk = checks.get(r.id)
        own = [by_id[i] for i in (chk.finding_ids if chk else []) if i in by_id]
        primary = [f for f in own if f.scope == "requirement" and f.req_ids and f.req_ids[0] == r.id]
        related = [f for f in own if f not in primary]
        label = r.doc_id or r.locator
        title = f"{r.id}" + (f" · {label}" if label else "") + f" — {r.summary}"
        section(anchor(r.id), f"#### {title}")
        w(f"> «{r.evidence.quote}»  ")
        w(f"> *{r.evidence.location}; раздел: {place(r) or '—'}; тип: {KIND_RU.get(r.kind, r.kind)}*")
        w("")
        if chk is None or all(s == "na" for s in chk.statuses.values()):
            w("⚠️ Результат проверки не получен (см. предупреждения прогона).")
            w("")
            continue
        na = [c.code for c in INDIVIDUAL if chk.statuses.get(c.code) == "na"]
        ok = [c.code for c in INDIVIDUAL if chk.statuses.get(c.code) == "pass"]
        if not own:
            line = "✅ Нарушений не выявлено"
            line += f" (соответствует: {', '.join(ok)}" + (f"; не применимо: {', '.join(na)})" if na else ")")
            w(line + ".")
            w("")
            continue
        sev = Counter(f.severity for f in own)
        parts = [f"{SEV_ICON[k]} {sev[k]}" for k in ("critical", "major", "minor") if sev.get(k)]
        bad = [c.code for c in INDIVIDUAL if chk.statuses.get(c.code) in ("fail", "warn")]
        w(
            f"**Оценка:** {plural(len(own), FIND_FORMS)} ({', '.join(parts)}); нарушены критерии: "
            f"{', '.join(bad)}" + (f"; соответствует: {', '.join(ok)}" if ok else "") + "."
        )
        w("")
        for f in primary:
            render_finding(f)
        if related:
            w("*Дефекты этого требования, описанные в объединённых замечаниях:*")
            w("")
            for f in related:
                where = (
                    "замечание к набору"
                    if f.scope == "set"
                    else "общее замечание с " + ", ".join(x for x in f.req_ids if x != r.id)
                )
                w(
                    f"- {link(f.id)} · {sev_label(f.severity)} · {crit_label(f.criterion)} — {f.title} ({where})"
                )
            w("")
        if chk.rewrite:
            w("**Предлагаемая формулировка:**")
            w("")
            for line in chk.rewrite.strip().splitlines():
                w(f"> {line}" if line.strip() else ">")
            w("")

    # ------------------------------------------------------------ 5. набор
    section("set", "## 5. Проверка всего набора требований")
    if res.set_overview:
        w(res.set_overview)
        w("")
    w("### 5.1. Оценка по критериям набора")
    w("")
    w("| Код | Критерий | Источник | Оценка | Замечания | Комментарий |")
    w("| :--- | :--- | :--- | :---: | :--- | :--- |")
    for sc in res.set_checks:
        c = BY_CODE[sc.criterion]
        ids = links(sc.finding_ids) or "—"
        w(
            f"| {c.code} | {cell(c.name)} | {cell(c.source)} | {STATUS_ICON[sc.status]} {STATUS_TEXT[sc.status]} | "
            f"{ids} | {cell(sc.summary)} |"
        )
    w("")
    w("### 5.2. Индивидуальные критерии в масштабе набора")
    w("")
    w("| Критерий | ✅ | ⚠️ | ❌ | — | Доля соответствующих |")
    w("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for c in INDIVIDUAL:
        cnt = Counter(chk.statuses.get(c.code, "na") for chk in res.checks)
        applicable = cnt["pass"] + cnt["warn"] + cnt["fail"]
        share = f"{cnt['pass'] / applicable:.0%}" if applicable else "—"
        w(f"| {c.code} {c.name} | {cnt['pass']} | {cnt['warn']} | {cnt['fail']} | {cnt['na']} | {share} |")
    w("")
    set_findings = [f for f in confirmed if f.scope == "set"]
    w("### 5.3. Замечания к набору требований")
    w("")
    if not set_findings:
        w("Замечаний к набору в целом не выявлено.")
        w("")
    for c in SET:
        group = [f for f in set_findings if f.criterion == c.code]
        if not group:
            continue
        w(f"#### {c.code}. {c.name}")
        w("")
        for f in group:
            render_finding(f)
    w("### 5.4. Покрытие требований критериями приёмки")
    w("")
    if res.coverage:
        ratio = st.coverage_ratio
        if ratio is not None:
            w(
                f"Покрыто критериями приёмки: **{ratio:.0%}** проверяемых требований (функциональные, API, данные, "
                "интеграция, нефункциональные, UI, процесс)."
            )
            w("")
        w("| Требование | Тип | Покрывающие критерии приёмки | Что не покрыто |")
        w("| :--- | :--- | :--- | :--- |")
        for row in res.coverage:
            r = req_by_id.get(row.req_id)
            covered = ", ".join(ref(x) for x in row.covered_by) or "❌ нет"
            w(
                f"| {ref(row.req_id)} | {KIND_RU.get(r.kind, r.kind) if r else ''} | {cell(covered)} | "
                f"{cell(row.note or '—')} |"
            )
        w("")
    else:
        w("Матрица покрытия не получена.")
        w("")

    # ------------------------------------------------------------ 6. сводка
    section("summary", "## 6. Сводка")
    w(
        f"- Выделено требований: **{st.requirements}** (с идентификатором в документе — {st.with_doc_id}, "
        f"идентификатор присвоен агентом — {st.requirements - st.with_doc_id})."
    )
    w(f"- Требований без замечаний: **{st.without_findings}**; с замечаниями: **{st.with_findings}**.")
    w(
        f"- Нарушений (подтверждённых замечаний): **{st.findings_total}** — "
        f"{SEV_ICON['critical']} критичных: **{st.by_severity.get('critical', 0)}**, "
        f"{SEV_ICON['major']} существенных: **{st.by_severity.get('major', 0)}**, "
        f"{SEV_ICON['minor']} незначительных: **{st.by_severity.get('minor', 0)}**; "
        f"к отдельным требованиям — {st.individual_findings}, к набору в целом — {st.set_findings}."
    )
    if st.coverage_ratio is not None:
        w(f"- Покрытие критериями приёмки: **{st.coverage_ratio:.0%}**.")
    w(
        f"- Отклонено при самопроверке: {st.rejected}; не подтверждено цитатой: {st.unverified} (приложение Б)."
    )
    w("")
    w("| Критерий | 🔴 | 🟠 | 🟡 | Всего |")
    w("| :--- | ---: | ---: | ---: | ---: |")
    for c in INDIVIDUAL + SET:
        group = [f for f in confirmed if f.criterion == c.code]
        if not group:
            continue
        cnt = Counter(f.severity for f in group)
        w(f"| {c.code} {c.name} | {cnt['critical']} | {cnt['major']} | {cnt['minor']} | {len(group)} |")
    w("")
    w("### Критичные дефекты")
    w("")
    critical = [f for f in confirmed if f.severity == "critical"]
    w(
        f"Критичных дефектов: **{len(critical)}**. Для непроверяемых, противоречивых и неполных формулировок "
        "ниже перечислены также существенные дефекты (🟠): они не блокируют работу сами по себе, но требуют "
        "исправления до утверждения документа."
    )
    w("")
    groups = list(CRITICAL_GROUPS) + [k for k in DEFECT_TYPES if k not in CRITICAL_GROUPS]
    for g in groups:
        if g in CRITICAL_GROUPS:
            items = [f for f in confirmed if f.defect_type == g and f.severity in ("critical", "major")]
            items.sort(key=lambda f: f.severity != "critical")
        else:
            items = [f for f in critical if f.defect_type == g]
        if not items and g not in CRITICAL_GROUPS:
            continue
        w(f"**{DEFECT_TYPES[g]} формулировки**" if g in CRITICAL_GROUPS else f"**{DEFECT_TYPES[g]}**")
        w("")
        if not items:
            w("Не выявлено.")
            w("")
            continue
        w("| ID | Серьёзность | Требования | Суть | Место |")
        w("| :--- | :--- | :--- | :--- | :--- |")
        for f in items:
            shown = f.req_ids[:4]
            reqs = ", ".join(ref(x) for x in shown) or "набор"
            if len(f.req_ids) > len(shown):
                reqs += f" и ещё {len(f.req_ids) - len(shown)}"
            places = (
                "; ".join(list(dict.fromkeys(e.location or "?" for e in f.evidence if e.verified))[:4]) or "—"
            )
            w(f"| {link(f.id)} | {SEV_ICON[f.severity]} | {cell(reqs)} | {cell(f.title)} | {cell(places)} |")
        w("")

    # ------------------------------------------------------------ 7. итог
    section("verdict", "## 7. Итог")
    if v:
        dev_icon, dev_text = READY_TEXT[v.development]
        test_icon, test_text = READY_TEXT[v.testing]
        w(
            f"**Разработка:** {dev_icon} {dev_text}."
            + (f" {v.development_rationale}" if v.development_rationale else "")
        )
        w("")
        w(
            f"**Тестирование:** {test_icon} {test_text}."
            + (f" {v.testing_rationale}" if v.testing_rationale else "")
        )
        w("")
        if v.summary:
            w(v.summary)
            w("")
        w("### Что исправить в первую очередь")
        w("")
        for i, p in enumerate(v.priorities, 1):
            ids = f" ({links(p.finding_ids)})" if p.finding_ids else ""
            action = f" — {p.action}" if p.action else ""
            w(f"{i}. **{p.title}**{action}{ids}")
        w("")
        w(
            "*Правило готовности:* разработка — «нельзя», если есть хотя бы один критичный дефект; «с оговорками» — "
            "если есть существенные. Тестирование — «нельзя» при критичных противоречиях, неоднозначностях или "
            "непроверяемости либо покрытии критериями приёмки ниже 50 %; «с оговорками» — при прочих критичных или "
            "существенных дефектах этих типов либо неполном покрытии."
        )
        w("")

    # ------------------------------------------------------------ приложения
    section("appendix", "## Приложение А. Сигналы статического анализатора")
    if res.lint:
        w(
            "Сигналы получены без модели и переданы ей как подсказки; подтверждённые дефекты отражены в разделах 4–5."
        )
        w("")
        w("| Место | Правило | Критерий | Фрагмент | Пояснение |")
        w("| :--- | :--- | :--- | :--- | :--- |")
        for h in res.lint:
            w(
                f"| {cell(h.location)} | {cell(h.title)} | {h.criterion} | «{cell(short(h.fragment, 80))}» | "
                f"{cell(h.message)} |"
            )
    else:
        w("Сигналов нет.")
    w("")
    w("## Приложение Б. Замечания, не вошедшие в отчёт")
    w("")
    dropped = [f for f in res.findings if f.status != "confirmed"]
    if dropped:
        w("| ID | Требования | Критерий | Суть | Причина |")
        w("| :--- | :--- | :--- | :--- | :--- |")
        for f in dropped:
            reqs = ", ".join(ref(x) for x in f.req_ids) or "набор"
            w(
                f'| <a id="{anchor(f.id)}"></a>{f.id} | {cell(reqs)} | {f.criterion} | {cell(f.title)} | '
                f"{cell(f.review_note or '')} |"
            )
    else:
        w("Все замечания модели прошли самопроверку.")
    w("")
    w("## Приложение В. Журнал обращений к модели")
    w("")
    w("| № | Этап | Пакет | Время, с | Попыток | Токены (вход / выход) | Кэш | Результат |")
    w("| ---: | :--- | :--- | ---: | ---: | :--- | :---: | :--- |")
    for i, c in enumerate(run.calls, 1):
        tok = f"{fmt_int(c.input_tokens or 0)} / {fmt_int(c.output_tokens or 0)}"
        result = "успешно" if c.ok else f"ошибка: {short(c.error or '', 80)}"
        w(
            f"| {i} | {cell(c.stage)} | {cell(c.label)} | {c.duration_s:.1f} | {c.attempts} | {tok} | "
            f"{'да' if c.cached else 'нет'} | {cell(result)} |"
        )
    w("")
    w(
        f"*Отчёт сформирован: {fmt_dt(run.finished_at or run.started_at)}. "
        f"Машиночитаемая версия результатов — JSON-файл рядом с отчётом.*"
    )
    w("")
    return "\n".join(out)


def write_report(res: AnalysisResult, path: Path, json_path: Path | None = None) -> list[Path]:
    """Записать отчёт. Формат по расширению: .json — только JSON, иначе Markdown (+ JSON рядом)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    data = res.model_dump(mode="json")
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return [path]
    path.write_text(render_markdown(res), encoding="utf-8")
    written.append(path)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(json_path)
    return written
