"""Каталог критериев качества требований и шкала серьёзности.

Источники:
* ISO/IEC/IEEE 29148:2018, п. 5.2.5 — характеристики отдельного требования;
  п. 5.2.6 — характеристики набора требований; п. 5.2.7 — языковые критерии.
* IEEE Std 830-1998, п. 4.3 — характеристики качественной спецификации (SRS).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    code: str
    key: str
    name: str
    scope: str  # "individual" | "set"
    source: str
    question: str


INDIVIDUAL: tuple[Criterion, ...] = (
    Criterion(
        "И1",
        "necessary",
        "Необходимость",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 a",
        "Требование описывает существенную возможность, характеристику или ограничение; "
        "без него возникнет пробел. Оно не устарело и не дублирует другое требование.",
    ),
    Criterion(
        "И2",
        "appropriate",
        "Уместность (без лишних проектных решений)",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 b",
        "Уровень детализации соответствует уровню документа; формулировка говорит «что», а не «как»; "
        "нет навязанных без обоснования решений по реализации (технологии, имена классов, методы, форматы).",
    ),
    Criterion(
        "И3",
        "unambiguous",
        "Однозначность",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 c, 5.2.7; IEEE 830 п. 4.3.2",
        "Допускает единственное толкование: нет размытых слов («быстро», «в момент», «и далее»), "
        "неясных ссылок и местоимений, неопределённых терминов, единиц измерения, часовых поясов, границ.",
    ),
    Criterion(
        "И4",
        "complete",
        "Полнота",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 d; IEEE 830 п. 4.3.3",
        "Достаточно сведений для реализации и проверки: субъект, условие, действие, объект, ограничения, "
        "реакция на ошибки и граничные случаи; нет «TBD» и ссылок на неописанное.",
    ),
    Criterion(
        "И5",
        "singular",
        "Единичность (атомарность)",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 e",
        "Формулирует одну возможность или характеристику; несколько независимых обязательств "
        "в одной формулировке нужно разделить.",
    ),
    Criterion(
        "И6",
        "feasible",
        "Осуществимость",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 f",
        "Реализуемо при описанной архитектуре, данных и ограничениях, без недопустимого риска.",
    ),
    Criterion(
        "И7",
        "verifiable",
        "Проверяемость",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 g; IEEE 830 п. 4.3.6",
        "Выполнение можно доказать объективно (тест, анализ, инспекция): есть измеримые значения, "
        "наблюдаемый результат и условия проверки.",
    ),
    Criterion(
        "И8",
        "correct",
        "Корректность и непротиворечивость",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 h; IEEE 830 п. 4.3.1, 4.3.4",
        "Точно отражает потребность и факты предметной области, не противоречит другим требованиям, "
        "примерам и критериям приёмки документа.",
    ),
    Criterion(
        "И9",
        "conforming",
        "Соответствие шаблону и стилю",
        "individual",
        "ISO/IEC/IEEE 29148 п. 5.2.5 i, 5.2.8; IEEE 830 п. 4.3.8",
        "Оформлено по принятому образцу: уникальный идентификатор, явный субъект и модальность "
        "(«система должна…»), единая терминология документа.",
    ),
)

SET: tuple[Criterion, ...] = (
    Criterion(
        "Н1",
        "set_complete",
        "Полнота набора",
        "set",
        "ISO/IEC/IEEE 29148 п. 5.2.6 a; IEEE 830 п. 4.3.3",
        "Набор покрывает все функции, данные, интерфейсы, нефункциональные свойства, ошибки и граничные "
        "случаи; нет пустых разделов, заглушек и ссылок на отсутствующие описания.",
    ),
    Criterion(
        "Н2",
        "set_consistent",
        "Согласованность набора",
        "set",
        "ISO/IEC/IEEE 29148 п. 5.2.6 b; IEEE 830 п. 4.3.4",
        "Требования не противоречат друг другу, примерам и критериям приёмки; одинаковые понятия, "
        "компоненты и единицы называются одинаково; нет дублирования.",
    ),
    Criterion(
        "Н3",
        "set_feasible",
        "Осуществимость набора",
        "set",
        "ISO/IEC/IEEE 29148 п. 5.2.6 c",
        "Набор в совокупности реализуем: требования совместимы по времени, данным и архитектуре.",
    ),
    Criterion(
        "Н4",
        "set_comprehensible",
        "Понятность набора",
        "set",
        "ISO/IEC/IEEE 29148 п. 5.2.6 d",
        "Понятно, что ожидается от системы и как она взаимодействует с окружением; термины определены.",
    ),
    Criterion(
        "Н5",
        "set_validatable",
        "Возможность валидации",
        "set",
        "ISO/IEC/IEEE 29148 п. 5.2.6 e",
        "Можно доказать, что набор удовлетворяет потребностям: измеримые цели, критерии приёмки "
        "покрывают требования.",
    ),
    Criterion(
        "Н6",
        "set_ranked",
        "Ранжирование по важности и стабильности",
        "set",
        "IEEE 830 п. 4.3.5",
        "У требований указаны приоритет и/или стабильность.",
    ),
    Criterion(
        "Н7",
        "set_modifiable",
        "Модифицируемость",
        "set",
        "IEEE 830 п. 4.3.7",
        "Структура позволяет вносить изменения: требования не дублируются, нет смешения нескольких "
        "требований в одном пункте, есть оглавление и перекрёстные ссылки, версия документа.",
    ),
    Criterion(
        "Н8",
        "set_traceable",
        "Прослеживаемость",
        "set",
        "IEEE 830 п. 4.3.8; ISO/IEC/IEEE 29148 п. 5.2.8",
        "У каждого требования есть уникальный идентификатор, указан источник; критерии приёмки и тесты "
        "ссылаются на требования.",
    ),
)

ALL: tuple[Criterion, ...] = INDIVIDUAL + SET
BY_CODE: dict[str, Criterion] = {c.code: c for c in ALL}

SEVERITIES: dict[str, tuple[str, str]] = {
    "critical": (
        "Критический",
        "Блокирует разработку или тестирование: противоречие, непроверяемое или неполное ключевое "
        "требование, нереализуемость. Исправить до начала работ.",
    ),
    "major": (
        "Существенный",
        "Высокий риск неверной реализации или дефектов; исправить до утверждения документа.",
    ),
    "minor": ("Незначительный", "Оформление, стиль, мелкие неточности; исправить при ближайшей правке."),
}
SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}

# Тип дефекта — для группировки критичных дефектов в сводке.
DEFECT_TYPES: dict[str, str] = {
    "contradiction": "Противоречивые",
    "unverifiable": "Непроверяемые",
    "incomplete": "Неполные",
    "ambiguous": "Неоднозначные",
    "infeasible": "Неосуществимые",
    "design": "Лишние проектные решения",
    "structure": "Нарушения структуры и оформления",
    "other": "Прочие",
}
CRITERION_DEFECT_TYPE: dict[str, str] = {
    "И1": "other",
    "И2": "design",
    "И3": "ambiguous",
    "И4": "incomplete",
    "И5": "structure",
    "И6": "infeasible",
    "И7": "unverifiable",
    "И8": "contradiction",
    "И9": "structure",
    "Н1": "incomplete",
    "Н2": "contradiction",
    "Н3": "infeasible",
    "Н4": "ambiguous",
    "Н5": "unverifiable",
    "Н6": "structure",
    "Н7": "structure",
    "Н8": "structure",
}

_ALIASES: dict[str, str] = {}
for _c in ALL:
    _ALIASES[_c.code.lower()] = _c.code
    _ALIASES[_c.key] = _c.code
    _ALIASES[_c.name.lower()] = _c.code
    _ALIASES[_c.name.split(" (")[0].lower()] = _c.code
# Латинские двойники кодов и английские названия.
for _i, _c in enumerate(INDIVIDUAL, 1):
    _ALIASES[f"i{_i}"] = _c.code
for _i, _c in enumerate(SET, 1):
    _ALIASES[f"n{_i}"] = _c.code
    _ALIASES[f"s{_i}"] = _c.code
_ALIASES.update(
    {
        "necessary": "И1",
        "appropriate": "И2",
        "unambiguous": "И3",
        "complete": "И4",
        "singular": "И5",
        "feasible": "И6",
        "verifiable": "И7",
        "correct": "И8",
        "conforming": "И9",
        "consistent": "Н2",
        "однозначность": "И3",
        "полнота": "И4",
        "проверяемость": "И7",
        "корректность": "И8",
        "непротиворечивость": "И8",
        "атомарность": "И5",
        "единичность": "И5",
        "уместность": "И2",
        "согласованность": "Н2",
        "прослеживаемость": "Н8",
        "ранжирование": "Н6",
        "модифицируемость": "Н7",
        "понятность": "Н4",
        "возможность валидации": "Н5",
        "single": "И5",
        "atomic": "И5",
        "atomicity": "И5",
        "singularity": "И5",
        "necessity": "И1",
        "appropriateness": "И2",
        "ambiguity": "И3",
        "ambiguous": "И3",
        "completeness": "И4",
        "feasibility": "И6",
        "verifiability": "И7",
        "testability": "И7",
        "testable": "И7",
        "correctness": "И8",
        "consistency": "Н2",
        "conformance": "И9",
        "conformity": "И9",
        "comprehensible": "Н4",
        "comprehensibility": "Н4",
        "validatable": "Н5",
        "able to be validated": "Н5",
        "ranked": "Н6",
        "ranking": "Н6",
        "priority": "Н6",
        "modifiable": "Н7",
        "modifiability": "Н7",
        "traceable": "Н8",
        "traceability": "Н8",
        "неоднозначность": "И3",
        "непроверяемость": "И7",
        "неполнота": "И4",
        "противоречивость": "И8",
        "противоречие": "И8",
        "осуществимость": "И6",
        "необходимость": "И1",
    }
)


def normalize_criterion(value: str, scope: str | None = None) -> str | None:
    """Привести код критерия из ответа модели к каноническому виду («И3», «Н2»)."""
    if not value:
        return None
    raw = value.strip()
    m = re.match(r"^\s*([ИиНнIiNnSs])\s*-?\s*(\d)\b", raw)
    if m:
        letter = m.group(1).lower()
        num = int(m.group(2))
        code = f"И{num}" if letter in ("и", "i") else f"Н{num}"
        if code in BY_CODE:
            return code
    low = raw.lower().strip(" .")
    if low in _ALIASES:
        return _ALIASES[low]
    for alias, code in _ALIASES.items():
        if len(alias) > 4 and alias in low:
            return code
    return None


def normalize_severity(value: str | None) -> str:
    low = (value or "").strip().lower()
    if low.startswith(("crit", "крит", "block", "блок", "high", "высок")):
        return "critical"
    if low.startswith(("maj", "сущ", "med", "сред", "знач")):
        return "major"
    return "minor"
