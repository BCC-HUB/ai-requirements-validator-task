from reqvalidator.lint import run_lint
from reqvalidator.loaders import load_document


def _lint_text(tmp_path, text: str):
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    return run_lint(load_document(path))


def test_sample_document_signals(fixtures):
    hits = run_lint(load_document(fixtures / "sample_requirements.md"))
    rules = {(h.rule, h.line) for h in hits}
    assert ("LNG-VAGUE", 7) in rules  # «быстро»
    assert ("LNG-VAGUE", 12) in rules  # «удобным», «надёжным»
    assert ("LNG-OPEN", 12) in rules  # «и т.д.»
    assert ("DOC-TBD", 30) in rules
    assert ("DOC-DASH", 20) in rules  # Swagger — «—»
    assert ("DOC-EMPTYROW", 36) in rules
    assert ("API-GET", 18) in rules
    for h in hits:
        assert h.location.startswith("строка")
        assert h.criterion[0] in "ИН"


def test_precise_wording_is_not_flagged(tmp_path):
    hits = _lint_text(tmp_path, "# Д\n\nНе более одного уведомления: максимум один push в месяц.\n")
    assert hits == []


def test_unfilled_template_choices_are_grouped(tmp_path):
    text = "# Д\n\n| Проверка | Создано | Проверено |\n| :-- | :-- | :-- |\n" + "".join(
        f"| Пункт {i} | Да / Нет | Да / Нет |\n" for i in range(5)
    )
    hits = [h for h in _lint_text(tmp_path, text) if h.rule == "DOC-CHOICE"]
    assert len(hits) == 1
    assert hits[0].location == "строки 5–9"
    assert "5" in hits[0].message


def test_mixed_currency_units_and_empty_section(tmp_path):
    text = "# Д\n\n## Данные\n\nСумма в тиынах, порог в тенге.\n\n## Тестирование\n\n## Итог\n\nТекст.\n"
    hits = _lint_text(tmp_path, text)
    assert any(h.rule == "DATA-UNITS" and h.line == 5 for h in hits)
    assert any(h.rule == "DOC-EMPTY" and h.fragment == "Тестирование" for h in hits)


def test_get_without_state_change_is_fine(tmp_path):
    text = "# Д\n\n## API\n\n| Метод | GET |\n\nЧтение профиля клиента.\n"
    assert not [h for h in _lint_text(tmp_path, text) if h.rule == "API-GET"]
