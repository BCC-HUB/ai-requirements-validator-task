from reqvalidator.document import Document, Line, normalize
from reqvalidator.loaders import load_document


def _doc(lines: list[str], fmt: str = "markdown") -> Document:
    return Document(
        path=__import__("pathlib").Path("x.md"),
        format=fmt,
        lines=[
            Line(no=i, text=t, heading_level=(len(t) - len(t.lstrip("#")) if t.startswith("#") else 0))
            for i, t in enumerate(lines, 1)
        ],
        title="t",
        sha256="0",
        size_bytes=0,
    )


def test_normalize_unifies_quotes_dashes_yo_and_markup():
    assert normalize("«Ёлка» — **важно**  `код`") == normalize('"елка" - важно код')
    assert normalize("A B") == "a b"


def test_locate_exact_and_multiline():
    d = _doc(["# Заголовок", "Первая строка текста", "продолжение абзаца здесь", "Другое"])
    m = d.locate("Первая строка текста продолжение абзаца")
    assert (m.line_start, m.line_end, m.exact) == (2, 3, True)


def test_locate_prefers_occurrence_near_hint():
    d = _doc(["повтор фразы тут", "x", "y", "z", "повтор фразы тут"])
    assert d.locate("повтор фразы", near=5).line_start == 5
    assert d.locate("повтор фразы", near=1).line_start == 1


def test_locate_elided_and_fuzzy_quotes():
    d = _doc(["Держатель карты задаёт порог от 1 000 до 5 000 включительно. По умолчанию выключено."])
    assert d.locate("Держатель карты задаёт порог … По умолчанию выключено").line_start == 1
    fuzzy = d.locate("Держатель карты задает порог от 1000 до 5 000 включительно")
    assert fuzzy is not None and not fuzzy.exact and fuzzy.score >= 0.8


def test_locate_returns_none_for_invented_text():
    d = _doc(["Система отправляет письмо", "Пользователь входит"])
    assert d.locate("Квантовый ускоритель синхронизирует галактики") is None
    assert d.locate("ab") is None


def test_location_labels_by_format():
    md = _doc(["a", "b", "c"])
    assert md.location_label(2) == "строка 2"
    assert md.location_label(1, 3) == "строки 1–3"
    pdf = _doc(["a", "b", "c"], fmt="pdf")
    for i, ln in enumerate(pdf.lines):
        ln.page, ln.page_line = (1, i + 1) if i < 2 else (2, 1)
    assert pdf.location_label(1, 2) == "с. 1, строки 1–2"
    assert pdf.location_label(2, 3) == "с. 1, строка 2 — с. 2, строка 1"


def test_sections_skip_single_top_title(fixtures):
    d = load_document(fixtures / "sample_requirements.md")
    assert d.section_of(7) == "Функциональные требования"
    assert d.section_of(18) == "API"
    assert d.outline()[0] == (1, 1, "Сервис бронирования переговорных")


def test_numbered_text_has_line_prefixes(fixtures):
    d = load_document(fixtures / "sample_requirements.md")
    first = d.numbered_text().splitlines()[0]
    assert first.startswith("L01| # Сервис")
