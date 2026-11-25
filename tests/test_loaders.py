from pathlib import Path

import pytest

from reqvalidator.loaders import DocumentError, load_document

ROOT = Path(__file__).resolve().parent.parent


def test_markdown_keeps_source_line_numbers(fixtures):
    path = fixtures / "sample_requirements.md"
    d = load_document(path)
    source = path.read_text(encoding="utf-8").splitlines()
    assert d.format == "markdown"
    assert d.title == "Сервис бронирования переговорных"
    assert d.line_count == len(source)
    assert d.line(7).text == source[6]
    assert len(d.sha256) == 64


def test_task_document_loads():
    d = load_document(ROOT / "docs" / "requirements.md")
    assert d.title == "Уведомление о расходах по карте"
    assert any("Ф-2." in ln.text for ln in d.lines)


def test_pdf_extracts_headings_tables_and_drops_running_header(fixtures):
    d = load_document(fixtures / "sample_requirements.pdf")
    texts = [ln.text for ln in d.lines]
    assert d.format == "pdf" and d.pages == 1
    assert d.title == "Сервис бронирования переговорных"
    assert texts[0] == "# Сервис бронирования переговорных"  # колонтитул с тем же текстом удалён
    assert "## Функциональные требования" in texts
    assert any(t.startswith("REQ-1.") and t.endswith("подтверждается быстро.") for t in texts)
    assert "| REST type | GET |" in texts
    assert "| Свободная переговорная | Забронировать на 30 минут | Бронь создана |" in texts
    assert texts[-1].replace(" ", "") == "||||"  # пустая строка таблицы согласования сохранена
    m = d.locate("Система отправляет напоминание за 10 минут")
    assert d.location_label(m.line_start).startswith("с. 1, строка ")


def test_html_loader(fixtures):
    d = load_document(fixtures / "sample_requirements.html")
    texts = [ln.text for ln in d.lines]
    assert d.title == "Сервис бронирования переговорных"
    assert "## API" in texts
    assert "| REST type | GET |" in texts
    assert any(t.startswith("- REQ-2.") for t in texts)


def test_docx_loader(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("Спецификация платежей", level=1)
    doc.add_heading("Требования", level=2)
    doc.add_paragraph("FR-1. Система проводит платёж.", style="List Bullet")
    doc.add_paragraph("Обычный абзац.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Поле", "Значение"
    table.cell(1, 0).text, table.cell(1, 1).text = "Метод", "POST"
    path = tmp_path / "spec.docx"
    doc.save(path)

    d = load_document(path)
    texts = [ln.text for ln in d.lines]
    assert d.title == "Спецификация платежей"
    assert "## Требования" in texts
    assert "- FR-1. Система проводит платёж." in texts
    assert "| Метод | POST |" in texts
    row = next(ln for ln in d.lines if ln.text == "| Метод | POST |")
    assert d.location_label(row.no) == "таблица 1, строка 2"
    para = next(ln for ln in d.lines if ln.text == "Обычный абзац.")
    assert d.location_label(para.no).startswith("абзац ")


def test_txt_loader_cp1251_and_numbered_headings(tmp_path):
    path = tmp_path / "req.txt"
    path.write_bytes("1. Общие положения\nСистема должна работать.\n1.1 Детали\nТекст.\n".encode("cp1251"))
    d = load_document(path)
    assert d.line(1).heading_level == 1
    assert d.line(3).heading_level == 2
    assert d.line(2).heading_level == 0
    assert d.section_of(4) == "1. Общие положения › 1.1 Детали"


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing.md", None, "не найден"),
        ("old.doc", b"binary", ".docx"),
        ("empty.md", b"\n\n  \n", "не найден текст"),
    ],
)
def test_loader_errors(tmp_path, name, content, message):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(DocumentError, match=message):
        load_document(path)
