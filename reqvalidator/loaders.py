"""Чтение входного документа: PDF, DOCX, Markdown, TXT, HTML → ``Document``.

PDF разбирается через pdfplumber: таблицы извлекаются построчно в виде
``| ячейка | ячейка |``, заголовки определяются по размеру шрифта, повторяющиеся
колонтитулы и номера страниц отбрасываются.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from reqvalidator.document import Document, Line, file_sha256

SUPPORTED_SUFFIXES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".html": "html",
    ".htm": "html",
}


class DocumentError(RuntimeError):
    """Документ не удалось прочитать."""


def load_document(path: str | Path) -> Document:
    path = Path(path)
    if not path.is_file():
        raise DocumentError(f"Файл не найден: {path}")
    fmt = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        if path.suffix.lower() == ".doc":
            raise DocumentError("Формат .doc не поддерживается: сохраните документ как .docx или .pdf.")
        fmt = "text"  # неизвестное расширение — пробуем как обычный текст
    loader = {
        "pdf": _load_pdf,
        "docx": _load_docx,
        "markdown": _load_markdown,
        "text": _load_text,
        "html": _load_html,
    }[fmt]
    lines, title, pages, warnings = loader(path)
    if fmt in ("markdown", "text"):
        # Номера строк должны совпадать с исходным файлом — пустые строки не трогаем.
        for ln in lines:
            ln.text = ln.text.rstrip()
    else:
        lines = _trim(lines)
    if not any(ln.text.strip() for ln in lines):
        raise DocumentError(
            f"В документе {path.name} не найден текст. Если это скан, распознайте его (OCR) перед проверкой."
        )
    for i, ln in enumerate(lines, 1):
        ln.no = i
    return Document(
        path=path,
        format=fmt,
        lines=lines,
        title=(title or path.stem).strip(),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        pages=pages,
        warnings=warnings,
    )


def _trim(lines: list[Line]) -> list[Line]:
    """Убрать пустые строки в начале/конце и схлопнуть серии пустых строк."""
    out: list[Line] = []
    for ln in lines:
        ln.text = ln.text.rstrip()
        if not ln.text.strip() and (not out or not out[-1].text.strip()):
            continue
        out.append(ln)
    while out and not out[-1].text.strip():
        out.pop()
    return out


# ---------------------------------------------------------------------------- текст / Markdown
def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+([A-ZА-ЯЁ].{1,90})$")


def _load_markdown(path: Path):
    text = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[Line] = []
    title = None
    in_code = False
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
        level = 0
        m = _MD_HEADING.match(stripped) if not in_code else None
        if m:
            level = len(m.group(1))
            if title is None and level == 1:
                title = m.group(2).strip()
        lines.append(Line(no=0, text=raw, heading_level=level))
    if title is None:
        title = next((ln.text.strip() for ln in lines if ln.text.strip()), None)
    # Markdown: номера строк совпадают с исходным файлом, поэтому пустые строки не удаляем.
    return lines, title, None, []


def _load_text(path: Path):
    text = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[Line] = []
    for raw in text.split("\n"):
        stripped = raw.strip()
        level = 0
        m = _NUM_HEADING.match(stripped)
        if m and not stripped.endswith((".", ";", ",", ":")):
            level = m.group(1).count(".") + 1
        lines.append(Line(no=0, text=raw, heading_level=level))
    title = next((ln.text.strip() for ln in lines if ln.text.strip()), None)
    return lines, title, None, []


# ---------------------------------------------------------------------------- DOCX
def _load_docx(path: Path):
    try:
        import docx  # python-docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover - зависимость обязательная
        raise DocumentError("Для чтения DOCX установите пакет python-docx") from exc

    try:
        d = docx.Document(str(path))
    except Exception as exc:
        raise DocumentError(f"Не удалось открыть DOCX: {exc}") from exc

    lines: list[Line] = []
    title = (d.core_properties.title or "").strip() or None
    para_no = 0
    table_no = 0
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, d)
            text = para.text.strip()
            para_no += 1
            if not text:
                lines.append(Line(no=0, text=""))
                continue
            style = (para.style.name if para.style is not None else "") or ""
            level = _docx_heading_level(style)
            if level:
                if title is None and level == 1:
                    title = text
                lines.append(
                    Line(no=0, text=f"{'#' * level} {text}", heading_level=level, ref=f"абзац {para_no}")
                )
            else:
                is_list = child.find(".//{*}numPr") is not None or "list" in style.lower()
                prefix = "- " if is_list else ""
                for part in text.split("\n"):
                    lines.append(Line(no=0, text=prefix + part, ref=f"абзац {para_no}"))
        elif tag == "tbl":
            table_no += 1
            table = Table(child, d)
            lines.append(Line(no=0, text=""))
            for r_idx, row in enumerate(table.rows, 1):
                cells: list[str] = []
                for cell in row.cells:
                    value = " ".join(cell.text.split())
                    if cells and cells[-1] == value and value:
                        continue  # объединённые ячейки python-docx отдаёт повторно
                    cells.append(value)
                lines.append(
                    Line(
                        no=0, text="| " + " | ".join(cells) + " |", ref=f"таблица {table_no}, строка {r_idx}"
                    )
                )
            lines.append(Line(no=0, text=""))
    if title is None:
        title = next((ln.text.lstrip("# ").strip() for ln in lines if ln.text.strip()), None)
    return lines, title, None, []


def _docx_heading_level(style: str) -> int:
    s = style.lower()
    if s in ("title", "название", "заголовок"):
        return 1
    m = re.match(r"(heading|заголовок)\s*(\d)", s)
    if m:
        return int(m.group(2))
    return 0


# ---------------------------------------------------------------------------- PDF
def _load_pdf(path: Path):
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise DocumentError("Для чтения PDF установите пакет pdfplumber") from exc

    warnings: list[str] = []
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise DocumentError(f"Не удалось открыть PDF: {exc}") from exc

    with pdf:
        meta_title = (pdf.metadata or {}).get("Title")
        n_pages = len(pdf.pages)
        # Элементы страницы: (page, top, kind, text, size, rows)
        items: list[dict] = []
        for pno, page in enumerate(pdf.pages, 1):
            items.extend(_pdf_page_items(page, pno))

    if not items:
        raise DocumentError("PDF не содержит текстового слоя. Распознайте скан (OCR) и повторите проверку.")

    items = _drop_running_headers(items, n_pages)
    body_size = _body_font_size(items)
    for it in items:
        it["heading"] = _heading_kind(it, body_size)
    sized = sorted({round(it["size"], 1) for it in items if it.get("heading") == "size"}, reverse=True)
    level_of = {size: min(i + 1, 6) for i, size in enumerate(sized)}
    bold_level = min(len(sized) + 1, 6)
    items = _merge_paragraphs(items)

    lines: list[Line] = []
    title = None
    last_page = None
    page_line = 0
    for it in items:
        if it["page"] != last_page:
            last_page = it["page"]
            page_line = 0
        if it["kind"] == "table":
            for row in it["rows"]:
                page_line += 1
                lines.append(Line(no=0, text=row, page=it["page"], page_line=page_line))
            continue
        page_line += 1
        kind = it.get("heading")
        level = (
            level_of.get(round(it["size"], 1), 0) if kind == "size" else (bold_level if kind == "bold" else 0)
        )
        text = it["text"]
        if level:
            text = f"{'#' * level} {text}"
            if title is None and it["page"] == 1:
                title = it["text"]
        lines.append(Line(no=0, text=text, heading_level=level, page=it["page"], page_line=page_line))
        page_line += it.get("extra_lines", 0)

    if (
        not title
        and meta_title
        and not re.search(r"untitled|microsoft word|\.docx?$|\.pdf$", meta_title, re.I)
    ):
        title = meta_title
    if not title:
        title = next((ln.text.strip() for ln in lines if ln.text.strip()), None)
    return lines, title, n_pages, warnings


def _pdf_page_items(page, pno: int) -> list[dict]:
    items: list[dict] = []
    try:
        tables = page.find_tables()
    except Exception:  # pragma: no cover - редкие повреждённые страницы
        tables = []
    bboxes = []
    for t in tables:
        try:
            rows = t.extract()
        except Exception:  # pragma: no cover
            continue
        clean_rows = []
        for row in rows:
            cells = [_WRAP_HYPHEN.sub(r"\1-\2", " ".join((c or "").split())) for c in row]
            if not any(cells):
                continue
            clean_rows.append("| " + " | ".join(cells) + " |")
        if not clean_rows:
            continue
        # Таблица из одной ячейки — чаще всего рамка блока кода, а не таблица.
        if len(clean_rows) == 1 and len(rows[0]) == 1:
            continue
        bboxes.append(t.bbox)
        items.append(
            {
                "page": pno,
                "top": t.bbox[1],
                "bottom": t.bbox[3],
                "x0": t.bbox[0],
                "kind": "table",
                "rows": clean_rows,
                "size": 0,
                "text": "",
            }
        )

    def inside(word) -> bool:
        cx = (word["x0"] + word["x1"]) / 2
        cy = (word["top"] + word["bottom"]) / 2
        return any(x0 - 1 <= cx <= x1 + 1 and top - 1 <= cy <= bottom + 1 for x0, top, x1, bottom in bboxes)

    words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False, x_tolerance=1.5)
    words = [w for w in words if not inside(w)]
    words.sort(key=lambda w: (round(w["top"]), w["x0"]))
    current: list[dict] = []
    for w in words:
        if current and abs(w["top"] - current[0]["top"]) > max(2.5, current[0]["size"] * 0.35):
            items.append(_pdf_line(current, pno, page.height))
            current = []
        current.append(w)
    if current:
        items.append(_pdf_line(current, pno, page.height))
    items.sort(key=lambda it: it["top"])
    return items


def _pdf_line(words: list[dict], pno: int, height: float) -> dict:
    words = sorted(words, key=lambda w: w["x0"])
    parts: list[str] = []
    prev = None
    for w in words:
        if prev is not None and w["x0"] - prev["x1"] < 0.8:
            parts[-1] += w["text"]  # слово разбито на несколько фрагментов без пробела
        else:
            parts.append(w["text"])
        prev = w
    sizes = sorted(w["size"] for w in words)
    size = sizes[len(sizes) // 2]
    bold = sum("bold" in (w.get("fontname") or "").lower() for w in words) > len(words) / 2
    return {
        "page": pno,
        "top": min(w["top"] for w in words),
        "bottom": max(w["bottom"] for w in words),
        "x0": words[0]["x0"],
        "rel": words[0]["top"] / max(1.0, height),
        "kind": "text",
        "text": " ".join(parts),
        "size": size,
        "bold": bold,
    }


def _body_font_size(items: list[dict]) -> float:
    counter: Counter = Counter()
    for it in items:
        if it["kind"] == "text":
            counter[round(it["size"], 1)] += len(it["text"])
    return counter.most_common(1)[0][0] if counter else 10.0


def _heading_kind(it: dict, body_size: float) -> str | None:
    """«size» — заголовок по кеглю, «bold» — короткая полужирная строка обычного кегля."""
    if it["kind"] != "text":
        return None
    text = it["text"].strip()
    if not (1 < len(text) <= 120) or text.endswith((",", ";", ":")):
        return None
    if it["size"] >= body_size * 1.05:
        return "size"
    if it.get("bold") and it["size"] >= body_size * 0.97 and len(text) <= 90 and not text.endswith("."):
        return "bold"
    return None


_LIST_START = re.compile(r"^(?:[-•–*]\s|\d{1,3}[.)]\s|[A-Za-zА-Яа-яЁё]{1,4}-\d+[.)]?\s|\(?[a-zа-я]\)\s)")
_WRAP_HYPHEN = re.compile(r"(\w)- (\w)")


def _merge_paragraphs(items: list[dict]) -> list[dict]:
    """Склеить переносы строк внутри абзаца: одинаковый кегль, обычный интервал, без маркера списка."""
    out: list[dict] = []
    for it in items:
        prev = out[-1] if out else None
        if (
            prev is not None
            and it["kind"] == "text"
            and prev["kind"] == "text"
            and not prev.get("heading")
            and not it.get("heading")
            and prev["page"] == it["page"]
            and abs(prev["size"] - it["size"]) < 0.3
            and 0 <= it["top"] - prev["bottom"] < it["size"] * 0.9
            and it["x0"] <= prev["x0"] + 2
            and not _LIST_START.match(it["text"])
            and not (prev["text"].rstrip().endswith(":") and not it["text"][:1].islower())
            and not it["text"].lstrip().startswith(("{", "[", "|"))
        ):
            joiner = "" if prev["text"].endswith("-") and it["text"][:1].islower() else " "
            prev["text"] = prev["text"] + joiner + it["text"]
            prev["bottom"] = it["bottom"]
            prev["extra_lines"] = prev.get("extra_lines", 0) + 1
            continue
        out.append(it)
    for it in out:
        if it["kind"] == "text":
            it["text"] = _WRAP_HYPHEN.sub(r"\1-\2", it["text"])
    return out


_PAGE_NO = re.compile(r"^(?:стр\.?|страница|page|с\.)?\s*\d{1,4}(?:\s*(?:/|из|of)\s*\d{1,4})?$", re.I)


def _drop_running_headers(items: list[dict], n_pages: int) -> list[dict]:
    """Удалить колонтитулы: повторяющиеся на многих страницах строки у края листа и номера страниц."""

    def key(it: dict) -> str:
        return re.sub(r"\d+", "#", it["text"].strip().lower())

    edge = [it for it in items if it["kind"] == "text" and (it["rel"] < 0.07 or it["rel"] > 0.93)]
    pages_by_key: dict[str, set[int]] = {}
    for it in edge:
        pages_by_key.setdefault(key(it), set()).add(it["page"])
    threshold = max(2, (n_pages + 1) // 2)
    repeated = {k for k, pages in pages_by_key.items() if len(pages) >= threshold and n_pages > 1}
    out = []
    for it in items:
        at_edge = it["kind"] == "text" and (it["rel"] < 0.07 or it["rel"] > 0.93)
        if at_edge and (key(it) in repeated or _PAGE_NO.match(it["text"].strip())):
            continue
        out.append(it)
    return out


# ---------------------------------------------------------------------------- HTML
class _HTMLToLines(HTMLParser):
    BLOCK: ClassVar[set[str]] = {
        "p",
        "div",
        "li",
        "tr",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "blockquote",
        "table",
        "section",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[Line] = []
        self.buf: list[str] = []
        self.level = 0
        self.cells: list[str] | None = None
        self.cell_buf: list[str] | None = None
        self.title: str | None = None
        self._in_title = False
        self._skip = 0
        self._li = False

    def _flush(self) -> None:
        text = " ".join("".join(self.buf).split())
        if text:
            prefix = "#" * self.level + " " if self.level else ("- " if self._li else "")
            self.lines.append(Line(no=0, text=prefix + text, heading_level=self.level))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in ("td", "th"):
            self.cell_buf = []
            return
        if tag == "tr":
            self._flush()
            self.cells = []
            return
        if tag in self.BLOCK:
            self._flush()
            self.level = int(tag[1]) if re.fullmatch(r"h[1-6]", tag) else 0
            self._li = tag == "li"

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in ("td", "th") and self.cell_buf is not None and self.cells is not None:
            self.cells.append(" ".join("".join(self.cell_buf).split()))
            self.cell_buf = None
            return
        if tag == "tr" and self.cells is not None:
            if any(self.cells):
                self.lines.append(Line(no=0, text="| " + " | ".join(self.cells) + " |"))
            self.cells = None
            return
        if tag in self.BLOCK:
            if self.level == 1 and self.title is None:
                self.title = " ".join("".join(self.buf).split()) or None
            self._flush()
            if tag in ("p", "table", "pre", "blockquote") or self.level:
                self.lines.append(Line(no=0, text=""))
            self.level = 0
            self._li = False

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
            return
        if self.cell_buf is not None:
            self.cell_buf.append(data)
        else:
            self.buf.append(data)


def _load_html(path: Path):
    parser = _HTMLToLines()
    parser.feed(_read_text(path))
    parser._flush()
    return parser.lines, parser.title, None, []
