"""Модель документа: пронумерованные строки, разделы и поиск цитат.

Все форматы приводятся к списку строк (``Line``). Номер строки — сквозной и
используется моделью как «адрес» фрагмента. Каждая строка знает, как назвать
своё место для человека (строка Markdown, страница PDF, абзац DOCX), поэтому
любое замечание в отчёте можно привязать к конкретному месту исходного файла.
"""

from __future__ import annotations

import bisect
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property
from pathlib import Path

FORMAT_NAMES = {
    "markdown": "Markdown",
    "text": "текст",
    "pdf": "PDF",
    "docx": "DOCX",
    "html": "HTML",
}


@dataclass
class Line:
    no: int
    text: str
    heading_level: int = 0
    page: int | None = None
    page_line: int | None = None
    ref: str | None = None  # человекочитаемая позиция, если она отличается от «строка N»

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0


@dataclass(frozen=True)
class QuoteMatch:
    line_start: int
    line_end: int
    score: float
    exact: bool


_QUOTES_RE = re.compile(r"[«»“”„‟\"'’‘`´]")
_DASHES_RE = re.compile(r"[‐‑‒–—―−]")
_MARKUP_RE = re.compile(r"[*_#|>\\]+")
_SPACES_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\]|\(\.\.\.\))\s*")


def normalize(text: str) -> str:
    """Нормализация для сравнения цитат: регистр, ё/е, кавычки, тире, разметка, пробелы."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    text = _QUOTES_RE.sub('"', text)
    text = _DASHES_RE.sub("-", text)
    text = _MARKUP_RE.sub(" ", text)
    text = text.replace('"', "")
    return _SPACES_RE.sub(" ", text).strip()


@dataclass
class Document:
    path: Path
    format: str
    lines: list[Line]
    title: str
    sha256: str
    size_bytes: int
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ базовое
    @property
    def format_name(self) -> str:
        return FORMAT_NAMES.get(self.format, self.format)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def line(self, no: int) -> Line:
        return self.lines[no - 1]

    def clamp(self, no: int) -> int:
        return max(1, min(self.line_count, no))

    def text_of(self, start: int, end: int | None = None) -> str:
        end = end or start
        start, end = self.clamp(min(start, end)), self.clamp(max(start, end))
        return "\n".join(ln.text for ln in self.lines[start - 1 : end])

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    def numbered_text(self) -> str:
        """Текст для модели: каждая строка с номером «L12| …», пустые строки сохраняются."""
        width = len(str(self.line_count))
        return "\n".join(f"L{ln.no:0{width}d}| {ln.text}" for ln in self.lines)

    # ------------------------------------------------------------------ разделы
    @cached_property
    def _section_paths(self) -> list[str]:
        stack: list[tuple[int, str]] = []
        paths: list[str] = []
        for ln in self.lines:
            if ln.is_heading:
                title = _MARKUP_RE.sub(" ", ln.text).strip()
                title = _SPACES_RE.sub(" ", title)
                while stack and stack[-1][0] >= ln.heading_level:
                    stack.pop()
                stack.append((ln.heading_level, title))
            # Заголовок первого уровня — обычно название документа; в путь его не включаем,
            # если он единственный такого уровня.
            visible = [t for lvl, t in stack if not (lvl == self._top_level and self._single_top)]
            paths.append(" › ".join(visible))
        return paths

    @cached_property
    def _top_level(self) -> int:
        levels = [ln.heading_level for ln in self.lines if ln.is_heading]
        return min(levels) if levels else 0

    @cached_property
    def _single_top(self) -> bool:
        """Единственный заголовок верхнего уровня в размеченном документе — это название документа."""
        if self.format == "text":
            return False
        return sum(1 for ln in self.lines if ln.is_heading and ln.heading_level == self._top_level) == 1

    def section_of(self, line_no: int) -> str:
        if not self.lines:
            return ""
        return self._section_paths[self.clamp(line_no) - 1]

    def outline(self) -> list[tuple[int, int, str]]:
        """Список заголовков: (номер строки, уровень, текст)."""
        return [(ln.no, ln.heading_level, ln.text.lstrip("# ").strip()) for ln in self.lines if ln.is_heading]

    # ------------------------------------------------------------------ места
    def location_label(self, start: int, end: int | None = None) -> str:
        end = end or start
        start, end = self.clamp(min(start, end)), self.clamp(max(start, end))
        first, last = self.line(start), self.line(end)
        if self.format == "pdf":
            if first.page == last.page:
                if first.page_line == last.page_line:
                    return f"с. {first.page}, строка {first.page_line}"
                return f"с. {first.page}, строки {first.page_line}–{last.page_line}"
            return f"с. {first.page}, строка {first.page_line} — с. {last.page}, строка {last.page_line}"
        if self.format == "docx":
            if start == end:
                return first.ref or f"фрагмент {start}"
            return f"{first.ref or start} — {last.ref or end}"
        if start == end:
            return f"строка {start}"
        return f"строки {start}–{end}"

    # ------------------------------------------------------------------ цитаты
    @cached_property
    def _norm_index(self) -> tuple[str, list[int], list[str]]:
        norm_lines = [normalize(ln.text) for ln in self.lines]
        offsets: list[int] = []
        pos = 0
        parts: list[str] = []
        for nl in norm_lines:
            offsets.append(pos)
            parts.append(nl)
            pos += len(nl) + 1
        return " ".join(parts), offsets, norm_lines

    def _offset_to_line(self, offset: int) -> int:
        _, offsets, _ = self._norm_index
        return bisect.bisect_right(offsets, offset)  # 1-based номер строки

    def _exact(self, needle: str, near: int | None) -> QuoteMatch | None:
        hay, _, _ = self._norm_index
        if not needle:
            return None
        hits = [m.start() for m in re.finditer(re.escape(needle), hay)]
        if not hits:
            return None
        if near is not None and len(hits) > 1:
            hits.sort(key=lambda off: abs(self._offset_to_line(off) - near))
        start = hits[0]
        return QuoteMatch(
            self._offset_to_line(start), self._offset_to_line(start + len(needle) - 1), 1.0, True
        )

    def locate(self, quote: str, near: int | None = None, min_score: float = 0.8) -> QuoteMatch | None:
        """Найти цитату в документе: сначала точно (после нормализации), затем нечётко.

        ``near`` — подсказка модели о номере строки; используется, когда цитата
        встречается несколько раз, и для ограничения нечёткого поиска.
        """
        needle = normalize(quote)
        if len(needle) < 3:
            return None
        hit = self._exact(needle, near)
        if hit:
            return hit
        # Цитата с пропусками («начало … конец»): ищем фрагменты по отдельности.
        pieces = [normalize(p) for p in _ELLIPSIS_RE.split(quote) if len(normalize(p)) >= 6]
        if len(pieces) > 1:
            found = [self._exact(p, near) for p in pieces]
            if all(found):
                lines = [f.line_start for f in found] + [f.line_end for f in found]  # type: ignore[union-attr]
                if max(lines) - min(lines) <= 12:
                    return QuoteMatch(min(lines), max(lines), 0.95, False)
        return self._fuzzy(needle, near, min_score)

    def _fuzzy(self, needle: str, near: int | None, min_score: float) -> QuoteMatch | None:
        _, _, norm_lines = self._norm_index
        best: QuoteMatch | None = None
        needle_tokens = set(needle.split())
        candidates = range(1, self.line_count + 1)
        for start in candidates:
            if not norm_lines[start - 1]:
                continue
            window = ""
            for end in range(start, min(start + 4, self.line_count + 1)):
                window = (window + " " + norm_lines[end - 1]).strip()
                if len(window) < len(needle) * 0.5:
                    continue
                # Дешёвый фильтр: окно должно содержать хотя бы половину слов цитаты.
                overlap = len(needle_tokens & set(window.split())) / max(1, len(needle_tokens))
                if overlap < 0.5:
                    continue
                score = _similarity(needle, window)
                if near is not None:
                    score -= min(abs(start - near), 50) * 0.0005  # лёгкое предпочтение подсказке
                if best is None or score > best.score:
                    best = QuoteMatch(start, end, score, False)
                if len(window) > len(needle) * 2.5:
                    break
        if best and best.score >= min_score:
            return QuoteMatch(best.line_start, best.line_end, round(min(best.score, 0.99), 3), False)
        return None


def _similarity(needle: str, window: str) -> float:
    """Сходство цитаты с окном текста: полное сравнение или покрытие цитаты внутри окна."""
    sm = SequenceMatcher(None, needle, window, autojunk=False)
    ratio = sm.ratio()
    if len(window) > len(needle):
        matched = sum(block.size for block in sm.get_matching_blocks())
        coverage = matched / max(1, len(needle))
        # Покрытие засчитываем, только если совпавшие куски достаточно крупные.
        big = sum(b.size for b in sm.get_matching_blocks() if b.size >= 4) / max(1, len(needle))
        ratio = max(ratio, min(coverage, big))
    return ratio


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
