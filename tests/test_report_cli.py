import json

import pytest

from reqvalidator import cli
from reqvalidator.llm import LLMSession
from reqvalidator.loaders import load_document
from reqvalidator.pipeline import Pipeline, PipelineOptions
from reqvalidator.report import cell, plural, render_markdown, write_report
from reqvalidator.version import AgentVersion
from tests.fakes import FakeClient

REQUIRED_SECTIONS = (
    "## 1. Паспорт прогона",
    "## 3. Как агент нарезал требования",
    "## 4. Проверка каждого требования",
    "## 5. Проверка всего набора требований",
    "## 6. Сводка",
    "### Критичные дефекты",
    "## 7. Итог",
    "### Что исправить в первую очередь",
)


@pytest.fixture
def result(fixtures):
    client = FakeClient()
    doc = load_document(fixtures / "sample_requirements.md")
    pipe = Pipeline(LLMSession(client, sleep=lambda s: None), client.settings, PipelineOptions(batch_size=3))
    return pipe.run(doc, AgentVersion("1.0.0", commit="abc123def456"))


def test_report_contains_all_required_sections(result):
    md = render_markdown(result)
    pos = [md.index(s) for s in REQUIRED_SECTIONS]
    assert pos == sorted(pos), "разделы идут в порядке, заданном в ТЗ"
    assert "«Сервис бронирования переговорных»" in md
    assert "`fake-model`" in md and "коммит abc123def456" in md
    assert "| [ТР-01](#tr-01) | REQ-1 |" in md
    assert '<a id="tr-01"></a>' in md and "(#d-01)" in md
    assert "«Сервис должен быть удобным и надёжным и т.д.»" in md
    assert "Непроверяемые формулировки" in md and "Противоречивые формулировки" in md
    for f in result.findings:
        if f.status == "confirmed":
            assert f"**{f.id}**" in md


def test_report_tables_are_well_formed(result):
    md = render_markdown(result)
    for line in md.splitlines():
        if line.startswith("| ") and not line.startswith("| :") and not line.startswith("| ---"):
            # Экранированные «\|» не должны ломать число колонок.
            unescaped = line.replace("\\|", "")
            assert unescaped.count("|") >= 3


def test_write_report_md_and_json(result, tmp_path):
    md_path, json_path = tmp_path / "out" / "r.md", tmp_path / "out" / "r.json"
    written = write_report(result, md_path, json_path)
    assert written == [md_path, json_path]
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["run"]["document_title"] == "Сервис бронирования переговорных"
    assert len(data["requirements"]) == 6
    only_json = write_report(result, tmp_path / "only.json")
    assert only_json == [tmp_path / "only.json"]


def test_helpers():
    assert plural(1, ("требование", "требования", "требований")) == "1 требование"
    assert plural(3, ("требование", "требования", "требований")) == "3 требования"
    assert plural(11, ("требование", "требования", "требований")) == "11 требований"
    assert plural(25, ("требование", "требования", "требований")) == "25 требований"
    assert cell("a|b\nc") == "a\\|b<br>c"


# ---------------------------------------------------------------------------- CLI
@pytest.fixture
def fake_cli(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("REQVALIDATOR_CACHE_DIR", "")
    monkeypatch.setattr(cli, "create_client", lambda settings: FakeClient(settings))


def test_cli_one_command_run(fake_cli, fixtures, tmp_path, capsys):
    out = tmp_path / "report.md"
    code = cli.main([str(fixtures / "sample_requirements.md"), str(out), "-q"])
    assert code == cli.EXIT_OK
    assert out.is_file() and out.with_suffix(".json").is_file()
    printed = capsys.readouterr().out
    assert "Отчёт записан" in printed and "Итог:" in printed


def test_cli_pdf_input(fake_cli, fixtures, tmp_path):
    out = tmp_path / "pdf-report.md"
    assert cli.main([str(fixtures / "sample_requirements.pdf"), str(out), "-q", "--no-json"]) == cli.EXIT_OK
    assert "(PDF," in out.read_text(encoding="utf-8")
    assert not out.with_suffix(".json").exists()


def test_cli_errors(monkeypatch, fixtures, tmp_path):
    # Нет ключа — ошибка конфигурации с подсказкой.
    assert cli.main([str(fixtures / "sample_requirements.md"), str(tmp_path / "r.md")]) == cli.EXIT_CONFIG
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(cli, "create_client", lambda settings: FakeClient(settings))
    assert cli.main([str(tmp_path / "missing.pdf"), str(tmp_path / "r.md")]) == cli.EXIT_DOCUMENT
    assert cli.main([]) == cli.EXIT_CONFIG


def test_locator_is_not_repeated_when_it_equals_section(result):
    r = result.requirements[0]
    result.requirements[0] = r.model_copy(update={"section": "Часть A › Цели", "locator": "Цели"})
    result.requirements[1] = result.requirements[1].model_copy(update={"locator": "Таблица: поле limit"})
    md = render_markdown(result)
    assert "Часть A › Цели · Цели" not in md and "| Часть A › Цели |" in md
    assert "· Таблица: поле limit" in md
