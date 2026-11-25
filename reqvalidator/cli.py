"""Командная строка: ``python -m reqvalidator <документ> <отчёт.md>``."""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

from reqvalidator import __version__
from reqvalidator.config import PROVIDERS, ConfigError, load_settings
from reqvalidator.llm import LLMAbortError, LLMError, LLMSession, ResponseCache, create_client
from reqvalidator.loaders import SUPPORTED_SUFFIXES, DocumentError, load_document
from reqvalidator.pipeline import Pipeline, PipelineOptions
from reqvalidator.report import READY_TEXT, write_report
from reqvalidator.version import agent_version

log = logging.getLogger("reqvalidator")

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_DOCUMENT, EXIT_LLM = 0, 1, 2, 3, 4


def build_parser() -> argparse.ArgumentParser:
    formats = ", ".join(sorted({s.lstrip(".") for s in SUPPORTED_SUFFIXES}))
    p = argparse.ArgumentParser(
        prog="reqvalidator",
        description=(
            "ИИ-агент статической проверки документа требований по ISO/IEC/IEEE 29148:2018 "
            "и IEEE Std 830-1998. Читает документ, проверяет каждое требование и набор целиком, "
            "записывает отчёт на русском языке."
        ),
        epilog="Пример: python -m reqvalidator docs/requirements.md reports/report.md",
    )
    p.add_argument("input", nargs="?", type=Path, help=f"файл требований ({formats})")
    p.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="файл отчёта (.md; при расширении .json — только JSON). По умолчанию reports/<имя>-report.md",
    )
    p.add_argument("--provider", choices=PROVIDERS, help="провайдер LLM (по умолчанию LLM_PROVIDER из .env)")
    p.add_argument("--model", help="модель (по умолчанию из .env, например GEMINI_MODEL)")
    p.add_argument("--effort", help="уровень рассуждений для anthropic / claude-cli (low … max)")
    p.add_argument("--env-file", type=Path, help="путь к файлу настроек (по умолчанию ./.env)")
    p.add_argument("--batch-size", type=int, help="требований в одном запросе проверки (по умолчанию 6)")
    p.add_argument("--concurrency", type=int, help="параллельных запросов к модели")
    p.add_argument("--no-review", action="store_true", help="не выполнять этап самопроверки замечаний")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    p.add_argument(
        "--json", dest="json_path", type=Path, help="куда записать JSON (по умолчанию рядом с отчётом)"
    )
    p.add_argument("--no-json", action="store_true", help="не записывать JSON-копию результатов")
    p.add_argument("--list-models", action="store_true", help="показать модели, доступные ключу, и выйти")
    p.add_argument("-v", "--verbose", action="store_true", help="подробный журнал")
    p.add_argument("-q", "--quiet", action="store_true", help="только ошибки и итог")
    p.add_argument("--version", action="version", version=f"reqvalidator {__version__}")
    return p


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(message)s" if not verbose else "%(asctime)s %(levelname)s %(message)s")
    )
    root = logging.getLogger("reqvalidator")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    for noisy in ("urllib3", "pdfminer", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose, args.quiet)

    try:
        settings = load_settings(env_file=args.env_file, provider=args.provider, model=args.model)
        settings = settings.with_overrides(
            effort=args.effort,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )
        if args.no_review:
            settings = settings.with_overrides(review=False)
        client = create_client(settings)
    except (ConfigError, LLMError) as exc:
        log.error("Ошибка настройки: %s", exc)
        return EXIT_CONFIG

    if args.list_models:
        try:
            for name in client.list_models():
                print(name)
        except (LLMError, LLMAbortError) as exc:
            log.error("Не удалось получить список моделей: %s", exc)
            return EXIT_LLM
        return EXIT_OK

    if args.input is None:
        build_parser().print_usage(sys.stderr)
        log.error(
            "Укажите путь к файлу требований, например: python -m reqvalidator docs/requirements.md "
            "reports/report.md"
        )
        return EXIT_CONFIG

    output: Path = args.output or Path("reports") / f"{args.input.stem}-report.md"
    json_path = (
        None
        if args.no_json or output.suffix.lower() == ".json"
        else (args.json_path or output.with_suffix(".json"))
    )

    try:
        doc = load_document(args.input)
    except DocumentError as exc:
        log.error("Ошибка чтения документа: %s", exc)
        return EXIT_DOCUMENT
    log.info(
        "Документ: «%s» (%s, %d строк%s)",
        doc.title,
        doc.format_name,
        doc.line_count,
        f", {doc.pages} стр." if doc.pages else "",
    )
    log.info("Модель: %s / %s", settings.provider, settings.model_label)

    cache = None if args.no_cache or settings.cache_dir is None else ResponseCache(settings.cache_dir)
    session = LLMSession(
        client,
        max_retries=settings.max_retries,
        min_interval=settings.min_interval,
        cache=cache,
    )
    version = agent_version(ignore_paths=[p for p in (output, json_path) if p])
    pipeline = Pipeline(
        session,
        settings,
        PipelineOptions(
            batch_size=settings.batch_size, concurrency=settings.concurrency, review=settings.review
        ),
    )
    t0 = time.monotonic()
    try:
        result = pipeline.run(doc, version)
    except LLMAbortError as exc:
        log.error("Прогон остановлен, отчёт не записан: %s", exc)
        return EXIT_LLM
    except LLMError as exc:
        log.error("Ошибка обращения к модели, отчёт не записан: %s", exc)
        return EXIT_LLM
    except KeyboardInterrupt:
        log.error("Прервано пользователем")
        return EXIT_ERROR

    written = write_report(result, output, json_path)
    st = result.stats
    v = result.verdict
    print(f"Отчёт записан: {written[0]}" + (f" (JSON: {written[1]})" if len(written) > 1 else ""))
    print(
        f"Требований: {st.requirements}; замечаний: {st.findings_total} "
        f"(критичных {st.by_severity.get('critical', 0)}, существенных {st.by_severity.get('major', 0)}, "
        f"незначительных {st.by_severity.get('minor', 0)}); время: {time.monotonic() - t0:.0f} с"
    )
    if v:
        print(f"Итог: разработка — {READY_TEXT[v.development][1]}; тестирование — {READY_TEXT[v.testing][1]}")
    for w in result.run.warnings:
        log.warning("Предупреждение: %s", w)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
