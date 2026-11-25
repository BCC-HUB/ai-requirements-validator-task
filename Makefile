# Короткие команды для разработки. Основной запуск: python -m reqvalidator <документ> <отчёт>
PY ?= python

.PHONY: install dev lint test run

install:
	$(PY) -m pip install -r requirements.txt

dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check reqvalidator tests
	ruff format --check reqvalidator tests

test:
	$(PY) -m pytest -q

run:
	$(PY) -m reqvalidator docs/requirements.md reports/requirements-report.md
