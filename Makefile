.DEFAULT_GOAL := help

.PHONY: help install install-training lint format test check notebook validate-data

help: ## Показать доступные команды
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*##/ {printf "%-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Установить окружение для разработки и notebook
	uv sync --extra notebook --extra dev

install-training: ## Установить GLiNER2 и зависимости для обучения
	uv sync --extra notebook --extra training --extra dev

lint: ## Проверить стиль и типовые ошибки
	uv run ruff check .

format: ## Отформатировать код
	uv run ruff format .

test: ## Запустить unit-тесты
	uv run pytest

check: lint test ## Полная быстрая проверка

notebook: ## Открыть учебный JupyterLab
	uv run jupyter lab notebooks/

validate-data: ## Проверить синтетический пример датасета
	uv run python scripts/validate_dataset.py data/synthetic/demo.jsonl
