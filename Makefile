.PHONY: setup format lint test run

setup:
	pip install -e .[dev]
	pre-commit install

format:
	black .
	isort .

lint:
	ruff .

test:
	pytest -q

run:
	python -m vgj_chat
