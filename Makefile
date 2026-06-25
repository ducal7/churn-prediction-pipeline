.PHONY: data train evaluate score test lint format all clean

PY ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
PYTEST ?= .venv/bin/pytest

data:
	$(PY) -m churn data

train:
	$(PY) -m churn train

evaluate:
	$(PY) -m churn evaluate

score:
	$(PY) -m churn score

test:
	$(PYTEST)

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

all: data train evaluate test lint

clean:
	rm -rf data models
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
