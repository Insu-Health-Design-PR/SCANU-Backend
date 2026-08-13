.PHONY: install api infer capture test lint clean devices kill destroy

PYTHON ?= python3
VENV ?= .venv
BIN = $(VENV)/bin
API_PORT ?= 8088

CONDA_ENV ?= scanu-training
CONDA_BASE ?= $(shell conda info --base 2>/dev/null)
CONDA_ACTIVATE = source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate "$(CONDA_ENV)" &&

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -U pip wheel
	$(BIN)/pip install -e ".[dev]"

api:
	@bash -c '$(CONDA_ACTIVATE) python -c "import os, uvicorn; uvicorn.run(\"app:app\", host=os.environ.get(\"API_HOST\", \"0.0.0.0\"), port=int(os.environ.get(\"API_PORT\", \"8088\")), timeout_graceful_shutdown=3, ws_ping_interval=None)"'

infer:
	@bash -c '$(CONDA_ACTIVATE) python apps/infer_worker.py'

capture:
	@bash -c '$(CONDA_ACTIVATE) python apps/capture_service.py'

devices:
	@v4l2-ctl --list-devices

kill:
	@echo "Killing processes on port $(API_PORT)…"
	@-fuser -k $(API_PORT)/tcp 2>/dev/null || true
	@-bash -c 'pids=$$(lsof -t -iTCP:$(API_PORT) -sTCP:LISTEN 2>/dev/null); if [ -n "$$pids" ]; then kill $$pids 2>/dev/null || kill -9 $$pids 2>/dev/null || true; fi'
	@echo "Done."

destroy:
	@echo "Stopping all Python processes…"
	@-pkill -f '[Pp]ython' 2>/dev/null || true
	@-killall -q python python3 2>/dev/null || true
	@echo "Done."

test:
	$(BIN)/pytest tests/ -v

lint:
	$(BIN)/ruff check .

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
