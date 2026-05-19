.PHONY: install dev run dashboard test test-fast lint format seed \
        docker-build docker-up docker-down docker-logs mock-webhook clean

# ── Local dev ─────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

## dev: run FastAPI with auto-reload (local development)
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

## run: alias for dev (kept for backward compatibility)
run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

## seed: load default team conventions into the vector store
seed:
	python scripts/seed_conventions.py

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

test-fast:
	pytest tests/ -v -x

lint:
	ruff check . && ruff format --check .

format:
	ruff format .

mock-webhook:
	python scripts/test_local.py

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f codereviewbot

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
