.PHONY: install run test lint docker-build docker-up docker-down clean

# ── Local dev ─────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

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
	docker compose logs -f api

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
