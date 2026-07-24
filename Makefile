.PHONY: install backend-install frontend-install backend-test backend-lint backend-format backend-typecheck frontend-lint frontend-build dev-backend dev-frontend docker-build docker-up docker-down clean

# ─── Install ────────────────────────────────────────────────────────────────

install: backend-install frontend-install

backend-install:
	cd backend && uv sync --extra dev

frontend-install:
	cd frontend && npm ci

# ─── Backend ─────────────────────────────────────────────────────────────────

backend-test:
	cd backend && uv run pytest -v

backend-lint:
	cd backend && uv run ruff check src/ tests/

backend-format:
	cd backend && uv run ruff format src/ tests/

dev-backend:
	cd backend && uv run uvicorn optionforge.main:app --reload --host 0.0.0.0 --port 8000

# ─── Frontend ────────────────────────────────────────────────────────────────

frontend-lint:
	cd frontend && npm run lint

frontend-build:
	cd frontend && npm run build

dev-frontend:
	cd frontend && npm run dev

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	rm -rf backend/.venv backend/__pycache__ backend/src/**/__pycache__ backend/tests/__pycache__
	rm -rf frontend/node_modules frontend/dist
