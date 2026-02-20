-include .env

API_HOST ?= 0.0.0.0
API_PORT ?= 8000
DASHBOARD_PORT ?= 8501

.PHONY: install-prod install-dev lint format test train run-api run-dashboard run-mlflow clean help check-model deploy-api deploy-dashboard deploy-all

help:
	@echo "Available commands:"
	@echo "  make install-prod  - Install production dependencies"
	@echo "  make install-dev   - Install all dependencies (including dev)"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Format code with ruff"
	@echo "  make test          - Run tests"
	@echo "  make train         - Train ML models with MLflow tracking"
	@echo "  make run-api       - Start FastAPI server"
	@echo "  make run-dashboard - Start Streamlit dashboard"
	@echo "  make run-mlflow    - Start MLflow UI"
	@echo "  make deploy-api       - Build staging tree and upload API to HF Spaces"
	@echo "  make deploy-dashboard - Build staging tree and upload dashboard to HF Spaces"
	@echo "  make deploy-all       - Deploy both API and dashboard to HF Spaces"
	@echo "  make clean         - Remove cache files"

install-prod:
	uv sync --no-group dev

install-dev:
	uv sync

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run pytest

train: ## Train ML models with MLflow tracking
	uv run python -m src.ml.train

run-api:
	PYTHONPATH=. uv run uvicorn src.api.main:app --reload --host $(API_HOST) --port $(API_PORT)

run-dashboard:
	PYTHONPATH=. uv run streamlit run src/dashboard/app.py --server.port $(DASHBOARD_PORT)

run-mlflow: ## Start MLflow UI
	uv run mlflow ui --backend-store-uri ./mlruns --port 5000

check-model:
	@test -f models/best_model.joblib || (echo "ERROR: models/best_model.joblib not found. Run 'make train' first." && exit 1)

deploy-api: check-model
	rm -rf .tmp/hf-api
	mkdir -p .tmp/hf-api/src/ml .tmp/hf-api/models
	cp deploy/hf-api/Dockerfile deploy/hf-api/README.md .tmp/hf-api/
	cp pyproject.toml uv.lock .tmp/hf-api/
	cp src/__init__.py .tmp/hf-api/src/
	cp -r src/api .tmp/hf-api/src/
	cp src/ml/__init__.py src/ml/predict.py src/ml/preprocessing.py .tmp/hf-api/src/ml/
	cp -r src/config .tmp/hf-api/src/
	cp models/best_model.joblib .tmp/hf-api/models/
	find .tmp/hf-api -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	huggingface-cli upload sam-bot/get-around-api .tmp/hf-api . --repo-type space --commit-message "Deploy API"

deploy-dashboard:
	rm -rf .tmp/hf-dashboard
	mkdir -p .tmp/hf-dashboard/src
	cp deploy/hf-dashboard/Dockerfile deploy/hf-dashboard/README.md .tmp/hf-dashboard/
	cp pyproject.toml uv.lock .tmp/hf-dashboard/
	cp src/__init__.py .tmp/hf-dashboard/src/
	cp -r src/dashboard .tmp/hf-dashboard/src/
	cp -r src/config .tmp/hf-dashboard/src/
	cp -r data .tmp/hf-dashboard/
	find .tmp/hf-dashboard -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	huggingface-cli upload sam-bot/get-around-dashboard .tmp/hf-dashboard . --repo-type space --commit-message "Deploy dashboard"

deploy-all: deploy-api deploy-dashboard

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
