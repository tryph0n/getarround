# Deployment Guide

## Prerequisites

- Python 3.11+ with `uv` installed
- `huggingface-cli` available (via `pip install huggingface-hub` or `uvx huggingface-hub`)
- Authenticated with HF Hub (`huggingface-cli login`)
- For API deployment: a trained model at `models/best_model.joblib` (run `make train` first)

## Architecture

The project source of truth is the repository root. Deployment follows an
**assemble-on-deploy** strategy:

- `deploy/hf-api/` and `deploy/hf-dashboard/` contain only HF Spaces metadata
  (`Dockerfile`, `README.md`, `.dockerignore`).
- The `make deploy-*` targets assemble the required source files together with
  the metadata into a temporary `.tmp/` directory, then upload the result to
  HF Spaces via `huggingface-cli`.
- HF Spaces builds the Docker image from the uploaded `Dockerfile`.

No git clone of a HF Space repo is needed. The Makefile handles everything.

## Deploy Commands

```bash
make deploy-api          # Deploy API to HF Spaces
make deploy-dashboard    # Deploy dashboard to HF Spaces
make deploy-all          # Deploy both
```

## HF Spaces URLs

| Space | URL |
|-------|-----|
| API | https://sam-bot-get-around-api.hf.space |
| Dashboard | https://sam-bot-get-around-dashboard.hf.space |

## What Gets Deployed

### API

| Category | Files |
|----------|-------|
| Metadata | `deploy/hf-api/Dockerfile`, `deploy/hf-api/README.md` |
| Dependencies | `pyproject.toml`, `uv.lock` |
| Source | `src/api/`, `src/ml/` (predict + preprocessing only), `src/config/` |
| Model | `models/best_model.joblib` |

### Dashboard

| Category | Files |
|----------|-------|
| Metadata | `deploy/hf-dashboard/Dockerfile`, `deploy/hf-dashboard/README.md` |
| Dependencies | `pyproject.toml`, `uv.lock` |
| Source | `src/dashboard/`, `src/config/` |
| Data | `data/` |

## Important Notes

- **Deploy the API before the dashboard.** The dashboard calls the API at
  runtime, so the API must be up and healthy first.
- **Port 7860 is mandatory.** HF Spaces Docker runtime expects the application
  to listen on port 7860. Both Dockerfiles are configured accordingly (FastAPI
  via `--port 7860`, Streamlit via `STREAMLIT_SERVER_PORT=7860`).
- The root `pyproject.toml` includes all dependencies (API + dashboard). This
  produces heavier Docker images but keeps the workflow simple with a single
  lock file.
- Dockerfiles install production dependencies only with
  `uv sync --no-group dev --frozen` (excludes the dev group containing MLflow
  and test tools).
- HF Spaces rebuilds the Docker image on every upload. There is no incremental
  build cache between deploys.

## Environment Variables

- **Local development:** copy `.env.example` to `.env` and adjust values.
  Default ports are 8000 (API) and 8501 (dashboard).
- **HF Spaces production:** environment variables are baked into the
  Dockerfiles (`ENVIRONMENT=production`, `LOG_LEVEL=INFO`, `API_URL`, Streamlit
  settings). To override or add secrets (e.g., API keys), use the HF Space
  Settings > "Repository secrets" / "Variables" UI -- these are injected at
  runtime without rebuilding the image.

## Verification

After deploying, confirm the services are live:

```bash
curl https://sam-bot-get-around-api.hf.space/health
```

Then open the dashboard in a browser:

```
https://sam-bot-get-around-dashboard.hf.space
```
