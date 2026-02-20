# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev group = no mlflow)
RUN uv sync --no-group dev --frozen

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user (HF Spaces requirement - uid 1000)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH="/home/user/.local/bin:/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Copy virtual environment from builder
COPY --from=builder --chown=user /app/.venv /app/.venv

# Copy application code (only API-relevant modules)
COPY --chown=user src/__init__.py ./src/__init__.py
COPY --chown=user src/api/ ./src/api/
COPY --chown=user src/ml/__init__.py ./src/ml/__init__.py
COPY --chown=user src/ml/predict.py ./src/ml/predict.py
COPY --chown=user src/ml/preprocessing.py ./src/ml/preprocessing.py
COPY --chown=user src/config/ ./src/config/

# Copy trained model
COPY --chown=user models/ ./models/

# HF Spaces default port
EXPOSE 7860

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
