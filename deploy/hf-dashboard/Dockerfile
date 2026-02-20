# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev group)
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
    LOG_LEVEL=INFO \
    API_URL="https://sam-bot-get-around-api.hf.space" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy virtual environment from builder
COPY --from=builder --chown=user /app/.venv /app/.venv

# Copy application code
COPY --chown=user src/__init__.py ./src/__init__.py
COPY --chown=user src/dashboard/ ./src/dashboard/
COPY --chown=user src/config/ ./src/config/

# Copy data files
COPY --chown=user data/ ./data/

# HF Spaces default port
EXPOSE 7860

# Run the dashboard
CMD ["streamlit", "run", "src/dashboard/app.py"]
