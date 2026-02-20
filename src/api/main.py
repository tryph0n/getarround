"""FastAPI application for Getaround pricing API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers.predict import router as predict_router
from src.config.settings import configure_logging, get_settings
from src.ml.predict import get_predictor

settings = get_settings()
configure_logging(settings)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Args:
        app: The FastAPI application instance.
    """
    # Startup: preload model
    logger.info("Starting up - preloading model")
    try:
        get_predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning("Model not available at startup: %s", e)
    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="Getaround Pricing API",
    description="""
API for predicting optimal rental prices for cars.

## Endpoints

- **POST /predict**: Predict rental prices based on car features
- **GET /health**: Health check endpoint

## Usage

Send a POST request to `/predict` with car features:

```json
{
    "cars": [{
        "model_key": "Citroen",
        "mileage": 100000,
        "engine_power": 120,
        "fuel": "diesel",
        "paint_color": "black",
        "car_type": "sedan",
        "private_parking_available": true,
        "has_gps": true,
        "has_air_conditioning": true,
        "automatic_car": false,
        "has_getaround_connect": false,
        "has_speed_regulator": true,
        "winter_tires": false
    }]
}
```

Response:
```json
{
    "prediction": [124]
}
```
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Status dictionary with model availability.
    """
    from src.ml.predict import _predictor_instance

    model_loaded = _predictor_instance is not None
    return {"status": "healthy", "model_loaded": model_loaded}
