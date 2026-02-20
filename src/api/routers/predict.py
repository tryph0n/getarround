"""Prediction router for pricing API."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas.prediction import PredictionInput, PredictionOutput
from src.ml.predict import ModelNotFoundError, get_predictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict rental prices",
    description="Predict optimal rental prices for cars based on their features.",
)
async def predict(data: PredictionInput) -> PredictionOutput:
    """Predict rental prices from car features.

    Args:
        data: Input containing list of car features.

    Returns:
        Predictions with list of predicted prices in EUR.

    Raises:
        HTTPException: If model is not available or prediction fails.
    """
    logger.info("Received prediction request for %d cars", len(data.cars))

    try:
        predictor = get_predictor()
        cars_dict = [car.model_dump() for car in data.cars]
        predictions = predictor.predict_from_features(cars_dict)
        logger.info("Predictions completed: %s", predictions)
        return PredictionOutput(prediction=predictions)
    except ModelNotFoundError as e:
        logger.error("Model not found: %s", e)
        raise HTTPException(status_code=503, detail="Model not available") from e
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error",
        ) from e
