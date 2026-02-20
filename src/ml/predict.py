"""Model inference for pricing predictions."""

import logging
from pathlib import Path
from typing import Union

import joblib
import pandas as pd

from src.ml.preprocessing import normalize_categories

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "best_model.joblib"
)

_predictor_instance: "PricingPredictor | None" = None


class ModelNotFoundError(Exception):
    """Raised when the model file cannot be found."""


class PricingPredictor:
    """Predictor class for car rental pricing."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
        """Initialize predictor with trained model.

        Args:
            model_path: Path to the trained model file.

        Raises:
            ModelNotFoundError: If the model file does not exist.
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model from disk.

        Raises:
            ModelNotFoundError: If the model file does not exist.
        """
        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            raise ModelNotFoundError(f"Model not found at {self.model_path}")

        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully")

    def predict_from_features(self, cars: list[dict]) -> list[int]:
        """Make predictions from car feature dictionaries.

        Args:
            cars: List of dictionaries with car features.

        Returns:
            List of predicted prices (rounded to int).
        """
        logger.debug("Predicting for %d cars", len(cars))
        df = pd.DataFrame(cars)
        df = normalize_categories(df)
        predictions = self.model.predict(df)
        return [int(round(p)) for p in predictions]


def get_predictor(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> PricingPredictor:
    """Get singleton predictor instance.

    Args:
        model_path: Path to the trained model file.

    Returns:
        Singleton PricingPredictor instance.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PricingPredictor(model_path)
    return _predictor_instance
