"""Integration tests for API and ML pipeline."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.factories import BudgetCarFactory, LuxuryCarFactory

_model_path = Path(__file__).parent.parent.parent / "models" / "best_model.joblib"
_skip_no_model = pytest.mark.skipif(
    not _model_path.exists() or os.getenv("CI") == "true",
    reason="Model file not available (CI or missing locally)",
)


@_skip_no_model
@pytest.mark.integration
class TestPredictionFlow:
    """End-to-end tests for prediction flow."""

    def test_full_prediction_flow(
        self, api_client: TestClient, sample_car_features: dict
    ) -> None:
        """Test complete prediction flow from API request to response."""
        response = api_client.post(
            "/predict",
            json={"cars": [sample_car_features]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], list)
        assert len(data["prediction"]) == 1

        prediction = data["prediction"][0]
        assert isinstance(prediction, int)
        assert prediction > 0

    def test_api_and_ml_produce_same_results(
        self, api_client: TestClient, sample_car_features: dict
    ) -> None:
        """API predictions should match direct ML predictions."""
        from src.ml.predict import get_predictor

        response = api_client.post(
            "/predict",
            json={"cars": [sample_car_features]},
        )
        api_prediction = response.json()["prediction"][0]

        predictor = get_predictor()
        ml_prediction = predictor.predict_from_features([sample_car_features])[0]

        assert api_prediction == ml_prediction

    def test_predictions_are_deterministic(
        self, api_client: TestClient, sample_car_features: dict
    ) -> None:
        """Same input should produce same output."""
        predictions = []

        for _ in range(3):
            response = api_client.post(
                "/predict",
                json={"cars": [sample_car_features]},
            )
            predictions.append(response.json()["prediction"][0])

        assert all(p == predictions[0] for p in predictions)

    def test_batch_vs_individual_predictions(
        self, api_client: TestClient, multiple_cars_features: list[dict]
    ) -> None:
        """Batch predictions should match individual predictions."""
        batch_response = api_client.post(
            "/predict",
            json={"cars": multiple_cars_features},
        )
        batch_predictions = batch_response.json()["prediction"]

        individual_predictions = []
        for car in multiple_cars_features:
            response = api_client.post("/predict", json={"cars": [car]})
            individual_predictions.append(response.json()["prediction"][0])

        assert batch_predictions == individual_predictions


@_skip_no_model
@pytest.mark.integration
class TestAPIResilience:
    """Tests for API resilience and edge cases."""

    def test_different_car_types(self, api_client: TestClient) -> None:
        """API should handle various car configurations."""
        car_types = ["sedan", "hatchback", "suv", "van", "estate", "convertible"]

        for car_type in car_types:
            car = {
                "model_key": "Generic",
                "mileage": 50000,
                "engine_power": 100,
                "fuel": "diesel",
                "paint_color": "black",
                "car_type": car_type,
                "private_parking_available": False,
                "has_gps": False,
                "has_air_conditioning": True,
                "automatic_car": False,
                "has_getaround_connect": False,
                "has_speed_regulator": False,
                "winter_tires": False,
            }

            response = api_client.post("/predict", json={"cars": [car]})
            assert response.status_code == 200, f"Failed for car_type={car_type}"


@_skip_no_model
@pytest.mark.integration
class TestModelQuality:
    """Tests for model quality (integration tests with real model)."""

    def test_model_file_exists(self):
        """Model file should exist."""
        model_path = Path("models/best_model.joblib")
        assert model_path.exists(), f"Model not found at {model_path}"

    def test_predictor_can_be_instantiated(self):
        """PricingPredictor should be instantiable."""
        from src.ml.predict import PricingPredictor

        predictor = PricingPredictor()
        assert predictor.model is not None

    def test_get_predictor_returns_singleton(self):
        """get_predictor should return same instance."""
        from src.ml.predict import get_predictor

        predictor1 = get_predictor()
        predictor2 = get_predictor()
        assert predictor1 is predictor2

    def test_predictions_are_positive(self, sample_car_features: dict):
        """Predictions should be positive values."""
        from src.ml.predict import get_predictor

        predictor = get_predictor()
        predictions = predictor.predict_from_features([sample_car_features])

        for pred in predictions:
            assert pred > 0

    def test_predictions_in_reasonable_range(self, sample_car_features: dict):
        """Predictions should be in reasonable price range."""
        from src.ml.predict import get_predictor

        predictor = get_predictor()
        predictions = predictor.predict_from_features([sample_car_features])

        for pred in predictions:
            assert 10 < pred < 1000  # Reasonable daily rental price

    def test_luxury_car_higher_price(self):
        """Luxury car should have higher predicted price."""
        from src.ml.predict import get_predictor

        predictor = get_predictor()

        budget_car = BudgetCarFactory()
        luxury_car = LuxuryCarFactory()

        budget_price = predictor.predict_from_features([budget_car])[0]
        luxury_price = predictor.predict_from_features([luxury_car])[0]

        assert luxury_price > budget_price
