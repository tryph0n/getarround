"""Unit tests for API endpoints."""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, api_client_with_mock: TestClient) -> None:
        """Health endpoint should return 200 OK."""
        response = api_client_with_mock.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(
        self, api_client_with_mock: TestClient
    ) -> None:
        """Health endpoint should return healthy status."""
        response = api_client_with_mock.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_single_car(
        self, api_client_with_mock: TestClient, sample_car_features: dict
    ) -> None:
        """Predict endpoint should handle single car."""
        response = api_client_with_mock.post(
            "/predict",
            json={"cars": [sample_car_features]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert len(data["prediction"]) == 1

    def test_predict_multiple_cars(
        self,
        api_client_with_mock_batch: TestClient,
        multiple_cars_features: list[dict],
    ) -> None:
        """Predict endpoint should handle multiple cars."""
        response = api_client_with_mock_batch.post(
            "/predict",
            json={"cars": multiple_cars_features},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert len(data["prediction"]) == len(multiple_cars_features)

    def test_predict_returns_integers(
        self, api_client_with_mock: TestClient, sample_car_features: dict
    ) -> None:
        """Predictions should be integers."""
        response = api_client_with_mock.post(
            "/predict",
            json={"cars": [sample_car_features]},
        )
        data = response.json()
        for pred in data["prediction"]:
            assert isinstance(pred, int)

    def test_predict_returns_expected_mock_value(
        self, api_client_with_mock: TestClient, sample_car_features: dict
    ) -> None:
        """Mock predictor should return fixed value of 100."""
        response = api_client_with_mock.post(
            "/predict",
            json={"cars": [sample_car_features]},
        )
        data = response.json()
        assert data["prediction"] == [100]


class TestPredictValidation:
    """Tests for input validation on /predict endpoint."""

    def test_empty_cars_list_fails(self, api_client_with_mock: TestClient) -> None:
        """Empty cars list should fail validation."""
        response = api_client_with_mock.post("/predict", json={"cars": []})
        assert response.status_code == 422

    def test_missing_required_field_fails(
        self, api_client_with_mock: TestClient
    ) -> None:
        """Missing required field should fail validation."""
        incomplete_car = {"model_key": "BMW"}  # Missing other required fields
        response = api_client_with_mock.post(
            "/predict", json={"cars": [incomplete_car]}
        )
        assert response.status_code == 422

    def test_invalid_mileage_type_fails(
        self, api_client_with_mock: TestClient, sample_car_features: dict
    ) -> None:
        """Invalid mileage type should fail validation."""
        invalid_car = sample_car_features.copy()
        invalid_car["mileage"] = "not_a_number"
        response = api_client_with_mock.post("/predict", json={"cars": [invalid_car]})
        assert response.status_code == 422
