"""Pytest configuration and fixtures."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.factories import BudgetCarFactory, CarFeaturesFactory, LuxuryCarFactory

# Set testing environment before importing app
os.environ["ENVIRONMENT"] = "testing"


@pytest.fixture
def mock_predictor():
    """Mock predictor that returns fixed predictions without loading real model."""
    mock = MagicMock()
    mock.predict_from_features.return_value = [100]  # Fixed price
    return mock


@pytest.fixture
def mock_predictor_batch():
    """Mock predictor for batch predictions."""

    def side_effect(cars):
        return [100 + i * 10 for i in range(len(cars))]

    mock = MagicMock()
    mock.predict_from_features.side_effect = side_effect
    return mock


@pytest.fixture
def api_client_with_mock(mock_predictor) -> Generator[TestClient, None, None]:
    """FastAPI test client with mocked predictor (for unit tests)."""
    with patch("src.api.routers.predict.get_predictor", return_value=mock_predictor):
        from src.api.main import app

        with TestClient(app) as client:
            yield client


@pytest.fixture
def api_client_with_mock_batch(
    mock_predictor_batch,
) -> Generator[TestClient, None, None]:
    """FastAPI test client with batch-aware mocked predictor."""
    with patch(
        "src.api.routers.predict.get_predictor", return_value=mock_predictor_batch
    ):
        from src.api.main import app

        with TestClient(app) as client:
            yield client


@pytest.fixture(scope="session")
def api_client() -> Generator[TestClient, None, None]:
    """FastAPI test client with real model (for integration tests only)."""
    from src.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_car_features() -> dict:
    """Sample car features using factory."""
    return CarFeaturesFactory()


@pytest.fixture
def multiple_cars_features() -> list[dict]:
    """Multiple car features for batch testing."""
    return CarFeaturesFactory.create_batch(3)


@pytest.fixture
def luxury_car() -> dict:
    """Luxury car features."""
    return LuxuryCarFactory()


@pytest.fixture
def budget_car() -> dict:
    """Budget car features."""
    return BudgetCarFactory()
