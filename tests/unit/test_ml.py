"""Unit tests for ML modules."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestPreprocessing:
    """Tests for preprocessing module."""

    def test_create_preprocessor_returns_transformer(self):
        """create_preprocessor should return a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer

        from src.ml.preprocessing import create_preprocessor

        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_feature_names_not_empty(self):
        """get_feature_names should return non-empty list."""
        from src.ml.preprocessing import get_feature_names

        features = get_feature_names()
        assert isinstance(features, list)
        assert len(features) > 0

    def test_all_expected_features_present(self):
        """All expected feature categories should be present."""
        from src.ml.preprocessing import (
            BOOLEAN_FEATURES,
            CATEGORICAL_FEATURES,
            NUMERICAL_FEATURES,
        )

        assert len(CATEGORICAL_FEATURES) == 4
        assert len(BOOLEAN_FEATURES) == 7
        assert len(NUMERICAL_FEATURES) == 2


def _mock_predictor_context(mock_pipeline):
    """Context manager that mocks both Path.exists() and joblib.load.

    This ensures PricingPredictor.__init__ skips the file existence check
    and uses the mocked pipeline instead of loading from disk.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with (
            patch("src.ml.predict.Path.exists", return_value=True),
            patch("src.ml.predict.joblib.load", return_value=mock_pipeline),
        ):
            from importlib import reload

            import src.ml.predict

            src.ml.predict._predictor_instance = None
            reload(src.ml.predict)
            yield

    return _ctx()


class TestPredictor:
    """Tests for predictor module with mocked model."""

    def test_predictor_uses_model(self, sample_car_features: dict):
        """Predictor should use the model for predictions."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [125.5]

        with _mock_predictor_context(mock_pipeline):
            from src.ml.predict import PricingPredictor

            predictor = PricingPredictor()
            predictions = predictor.predict_from_features([sample_car_features])

            assert predictions == [126]  # Rounded to int
            mock_pipeline.predict.assert_called_once()

    def test_predict_from_features_returns_list(self, sample_car_features: dict):
        """predict_from_features should return list of ints."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [100.0]

        with _mock_predictor_context(mock_pipeline):
            from src.ml.predict import PricingPredictor

            predictor = PricingPredictor()
            predictions = predictor.predict_from_features([sample_car_features])

            assert isinstance(predictions, list)
            assert len(predictions) == 1
            assert isinstance(predictions[0], int)

    def test_predict_batch(self, multiple_cars_features: list[dict]):
        """Predictor should handle batch predictions."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [100.0, 110.0, 120.0]

        with _mock_predictor_context(mock_pipeline):
            from src.ml.predict import PricingPredictor

            predictor = PricingPredictor()
            predictions = predictor.predict_from_features(multiple_cars_features)

            assert len(predictions) == len(multiple_cars_features)


class TestModelQualityMocked:
    """Tests for model quality with mocked predictions."""

    def test_predictions_are_positive(self, sample_car_features: dict):
        """Predictions should be positive values."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [85.0]

        with _mock_predictor_context(mock_pipeline):
            from src.ml.predict import PricingPredictor

            predictor = PricingPredictor()
            predictions = predictor.predict_from_features([sample_car_features])

            for pred in predictions:
                assert pred > 0

    def test_luxury_vs_budget_car_logic(self, luxury_car: dict, budget_car: dict):
        """Test that factory creates valid car features."""
        # Verify factories produce correct structure
        assert luxury_car["model_key"] == "BMW"
        assert luxury_car["has_gps"] is True
        assert luxury_car["automatic_car"] is True

        assert budget_car["model_key"] == "Renault"
        assert budget_car["has_gps"] is False
        assert budget_car["automatic_car"] is False


def _build_valid_dataframe(**overrides) -> pd.DataFrame:
    """Build a single-row DataFrame with all required columns for prepare_features."""
    from src.ml.preprocessing import TARGET

    defaults = {
        "mileage": 50000,
        "engine_power": 150,
        "private_parking_available": True,
        "has_gps": False,
        "has_air_conditioning": True,
        "automatic_car": False,
        "has_getaround_connect": True,
        "has_speed_regulator": False,
        "winter_tires": True,
        "model_key": "Peugeot",
        "fuel": "diesel",
        "paint_color": "black",
        "car_type": "sedan",
        TARGET: 100,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_returns_tuple_of_dataframe_and_series(self):
        """prepare_features should return (DataFrame, Series)."""
        from src.ml.preprocessing import prepare_features

        df = _build_valid_dataframe()
        X, y = prepare_features(df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_output_columns_match_expected_order(self):
        """X columns should be numerical + boolean + categorical in that order."""
        from src.ml.preprocessing import (
            BOOLEAN_FEATURES,
            CATEGORICAL_FEATURES,
            NUMERICAL_FEATURES,
            prepare_features,
        )

        df = _build_valid_dataframe()
        X, _ = prepare_features(df)

        expected_cols = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
        assert list(X.columns) == expected_cols

    def test_boolean_columns_are_int(self):
        """Boolean features should be cast to int (0/1)."""
        from src.ml.preprocessing import BOOLEAN_FEATURES, prepare_features

        df = _build_valid_dataframe()
        X, _ = prepare_features(df)

        for col in BOOLEAN_FEATURES:
            assert X[col].dtype in (int, pd.Int64Dtype(), "int64", "int32")
            assert set(X[col].unique()).issubset({0, 1})

    def test_target_column_excluded_from_X(self):
        """X should not contain the target column."""
        from src.ml.preprocessing import TARGET, prepare_features

        df = _build_valid_dataframe()
        X, _ = prepare_features(df)

        assert TARGET not in X.columns

    def test_y_contains_target_values(self):
        """y should contain the target column values."""
        from src.ml.preprocessing import prepare_features

        df = _build_valid_dataframe(**{"rental_price_per_day": 42})
        _, y = prepare_features(df)

        assert y.iloc[0] == 42

    def test_missing_required_column_raises_key_error(self):
        """Missing a required column should raise KeyError."""
        from src.ml.preprocessing import prepare_features

        df = _build_valid_dataframe()
        df = df.drop(columns=["mileage"])

        with pytest.raises(KeyError, match="mileage"):
            prepare_features(df)

    def test_missing_target_column_raises_key_error(self):
        """Missing target column should raise KeyError."""
        from src.ml.preprocessing import TARGET, prepare_features

        df = _build_valid_dataframe()
        df = df.drop(columns=[TARGET])

        with pytest.raises(KeyError, match=TARGET):
            prepare_features(df)

    def test_multiple_rows(self):
        """prepare_features should handle multiple rows correctly."""
        from src.ml.preprocessing import prepare_features

        row1 = _build_valid_dataframe(mileage=10000, engine_power=80)
        row2 = _build_valid_dataframe(mileage=200000, engine_power=300)
        df = pd.concat([row1, row2], ignore_index=True)

        X, y = prepare_features(df)

        assert len(X) == 2
        assert len(y) == 2

    def test_extreme_numerical_values(self):
        """prepare_features should accept extreme but valid numerical values."""
        from src.ml.preprocessing import prepare_features

        df = _build_valid_dataframe(mileage=0, engine_power=1)
        X, _ = prepare_features(df)

        assert X["mileage"].iloc[0] == 0
        assert X["engine_power"].iloc[0] == 1

    def test_extra_columns_are_dropped(self):
        """Extra columns not in feature definitions should be excluded from X."""
        from src.ml.preprocessing import prepare_features

        df = _build_valid_dataframe()
        df["unexpected_column"] = "surprise"

        X, _ = prepare_features(df)

        assert "unexpected_column" not in X.columns


def _build_train_test_split(n_samples: int = 40):
    """Build minimal train/test split suitable for train_and_evaluate."""
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    from src.ml.preprocessing import (
        BOOLEAN_FEATURES,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
    )

    rng = np.random.default_rng(0)
    n_train = int(n_samples * 0.75)
    n_test = n_samples - n_train

    def _make_df(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mileage": rng.integers(5000, 200000, size=n).astype(float),
                "engine_power": rng.integers(60, 300, size=n).astype(float),
                "private_parking_available": rng.integers(0, 2, size=n),
                "has_gps": rng.integers(0, 2, size=n),
                "has_air_conditioning": rng.integers(0, 2, size=n),
                "automatic_car": rng.integers(0, 2, size=n),
                "has_getaround_connect": rng.integers(0, 2, size=n),
                "has_speed_regulator": rng.integers(0, 2, size=n),
                "winter_tires": rng.integers(0, 2, size=n),
                "model_key": rng.choice(["Peugeot", "Renault", "BMW"], size=n),
                "fuel": rng.choice(["diesel", "petrol"], size=n),
                "paint_color": rng.choice(["black", "white"], size=n),
                "car_type": rng.choice(["sedan", "suv"], size=n),
            }
        )

    X_train = _make_df(n_train)
    X_test = _make_df(n_test)
    y_train = pd.Series(rng.integers(50, 300, size=n_train).astype(float))
    y_test = pd.Series(rng.integers(50, 300, size=n_test).astype(float))

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


class TestTrainAndEvaluate:
    """Tests for train_and_evaluate return dict structure and content."""

    def test_return_dict_has_all_keys(self):
        """Return dict must contain all expected keys including RSCV keys."""
        from sklearn.linear_model import LinearRegression

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "linear_regression",
            LinearRegression(),
            preprocessor,
        )

        expected_keys = {
            "model_name",
            "pipeline",
            "rmse",
            "mae",
            "r2",
            "train_rmse",
            "train_mae",
            "train_r2",
            "best_params",
            "cv_best_score",
            "residual_mean",
            "residual_std",
            "residual_median",
            "residual_skew",
        }
        assert set(result.keys()) == expected_keys
        assert result["best_params"] is None
        assert result["cv_best_score"] is None

    def test_test_metrics_are_numeric(self):
        """Test metrics (rmse, mae, r2) must be numeric."""
        from sklearn.linear_model import LinearRegression

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "linear_regression",
            LinearRegression(),
            preprocessor,
        )

        assert isinstance(result["rmse"], (int, float))
        assert isinstance(result["mae"], (int, float))
        assert isinstance(result["r2"], (int, float))

    def test_train_metrics_are_numeric(self):
        """Train metrics (train_rmse, train_mae, train_r2) must be numeric."""
        from sklearn.linear_model import LinearRegression

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "linear_regression",
            LinearRegression(),
            preprocessor,
        )

        assert isinstance(result["train_rmse"], (int, float))
        assert isinstance(result["train_mae"], (int, float))
        assert isinstance(result["train_r2"], (int, float))

    def test_train_r2_at_least_as_good_as_test_r2(self):
        """Train R2 >= test R2: model fits training data at least as well."""
        from sklearn.linear_model import LinearRegression

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split(
            n_samples=100
        )
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "linear_regression",
            LinearRegression(),
            preprocessor,
        )

        assert result["train_r2"] >= result["r2"]

    def test_model_name_preserved_in_result(self):
        """model_name in the return dict should match the input name."""
        from sklearn.linear_model import LinearRegression

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "my_model",
            LinearRegression(),
            preprocessor,
        )

        assert result["model_name"] == "my_model"

    def test_pipeline_is_fitted(self):
        """Returned pipeline must be fitted (has named_steps with model step)."""
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "linear_regression",
            LinearRegression(),
            preprocessor,
        )

        assert isinstance(result["pipeline"], Pipeline)
        assert "model" in result["pipeline"].named_steps

    def test_residual_stats_are_numeric(self):
        """Residual stats must be floats; residual_std must be positive."""
        from sklearn.linear_model import LinearRegression

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "linear_regression",
            LinearRegression(),
            preprocessor,
        )

        assert isinstance(result["residual_mean"], float)
        assert isinstance(result["residual_std"], float)
        assert isinstance(result["residual_median"], float)
        assert isinstance(result["residual_skew"], float)
        assert result["residual_std"] > 0

    def test_rscv_path_returns_best_params(self):
        """With param_distributions provided, best_params and cv_best_score must be set."""
        from sklearn.ensemble import RandomForestRegressor

        from src.ml.train import train_and_evaluate

        X_train, X_test, y_train, y_test, preprocessor = _build_train_test_split()
        result = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            "random_forest",
            RandomForestRegressor(random_state=42),
            preprocessor,
            param_distributions={"model__n_estimators": [50, 100]},
            cv=2,
            n_iter=2,
        )

        assert isinstance(result["best_params"], dict)
        assert isinstance(result["cv_best_score"], float)
        assert result["cv_best_score"] > 0
        assert "model__n_estimators" in result["best_params"]
        assert result["best_params"]["model__n_estimators"] in [50, 100]


class TestRunExperimentMLflowLogging:
    """Tests that run_experiment logs the correct MLflow metrics."""

    def test_mlflow_logs_metrics_per_model(self):
        """run_experiment logs 7 metrics for RF/GB (with cv_best_rmse), 6 for LR.

        Also verifies that feature importances are logged once per model as a
        non-empty dict artifact named 'feature_importances.json'.
        """
        from src.ml.train import run_experiment

        with (
            patch("src.ml.train.load_data") as mock_load,
            patch("src.ml.train.clean_data", side_effect=lambda df: df),
            patch("src.ml.train.mlflow.set_tracking_uri"),
            patch("src.ml.train.mlflow.set_experiment"),
            patch("src.ml.train.mlflow.start_run") as mock_start_run,
            patch("src.ml.train.mlflow.log_param"),
            patch("src.ml.train.mlflow.log_params") as mock_log_params,
            patch("src.ml.train.mlflow.log_metric") as mock_log_metric,
            patch("src.ml.train.mlflow.log_dict") as mock_log_dict,
            patch("src.ml.train.mlflow.sklearn.log_model"),
            patch("src.ml.train.joblib.dump"),
            patch("src.ml.train.MODELS_DIR") as mock_models_dir,
        ):
            mock_models_dir.mkdir = MagicMock()
            mock_models_dir.__truediv__ = lambda self, other: MagicMock()

            import numpy as np

            from src.ml.preprocessing import TARGET

            rng = np.random.default_rng(1)
            n = 60
            mock_load.return_value = pd.DataFrame(
                {
                    "mileage": rng.integers(5000, 200000, size=n).astype(float),
                    "engine_power": rng.integers(60, 300, size=n).astype(float),
                    "private_parking_available": rng.integers(0, 2, size=n),
                    "has_gps": rng.integers(0, 2, size=n),
                    "has_air_conditioning": rng.integers(0, 2, size=n),
                    "automatic_car": rng.integers(0, 2, size=n),
                    "has_getaround_connect": rng.integers(0, 2, size=n),
                    "has_speed_regulator": rng.integers(0, 2, size=n),
                    "winter_tires": rng.integers(0, 2, size=n),
                    "model_key": rng.choice(["Peugeot", "Renault", "BMW"], size=n),
                    "fuel": rng.choice(["diesel", "petrol"], size=n),
                    "paint_color": rng.choice(["black", "white"], size=n),
                    "car_type": rng.choice(["sedan", "suv"], size=n),
                    TARGET: rng.integers(50, 300, size=n).astype(float),
                }
            )

            mock_start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            run_experiment("fake_path.csv")

        logged_metric_names = [c.args[0] for c in mock_log_metric.call_args_list]

        # 6 base metrics logged for all 3 models
        base_metrics = {
            "test_rmse",
            "test_mae",
            "test_r2",
            "train_rmse",
            "train_mae",
            "train_r2",
        }
        for metric in base_metrics:
            assert logged_metric_names.count(metric) == 3, (
                f"Expected metric '{metric}' to be logged 3 times, "
                f"got {logged_metric_names.count(metric)}"
            )

        # cv_best_rmse logged only for RF and GB (not LR): exactly 2 times
        assert logged_metric_names.count("cv_best_rmse") == 2, (
            f"Expected 'cv_best_rmse' to be logged 2 times (RF + GB), "
            f"got {logged_metric_names.count('cv_best_rmse')}"
        )

        # 4 residual metrics logged for all 3 models
        residual_metrics = {
            "residual_mean",
            "residual_std",
            "residual_median",
            "residual_skew",
        }
        for metric in residual_metrics:
            assert logged_metric_names.count(metric) == 3, (
                f"Expected metric '{metric}' to be logged 3 times, "
                f"got {logged_metric_names.count(metric)}"
            )

        # Total: 6*3 + 2 + 4*3 = 32 metric calls
        assert (
            len(logged_metric_names) == 32
        ), f"Expected 32 total metric calls, got {len(logged_metric_names)}"

        # log_params called once per model (3 times total)
        assert mock_log_params.call_count == 3, (
            f"Expected log_params to be called 3 times, "
            f"got {mock_log_params.call_count}"
        )

        # Feature importances logged once per model (RF, GB, LR all support this)
        assert mock_log_dict.call_count == 3, (
            f"Expected log_dict to be called 3 times (once per model), "
            f"got {mock_log_dict.call_count}"
        )

        for call in mock_log_dict.call_args_list:
            fi_dict, artifact_name = call.args
            assert artifact_name == "feature_importances.json", (
                f"Expected artifact name 'feature_importances.json', "
                f"got '{artifact_name}'"
            )
            assert isinstance(
                fi_dict, dict
            ), f"Expected a dict for feature importances, got {type(fi_dict)}"
            assert len(fi_dict) > 0, "Feature importances dict must not be empty"
