"""Model training with MLflow tracking."""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config.settings import get_settings
from src.ml.preprocessing import (
    clean_data,
    create_preprocessor,
    load_data,
    prepare_features,
)

logger = logging.getLogger(__name__)

MODELS = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__max_features": [1.0, "sqrt"],
    },
    "gradient_boosting": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
}

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    model,
    preprocessor,
    param_distributions=None,
    cv=5,
    n_iter=20,
) -> dict:
    """Train model and return metrics.

    Train metrics are computed alongside test metrics to enable overfitting
    detection: a large gap between train R2 and test R2 indicates the model
    memorizes training data rather than generalizing.

    Hyperparameter search via RandomizedSearchCV explores the param space
    efficiently. When enabled, the returned pipeline uses the best
    hyperparameters found during cross-validation.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        model_name: Name of the model for logging.
        model: Sklearn estimator instance.
        preprocessor: Sklearn preprocessor (ColumnTransformer).
        param_distributions: Dict mapping parameter names to distributions or
            lists of values for RandomizedSearchCV. If None, fits the pipeline
            directly without hyperparameter search.
        cv: Number of cross-validation folds (used when param_distributions
            is provided).
        n_iter: Number of parameter settings sampled by RandomizedSearchCV
            (used when param_distributions is provided).

    Returns:
        Dictionary with model name, pipeline, test metrics (rmse, mae, r2),
        train metrics (train_rmse, train_mae, train_r2), best_params (dict of
        best hyperparameters found by RSCV, or None), cv_best_score (best
        CV RMSE from RSCV, or None), and residual statistics (residual_mean,
        residual_std, residual_median, residual_skew). Residual statistics
        diagnose prediction error patterns: mean near 0 confirms no systematic
        bias, std quantifies error spread, skewness reveals if the model
        systematically under-predicts for certain price ranges.
    """
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    logger.info("Training %s...", model_name)
    if param_distributions is not None:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
        logger.info(
            "Running RandomizedSearchCV for %s (n_iter=%d, cv=%d)...",
            model_name,
            n_iter,
            cv,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = dict(search.best_params_)
        cv_best_score = float(-search.best_score_)
        logger.info("%s - Best CV RMSE: %.2f", model_name, cv_best_score)
    else:
        pipeline.fit(X_train, y_train)
        best_params = None
        cv_best_score = None

    y_train_pred = pipeline.predict(X_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    y_pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    logger.info("%s - Test RMSE: %.2f, MAE: %.2f, R2: %.3f", model_name, rmse, mae, r2)
    logger.info(
        "%s - Train RMSE: %.2f, MAE: %.2f, R2: %.3f",
        model_name,
        train_rmse,
        train_mae,
        train_r2,
    )

    # Residual analysis: diagnoses systematic bias and error asymmetry
    # that symmetric metrics (RMSE) cannot reveal.
    residuals = y_test - y_pred
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals))
    residual_median = float(np.median(residuals))
    residual_skew = float(pd.Series(residuals).skew())

    r2_delta = train_r2 - r2
    if r2_delta > 0.10:
        logger.warning(
            "%s - Possible overfitting: train R2=%.3f vs test R2=%.3f (delta=%.3f)",
            model_name,
            train_r2,
            r2,
            r2_delta,
        )

    if abs(residual_mean) > 1.0:
        logger.warning(
            "%s - Possible systematic bias: residual mean=%.2f EUR",
            model_name,
            residual_mean,
        )

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "best_params": best_params,
        "cv_best_score": cv_best_score,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_median": residual_median,
        "residual_skew": residual_skew,
    }


def run_experiment(data_path: str, experiment_name: str = "getaround_pricing") -> str:
    """Run full training experiment, return best model path.

    Args:
        data_path: Path to the CSV data file.
        experiment_name: MLflow experiment name.

    Returns:
        Path to the saved best model.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)

    logger.info("Cleaning data...")
    df = clean_data(df)

    logger.info("Preparing features...")
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        "Train/test split: %d train samples, %d test samples",
        len(X_train),
        len(X_test),
    )

    preprocessor = create_preprocessor()
    results = []

    for model_name, model in MODELS.items():
        with mlflow.start_run(run_name=model_name):
            result = train_and_evaluate(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name,
                model,
                preprocessor,
                param_distributions=PARAM_DISTRIBUTIONS.get(model_name),
            )
            results.append(result)

            mlflow.log_param("model_name", model_name)
            if result["best_params"] is not None:
                mlflow.log_params(result["best_params"])
            else:
                mlflow.log_params(model.get_params())
            mlflow.log_metric("test_rmse", result["rmse"])
            mlflow.log_metric("test_mae", result["mae"])
            mlflow.log_metric("test_r2", result["r2"])
            mlflow.log_metric("train_rmse", result["train_rmse"])
            mlflow.log_metric("train_mae", result["train_mae"])
            mlflow.log_metric("train_r2", result["train_r2"])
            if result["cv_best_score"] is not None:
                mlflow.log_metric("cv_best_rmse", result["cv_best_score"])
            mlflow.log_metric("residual_mean", result["residual_mean"])
            mlflow.log_metric("residual_std", result["residual_std"])
            mlflow.log_metric("residual_median", result["residual_median"])
            mlflow.log_metric("residual_skew", result["residual_skew"])

            mlflow.sklearn.log_model(result["pipeline"], "model")

            # Extract feature importances for interpretability
            fitted_model = result["pipeline"].named_steps["model"]
            feature_names = (
                result["pipeline"].named_steps["preprocessor"].get_feature_names_out()
            )
            if hasattr(fitted_model, "feature_importances_"):
                importances = fitted_model.feature_importances_
            elif hasattr(fitted_model, "coef_"):
                importances = np.abs(fitted_model.coef_)
            else:
                importances = None

            if importances is not None:
                fi_dict = dict(
                    zip(
                        feature_names.tolist(),
                        importances.tolist(),
                    )
                )
                fi_sorted = dict(
                    sorted(
                        fi_dict.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )
                )
                mlflow.log_dict(fi_sorted, "feature_importances.json")

    best_result = min(results, key=lambda x: x["rmse"])
    logger.info(
        "Best model: %s with RMSE=%.2f",
        best_result["model_name"],
        best_result["rmse"],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_result["pipeline"], model_path)
    logger.info("Best model saved to %s", model_path)

    return str(model_path)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    data_path = (
        sys.argv[1] if len(sys.argv) > 1 else "data/get_around_pricing_project.csv"
    )
    run_experiment(data_path)
