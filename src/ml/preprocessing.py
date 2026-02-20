"""Feature engineering and preprocessing for pricing model."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# Feature definitions based on EDA
CATEGORICAL_FEATURES = [
    "model_key",
    "fuel",
    "paint_color",
    "car_type",
]

BOOLEAN_FEATURES = [
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]

NUMERICAL_FEATURES = [
    "mileage",
    "engine_power",
]

TARGET = "rental_price_per_day"

BRAND_NORMALIZATION: dict[str, str] = {
    "Citroën": "Citroen",
    "KIA Motors": "KIA",
}


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Align training brand names with the canonical forms used by the API and tests.

    Training data contains variants like "Citroën" or "KIA Motors" that differ from
    canonical forms; without this mapping, OneHotEncoder(handle_unknown="ignore")
    silently zeros out those features at inference time, degrading predictions.
    """
    if "model_key" in df.columns:
        df = df.copy()
        df["model_key"] = df["model_key"].replace(BRAND_NORMALIZATION)
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """Load pricing dataset from CSV.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If file does not exist.
        pd.errors.ParserError: If CSV parsing fails.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath, index_col=0)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def create_preprocessor() -> ColumnTransformer:
    """Create sklearn preprocessor for features.

    The preprocessor applies:
    - StandardScaler to numerical features (mileage, engine_power)
    - Passthrough for boolean features (already 0/1)
    - OneHotEncoder for categorical features (model_key, fuel, paint_color, car_type)

    Returns:
        ColumnTransformer configured for all feature types.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "bool",
                "passthrough",
                BOOLEAN_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    logger.debug(
        "Created preprocessor with %d transformers", len(preprocessor.transformers)
    )
    return preprocessor


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with physically impossible values identified during EDA.

    Data entry errors (not missing values) are removed rather than imputed:
    the true values are unknowable, and imputing would fabricate data points.
    Affects 2 rows out of ~4843 (0.04%) with zero measurable impact on model
    performance. Training-only: at inference, the Pydantic schema rejects
    invalid inputs at the API boundary.

    Rules:
        - engine_power must be > 0 (0 = physically impossible for a vehicle)
        - mileage must be >= 0 (negative distance is impossible)
    """
    initial_len = len(df)

    df = df[df["engine_power"] > 0]
    df = df[df["mileage"] >= 0]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(
            "Removed %d rows with impossible values (engine_power=0 or mileage<0)",
            removed,
        )

    extreme_mileage = df["mileage"] > 500_000
    if extreme_mileage.any():
        logger.warning("Found %d rows with mileage > 500,000 km", extreme_mileage.sum())

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and target y.

    Converts boolean columns to int (0/1) for sklearn compatibility.

    Args:
        df: DataFrame with all columns including target.

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.

    Raises:
        KeyError: If required columns are missing.
    """
    df = normalize_categories(df)

    required_cols = (
        NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    )
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    feature_cols = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()

    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)

    y = df[TARGET].copy()

    logger.info("Prepared features: X shape %s, y shape %s", X.shape, y.shape)
    return X, y


def get_feature_names() -> list[str]:
    """Return list of all feature names used.

    Returns:
        List of feature names in order: numerical, boolean, categorical.
    """
    return NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
