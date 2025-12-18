"""
Data preprocessing script for the Water Potability dataset.

Steps:
- Load dataset
- Display class distribution
- Handle missing values using median imputation
- Cap outliers using the IQR method
- Split data into train and test sets
- Save processed artifacts to disk
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def remove_outliers_iqr(X: np.ndarray) -> np.ndarray:
    """
    Remove outliers from numerical features using the IQR method.

    Values outside [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] are clipped.

    Parameters:
    X : np.ndarray
        Feature matrix.

    Returns:
    np.ndarray
        Feature matrix with outliers capped.
    """
    logger.info("Removing outliers using IQR method")

    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return np.clip(X, lower, upper)


def preprocess(df: pd.DataFrame = None) -> tuple:
    """
    Perform full preprocessing pipeline:
    - Load data (if not provided)
    - Handle missing values
    - Remove outliers
    - Train-test split
    - Save processed datasets

    Parameters:
    df : pd.DataFrame, optional
        Input dataframe. If None, loads from default path.

    Returns:
    tuple
        (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting preprocessing pipeline")

    # Load dataset if not provided
    if df is None:
        data_path = r"data\water_potability.csv"
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Using provided dataframe")

    # Class distribution
    if "Potability" in df.columns:
        class_dist = df["Potability"].value_counts(normalize=True)
        logger.info("Class distribution:\n%s", class_dist)

        # Split features and target
        X = df.drop("Potability", axis=1)
        y = df["Potability"]
    else:
        # Handle case where target might not be present (e.g. inference)
        # For this specific refactor I'll assume training context as per original code
        logger.warning(
            "Potability column missing, assuming inference mode or invalid data for training"
        )
        X = df
        y = None

    # Handle missing values
    logger.info("Imputing missing values using median strategy")
    imputer = SimpleImputer(strategy="median")
    # Wrap in DataFrame to keep columns if needed, but SimpleImputer returns ndarray usually.
    X = imputer.fit_transform(X)

    # Outlier handling
    X = remove_outliers_iqr(X)

    if y is not None:
        # Train-test split
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Save artifacts
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        logger.info(f"Saving processed data to '{artifacts_dir}' directory")

        pd.DataFrame(X_train).to_csv(f"{artifacts_dir}/X_train.csv", index=False)
        pd.DataFrame(X_test).to_csv(f"{artifacts_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{artifacts_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{artifacts_dir}/y_test.csv", index=False)

        logger.info("Preprocessing completed successfully")
        return X_train, X_test, y_train, y_test
    else:
        # If no target, just return processed X (unlikely path for this specific task
        # but good practice)
        return X


if __name__ == "__main__":
    preprocess()