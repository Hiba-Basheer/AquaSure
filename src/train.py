"""
Model training script for Water Potability prediction using XGBoost.

Steps:
- Load preprocessed training and testing data
- Train an XGBoost classifier
- Evaluate model performance
- Log metrics and model to MLflow
- Persist trained model locally
"""

import os
import logging
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train() -> None:
    """
    Train an XGBoost classifier for water potability prediction,
    evaluate it, and log results using MLflow.
    """
    logger.info("Starting model training")

    # Set MLflow experiment
    mlflow.set_experiment("Water Potability XGBoost")

    # Load data
    logger.info("Loading training and testing datasets")
    X_train = pd.read_csv("artifacts/X_train.csv")
    X_test = pd.read_csv("artifacts/X_test.csv")
    y_train = pd.read_csv("artifacts/y_train.csv").values.ravel()
    y_test = pd.read_csv("artifacts/y_test.csv").values.ravel()

    with mlflow.start_run():
        logger.info("Initializing XGBoost model")

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            scale_pos_weight=2.5,
            eval_metric="logloss",
            random_state=42
        )

        # Train model
        logger.info("Training XGBoost model")
        model.fit(X_train, y_train)

        # Predictions
        logger.info("Generating predictions")
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba > 0.4).astype(int)

        # Evaluation
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision, recall, _ = precision_recall_curve(y_test, proba)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        # Log metrics
        logger.info("Logging metrics to MLflow")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("avg_precision", precision.mean())

        # Log model
        logger.info("Logging model to MLflow")
        mlflow.xgboost.log_model(model, "model")

    # Save model locally
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/xgboost_model.pkl")
    logger.info("Model saved to model/xgboost_model.pkl")

    logger.info("Training completed successfully")


if __name__ == "__main__":
    train()
