"""Central configuration and feature contract for the churn service."""
from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model_bundle.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DATA_URL = ("https://raw.githubusercontent.com/IBM/" "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
EXPECTED_DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_RAW_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
FALSE_NEGATIVE_COST = 250.0
FALSE_POSITIVE_COST = 40.0
PSI_ALERT_THRESHOLD = 0.20
