"""Leakage-safe feature cleaning and sklearn pipeline construction."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

class CustomerDataCleaner(BaseEstimator, TransformerMixin):
    """Convert raw CRM-style fields to model-ready pandas dtypes."""
    def fit(self, X: pd.DataFrame, y: Any = None) -> "CustomerDataCleaner":
        self._validate_columns(X); return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X); result = X.loc[:, FEATURE_COLUMNS].copy()
        for column in NUMERIC_FEATURES:
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for column in CATEGORICAL_FEATURES:
            result[column] = result[column].map(lambda value: np.nan if pd.isna(value) else str(value))
        return result
    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame): raise TypeError("Expected a pandas DataFrame as model input.")
        missing = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing: raise ValueError(f"Missing required feature columns: {missing}")

def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("cleaner", CustomerDataCleaner()), ("preprocess", preprocess), ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=1, class_weight="balanced_subsample"))])
