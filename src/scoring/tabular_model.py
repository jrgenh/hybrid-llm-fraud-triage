from pathlib import Path
from typing import Any
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

FEATURE_COLUMNS: list[str] = [
    "amount_usd",
    "merchant_category",
    "tx_count_1h",
    "tx_count_30d_avg_hourly",
    "is_new_device",
    "distance_from_home_km",
    "recent_declines_24h",
]

CATEGORICAL_COLUMNS: list[str] = ["merchant_category"]


class FraudTabularScorer:
    """Trains and serves a LightGBM fraud model with local SHAP explanations."""

    def __init__(self, model: lgb.LGBMClassifier | None = None) -> None:
        self.model = model
        self._explainer: shap.TreeExplainer | None = None

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS].copy()
        for col in CATEGORICAL_COLUMNS:
            X[col] = X[col].astype("category")
        X["is_new_device"] = X["is_new_device"].astype(int)
        return X

    def fit(self, df: pd.DataFrame, target_col: str = "is_fraud") -> "FraudTabularScorer":
        X = self._prepare_features(df)
        y = df[target_col].astype(int)

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=5000,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        self._explainer = shap.TreeExplainer(self.model)
        return self

    def _get_shap_matrix(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted or loaded.")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)

        shap_values = self._explainer.shap_values(X)
        # For binary classification, TreeExplainer may return a list [neg, pos] or 3D array
        if isinstance(shap_values, list):
            return np.asarray(shap_values[1])
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return arr[:, :, 1]
        return arr

    def score_dataframe(self, df: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
        """Returns risk probabilities and top signed SHAP drivers per row."""
        if self.model is None:
            raise RuntimeError("Model is not fitted or loaded.")

        X = self._prepare_features(df)
        probs = self.model.predict_proba(X)[:, 1]
        shap_mat = self._get_shap_matrix(X)

        top_drivers_list: list[dict[str, float]] = []
        for row_idx in range(len(X)):
            row_shap = shap_mat[row_idx]
            ranked_indices = np.argsort(np.abs(row_shap))[::-1][:top_k]
            drivers = {
                FEATURE_COLUMNS[col_idx]: round(float(row_shap[col_idx]), 4)
                for col_idx in ranked_indices
            }
            top_drivers_list.append(drivers)

        out = df.copy()
        out["ml_risk_score"] = np.round(probs, 4)
        out["top_shap_drivers"] = top_drivers_list
        return out

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model}, target)

    @classmethod
    def load(cls, path: str | Path) -> "FraudTabularScorer":
        payload: dict[str, Any] = joblib.load(path)
        return cls(model=payload["model"])
