import pandas as pd
import pytest


@pytest.fixture
def sample_scored_df() -> pd.DataFrame:
    """Provides three transactions spanning low, gray-area, and high ML risk bands."""
    return pd.DataFrame(
        [
            {
                "transaction_id": "tx_low",
                "user_id": "usr_1001",
                "amount_usd": 42.50,
                "merchant_category": "groceries",
                "tx_count_1h": 1,
                "tx_count_30d_avg_hourly": 0.12,
                "is_new_device": False,
                "distance_from_home_km": 6.4,
                "recent_declines_24h": 0,
                "ml_risk_score": 0.12,
                "top_shap_drivers": {
                    "distance_from_home_km": -0.42,
                    "tx_count_1h": -0.18,
                    "amount_usd": -0.11,
                    "is_new_device": -0.09,
                },
            },
            {
                "transaction_id": "tx_mid",
                "user_id": "usr_1042",
                "amount_usd": 1180.00,
                "merchant_category": "electronics",
                "tx_count_1h": 3,
                "tx_count_30d_avg_hourly": 0.08,
                "is_new_device": True,
                "distance_from_home_km": 940.0,
                "recent_declines_24h": 1,
                "ml_risk_score": 0.64,
                "top_shap_drivers": {
                    "distance_from_home_km": 0.34,
                    "is_new_device": 0.25,
                    "tx_count_1h": 0.19,
                    "amount_usd": 0.14,
                },
            },
            {
                "transaction_id": "tx_high",
                "user_id": "usr_1088",
                "amount_usd": 2750.00,
                "merchant_category": "digital_goods",
                "tx_count_1h": 11,
                "tx_count_30d_avg_hourly": 0.05,
                "is_new_device": True,
                "distance_from_home_km": 2650.0,
                "recent_declines_24h": 5,
                "ml_risk_score": 0.93,
                "top_shap_drivers": {
                    "tx_count_1h": 0.88,
                    "recent_declines_24h": 0.67,
                    "distance_from_home_km": 0.54,
                    "is_new_device": 0.41,
                },
            },
        ]
    )

