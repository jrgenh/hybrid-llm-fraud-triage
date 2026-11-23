from enum import Enum
from pydantic import BaseModel, Field


class TriageAction(str, Enum):
    APPROVE = "APPROVE"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    BLOCK = "BLOCK"


class FraudTypology(str, Enum):
    LEGITIMATE_ANOMALY = "LEGITIMATE_ANOMALY"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    CARD_NOT_PRESENT_RING = "CARD_NOT_PRESENT_RING"
    CARD_TESTING = "CARD_TESTING"
    FRIENDLY_FRAUD = "FRIENDLY_FRAUD"


class TransactionContext(BaseModel):
    """Input payload assembled from Tier 1 (ML) and Tier 2 (Feature Store)."""
    transaction_id: str
    user_id: str
    amount_usd: float
    merchant_category: str
    ml_risk_score: float = Field(..., ge=0.0, le=1.0)
    top_shap_drivers: dict[str, float] = Field(
        ..., description="Feature name mapped to its signed SHAP log-odds impact"
    )
    tx_count_1h: int
    tx_count_30d_avg_hourly: float
    is_new_device: bool
    distance_from_home_km: float
    recent_declines_24h: int


class FraudTriageDecision(BaseModel):
    """Schema-enforced output from the Tier 3 LLM investigation layer."""
    transaction_id: str
    recommended_action: TriageAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    primary_typology: FraudTypology
    cited_risk_factors: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Explicit telemetry metrics or SHAP drivers supporting the decision.",
    )
    sar_narrative: str = Field(
        ...,
        max_length=600,
        description="Concise, auditable Suspicious Activity Report summary grounded strictly in input context.",
    )
