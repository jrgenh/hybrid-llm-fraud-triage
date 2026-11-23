import re
from src.pipeline import HybridFraudPipeline
from src.scoring.tabular_model import FEATURE_COLUMNS, FraudTabularScorer
from src.triage.agent import TriageAgent
from src.triage.schemas import (
    FraudTriageDecision,
    FraudTypology,
    TriageAction,
    TransactionContext,
)


KNOWN_FEATURE_TOKENS = set(FEATURE_COLUMNS) | {
    "ml_risk_score",
    "tx_count_30d_avg_hourly",
    "user_id",
    "transaction_id",
    "merchant_category",
}


class StubTriageAgent(TriageAgent):
    """Deterministic stub for unit testing routing and grounding checks without live API calls."""

    def __init__(self) -> None:
        self.calls: list[TransactionContext] = []

    def triage(self, context: TransactionContext) -> FraudTriageDecision:
        self.calls.append(context)
        top_feature = next(iter(context.top_shap_drivers.keys()))
        device_clause = (
            "from a new device"
            if context.is_new_device
            else "from a previously seen device"
        )
        return FraudTriageDecision(
            transaction_id=context.transaction_id,
            recommended_action=TriageAction.STEP_UP_AUTH,
            confidence=0.78,
            primary_typology=FraudTypology.ACCOUNT_TAKEOVER,
            cited_risk_factors=[
                f"Primary driver {top_feature} with {context.tx_count_1h} tx in 1h",
                f"Distance {context.distance_from_home_km} km and {context.recent_declines_24h} recent declines",
            ],
            sar_narrative=(
                f"Borderline transaction of ${context.amount_usd:.2f} in {context.merchant_category} "
                f"{device_clause}; {top_feature} contributed to a {context.ml_risk_score:.2f} risk score."
            ),
        )


def _assert_grounded_output(ctx: TransactionContext, decision: FraudTriageDecision) -> None:
    combined_text = " ".join([decision.sar_narrative, *decision.cited_risk_factors])

    # Check any snake_case identifiers referenced in the output
    mentioned_tokens = set(re.findall(r"\b[a-z]+_[a-z0-9_]+\b", combined_text))
    unknown_tokens = mentioned_tokens - KNOWN_FEATURE_TOKENS
    assert not unknown_tokens, f"Unrecognized feature/identifier tokens in output: {unknown_tokens}"

    # Prevent contradictory device claims when is_new_device is False
    if not ctx.is_new_device:
        lowered = combined_text.lower()
        assert "new device" not in lowered and "unrecognized device" not in lowered, (
            f"Contradictory device claim for {ctx.transaction_id} where is_new_device=False"
        )


def test_hybrid_routing_and_grounding(sample_scored_df) -> None:
    # Build a lightweight mock/stub scorer using pre-computed scores and SHAP dicts
    class PrecomputedScorer(FraudTabularScorer):
        def __init__(self, df_in) -> None:
            super().__init__(model=None)
            self._df_in = df_in

        def score_dataframe(self, df, top_k: int = 4):
            return self._df_in.copy()

    agent = StubTriageAgent()
    pipeline = HybridFraudPipeline(
        scorer=PrecomputedScorer(sample_scored_df),
        triage_agent=agent,
        low_threshold=0.45,
        high_threshold=0.85,
    )

    results = pipeline.process_dataframe(sample_scored_df)
    by_id = {r.transaction_id: r for r in results}

    assert by_id["tx_low"].path_taken == "fast_approve"
    assert by_id["tx_low"].action == TriageAction.APPROVE

    assert by_id["tx_high"].path_taken == "fast_block"
    assert by_id["tx_high"].action == TriageAction.BLOCK

    assert by_id["tx_mid"].path_taken == "llm_triage"
    assert by_id["tx_mid"].decision is not None
    assert len(agent.calls) == 1
    _assert_grounded_output(agent.calls[0], by_id["tx_mid"].decision)

