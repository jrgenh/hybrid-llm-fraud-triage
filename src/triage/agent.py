import os
from typing import Protocol
from openai import OpenAI
from src.triage.schemas import FraudTriageDecision, TransactionContext

SYSTEM_PROMPT = """You are a senior financial risk and fraud triage analyst.
Given a transaction's tabular risk score, signed SHAP feature attributions, and behavioral context:
1. Classify the primary fraud typology and recommend an operational action:
   - APPROVE: Legitimate anomaly or low-risk behavior with clear explanation.
   - STEP_UP_AUTH: Ambiguous or elevated risk warranting 3DS/OTP verification before settlement.
   - BLOCK: Strong multi-signal indicators of account takeover or card testing.
2. Ground every claim strictly in the provided telemetry and SHAP drivers. Never invent device attributes, locations, or transaction counts not present in the input.
3. Keep the SAR narrative concise, objective, and audit-ready.
"""


class TriageAgent(Protocol):
    def triage(self, context: TransactionContext) -> FraudTriageDecision: ...


class LLMFraudTriageAgent:
    """Schema-enforced LLM triage agent for gray-area fraud investigations."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _build_user_prompt(self, ctx: TransactionContext) -> str:
        velocity_ratio = (
            round(ctx.tx_count_1h / ctx.tx_count_30d_avg_hourly, 2)
            if ctx.tx_count_30d_avg_hourly > 0
            else float(ctx.tx_count_1h)
        )
        shap_lines = "\n".join(
            f"  - {feat}: {val:+.4f}" for feat, val in ctx.top_shap_drivers.items()
        )
        return (
            f"Evaluate this borderline transaction:\n"
            f"- Transaction ID: {ctx.transaction_id}\n"
            f"- User ID: {ctx.user_id}\n"
            f"- Amount (USD): ${ctx.amount_usd:,.2f}\n"
            f"- Merchant Category: {ctx.merchant_category}\n"
            f"- LightGBM Risk Score: {ctx.ml_risk_score:.4f}\n"
            f"- 1h Tx Count: {ctx.tx_count_1h} (30d hourly baseline: {ctx.tx_count_30d_avg_hourly}, "
            f"ratio: {velocity_ratio}x)\n"
            f"- New Device: {ctx.is_new_device}\n"
            f"- Distance From Home (km): {ctx.distance_from_home_km}\n"
            f"- Recent Declines (24h): {ctx.recent_declines_24h}\n"
            f"- Top SHAP Drivers (log-odds impact toward fraud):\n{shap_lines}"
        )

    def triage(self, context: TransactionContext) -> FraudTriageDecision:
        # print("   --- running triage() in agent.py " + str(context)[0:100])
        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=self._build_user_prompt(context),
            text_format=FraudTriageDecision,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"Failed to parse structured triage output for {context.transaction_id}")
        return response.output_parsed
