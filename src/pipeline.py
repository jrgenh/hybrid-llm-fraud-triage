from dataclasses import dataclass
from typing import Any
import pandas as pd
from src.scoring.tabular_model import FraudTabularScorer
from src.triage.agent import TriageAgent
from src.triage.schemas import FraudTriageDecision, TriageAction, TransactionContext


@dataclass
class TriageRouteResult:
    transaction_id: str
    ml_risk_score: float
    path_taken: str  # "fast_approve" | "fast_block" | "llm_triage"
    action: TriageAction
    decision: FraudTriageDecision | None = None


class HybridFraudPipeline:
    """Combines fast tabular scoring with selective LLM triage for borderline cases."""

    def __init__(
        self,
        scorer: FraudTabularScorer,
        triage_agent: TriageAgent,
        low_threshold: float = 0.45,     ### overwritten
        high_threshold: float = 0.85,    ### overwritten
    ) -> None:
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError("Thresholds must satisfy 0.0 <= low_threshold < high_threshold <= 1.0")
        self.scorer = scorer
        self.triage_agent = triage_agent
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @staticmethod
    def _row_to_context(row: pd.Series) -> TransactionContext:
        raw_drivers: dict[str, Any] = row["top_shap_drivers"]
        drivers = {str(k): float(v) for k, v in raw_drivers.items()}
        return TransactionContext(
            transaction_id=str(row["transaction_id"]),
            user_id=str(row["user_id"]),
            amount_usd=float(row["amount_usd"]),
            merchant_category=str(row["merchant_category"]),
            ml_risk_score=float(row["ml_risk_score"]),
            top_shap_drivers=drivers,
            tx_count_1h=int(row["tx_count_1h"]),
            tx_count_30d_avg_hourly=float(row["tx_count_30d_avg_hourly"]),
            is_new_device=bool(row["is_new_device"]),
            distance_from_home_km=float(row["distance_from_home_km"]),
            recent_declines_24h=int(row["recent_declines_24h"]),
        )

    def process_dataframe(self, df: pd.DataFrame) -> list[TriageRouteResult]:
        scored_df = self.scorer.score_dataframe(df)
        results: list[TriageRouteResult] = []

        line_cnt: int = -1
        for _, row in scored_df.iterrows():
            line_cnt += 1
            
            score = float(row["ml_risk_score"])
            tx_id = str(row["transaction_id"])

            # if (line_cnt < 50):
            if ((score > 0) and (score < 1)):
                print('   --- in process_dataframe() in pipeline.py ' + ' line_cnt: ' + str(line_cnt) + '  score: ' + str(score) + '  low_threshold: ' + str(self.low_threshold))

            if score < self.low_threshold:
                results.append(
                    TriageRouteResult(
                        transaction_id=tx_id,
                        ml_risk_score=score,
                        path_taken="fast_approve",
                        action=TriageAction.APPROVE,
                    )
                )
            elif score > self.high_threshold:
                results.append(
                    TriageRouteResult(
                        transaction_id=tx_id,
                        ml_risk_score=score,
                        path_taken="fast_block",
                        action=TriageAction.BLOCK,
                    )
                )
            else:
                ctx = self._row_to_context(row)
                print("   --- running in process_dataframe() in pipeline.py " + str(ctx)[0:100])
                decision = self.triage_agent.triage(ctx)
                results.append(
                    TriageRouteResult(
                        transaction_id=tx_id,
                        ml_risk_score=score,
                        path_taken="llm_triage",
                        action=decision.recommended_action,
                        decision=decision,
                    )
                )

        print('   --- final line_cnt: ' + str(line_cnt))

        return results
