import os
from data.synthetic_generator import generate_transactions
from src.pipeline import HybridFraudPipeline
from src.scoring.tabular_model import FraudTabularScorer
from src.triage.agent import LLMFraudTriageAgent


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment before running live LLM triage.")

    train_size = 5000
    eval_size = 1000 # 250
    train_df = generate_transactions(n_samples=train_size, fraud_rate=0.04, seed=42)
    eval_df = generate_transactions(n_samples=eval_size, fraud_rate=0.04, seed=99)

    scorer = FraudTabularScorer().fit(train_df)
    triage_agent = LLMFraudTriageAgent(model="gpt-4o-mini")

    pipeline = HybridFraudPipeline(
        scorer=scorer,
        triage_agent=triage_agent,
      # low_threshold=0.45,
        low_threshold = 0.04,
        high_threshold=0.85,
    )

    print('   --- In main() in run_triage.py')
    results = pipeline.process_dataframe(eval_df)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.path_taken] = counts.get(r.path_taken, 0) + 1

    print("Routing summary:", counts)
    for r in results:
        if r.path_taken == "llm_triage" and r.decision is not None:
            print("\n--- LLM Triage Decision ---")
            print(r.decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()


### Run with:
### cd .../projects/2026_11_22_hybrid_LLM_and_tabular_fraud
### export OPENAI_API_KEY="your-api-key"
### PYTHONPATH=. .venv/bin/python -m src.run_triage
### PYTHONPATH=. uv run python -m src.run_triage

