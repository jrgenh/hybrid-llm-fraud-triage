import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd

def generate_transactions(n_samples: int = 5000, fraud_rate: float = 0.04, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    records = []
    categories = ["groceries", "electronics", "travel", "digital_goods", "dining", "utilities"]

    for i in range(n_samples):
        is_fraud = rng.random() < fraud_rate
        tx_id = f"tx_{i:06d}"
        user_id = f"usr_{rng.integers(1000, 2500)}"
        base_hourly_tx = round(float(rng.uniform(0.02, 0.25)), 3)

        if not is_fraud:
            is_gray_legit = rng.random() < 0.10
            amount = round(float(rng.lognormal(mean=4.2 if not is_gray_legit else 6.1, sigma=0.7)), 2)
            category = "travel" if is_gray_legit else random.choice(categories)
            tx_count_1h = int(rng.choice([1, 2], p=[0.85, 0.15]))
            is_new_device = bool(rng.random() < (0.25 if is_gray_legit else 0.03))
            distance_km = round(float(rng.uniform(300, 2500) if is_gray_legit else rng.exponential(15)), 1)
            recent_declines = int(rng.choice([0, 1], p=[0.95, 0.05]))
            typology = "LEGITIMATE_ANOMALY" if is_gray_legit else "NORMAL"
        else:
            typology = random.choice(["ACCOUNT_TAKEOVER", "CARD_TESTING"])
            if typology == "ACCOUNT_TAKEOVER":
                amount = round(float(rng.uniform(650.0, 3200.0)), 2)
                category = random.choice(["electronics", "digital_goods"])
                tx_count_1h = int(rng.integers(2, 6))
                is_new_device = True
                distance_km = round(float(rng.uniform(500, 4000)), 1)
                recent_declines = int(rng.integers(1, 4))
            else:
                amount = round(float(rng.uniform(1.00, 15.00)), 2)
                category = "digital_goods"
                tx_count_1h = int(rng.integers(5, 18))
                is_new_device = bool(rng.random() < 0.7)
                distance_km = round(float(rng.uniform(50, 1500)), 1)
                recent_declines = int(rng.integers(2, 8))

        records.append({
            "transaction_id": tx_id,
            "user_id": user_id,
            "amount_usd": amount,
            "merchant_category": category,
            "tx_count_1h": tx_count_1h,
            "tx_count_30d_avg_hourly": base_hourly_tx,
            "is_new_device": is_new_device,
            "distance_from_home_km": distance_km,
            "recent_declines_24h": recent_declines,
            "is_fraud": int(is_fraud),
            "ground_truth_typology": typology,
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud triage dataset.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.04)
    parser.add_argument("--out", type=str, default="data/transactions.parquet")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_transactions(n_samples=args.samples, fraud_rate=args.fraud_rate)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} transactions ({df['is_fraud'].sum()} fraud) to {out_path}")
