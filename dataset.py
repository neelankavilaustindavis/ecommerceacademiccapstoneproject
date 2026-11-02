"""Synthetic retail order fixtures for an educational support-agent demo.

Every record is generated locally and deterministically. No real retailer,
customer, order, or transaction data is represented.
"""
from __future__ import annotations
import random
from collections import Counter
from typing import Any, Dict, List

SEED = 42
NUM_ORDERS = 50
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.22, 0.18, 0.16, 0.18, 0.26]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.16, 0.26, 0.34, 0.12, 0.12]
ORDER_VALUE_MIN_INR = 299
ORDER_VALUE_MAX_INR = 14999
DELAYED_SHIPMENT_WEIGHT = 0.22


def generate_orders(seed: int = SEED, num_orders: int = NUM_ORDERS) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for i in range(1, num_orders + 1):
        records.append({
            "record_id": f"ORD-{1000+i}",
            "category": rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0],
            "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0],
            "order_value_inr": rng.randint(ORDER_VALUE_MIN_INR, ORDER_VALUE_MAX_INR),
            "days_since_created": rng.randint(0, 30),
            "delayed_shipment": rng.random() < DELAYED_SHIPMENT_WEIGHT,
        })
    return records

ORDERS = generate_orders()
ORDERS_BY_ID = {row["record_id"]: row for row in ORDERS}


def validate_and_summarize_dataset(orders: List[Dict[str, Any]] = ORDERS) -> Dict[str, Any]:
    assert len(orders) >= 40
    categories = Counter(r["category"] for r in orders)
    statuses = Counter(r["status"] for r in orders)
    assert all(categories[c] >= 3 for c in CATEGORIES)
    assert all(statuses[s] >= 1 for s in STATUSES)
    delayed = sum(bool(r["delayed_shipment"]) for r in orders)
    pct = delayed / len(orders) * 100
    assert 10 <= pct <= 30
    days = sorted(r["days_since_created"] for r in orders)
    return {"seed": SEED, "total_records": len(orders), "category_counts": dict(categories),
            "status_counts": dict(statuses), "delayed_count": delayed,
            "delayed_percentage": round(pct, 2),
            "order_value_range_inr": (ORDER_VALUE_MIN_INR, ORDER_VALUE_MAX_INR),
            "days_since_created_p80": days[int(.8 * (len(days)-1))]}


def print_dataset_report() -> None:
    print("SYNTHETIC RETAIL ORDER DATASET")
    print(validate_and_summarize_dataset())

if __name__ == "__main__":
    print_dataset_report()
