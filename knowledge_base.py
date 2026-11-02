"""
Part 1, Task 2 — Illustrative Retail Policy Knowledge Base (knowledge_base.py)
Track: E-commerce & Retail (unofficial educational demo)

Contains 12 fictional policy examples for software testing. They are not actual or verified policies of any company.
"""

from __future__ import annotations

from typing import Dict, List

ILLUSTRATIVE_ONLY: bool = True

KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "doc_id": "DOC-01",
        "topic": "return_window_by_category",
        "title": "Return Window by Product Category",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "For this example, category-specific return windows are measured from the confirmed delivery date. "
            "Beauty and personal-care products are eligible for return within 15 days only if the outer seal, hygiene sticker, and batch packaging remain completely unopened and unused. "
            "Apparel and Footwear orders qualify for a 30-day return window provided all original brand tags, boxes, and dust bags are intact, whereas Home decor items carry a 15-day return window. "
            "Electronics and personal grooming appliances have a 7-day return window strictly limited to dead-on-arrival or verified manufacturing defects."
        ),
    },
    {
        "doc_id": "DOC-02",
        "topic": "cod_refund_timelines",
        "title": "Cash on Delivery (COD) Refund Timelines",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Refunds for Cash on Delivery (COD) orders are initiated only after the returned package passes physical quality verification at the hypothetical retailer fulfillment center. "
            "Customers who opt for an illustrative wallet refund receive the full credit within 24 hours of quality-check approval for immediate reuse on future orders. "
            "Customers who select a direct NEFT bank transfer must provide a valid Indian bank account and IFSC code in the app, and funds settle within 5 to 7 business days after approval."
        ),
    },
    {
        "doc_id": "DOC-03",
        "topic": "delivery_slas",
        "title": "Standard and Express Delivery SLAs",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "In this example, standard retail orders shipped to Tier-1 metro cities (Mumbai, Delhi NCR, Bengaluru, Hyderabad, Chennai, Kolkata, and Pune) are delivered within 2 to 3 business days of dispatch. "
            "Orders addressed to Tier-2 and Tier-3 cities have a standard delivery SLA of 4 to 6 business days, while remote or northeastern PIN codes require 7 to 9 business days. "
            "For selected metro PIN codes, the example allows same-day or next-day express delivery for stocked items when the order is placed before 1:00 PM IST."
        ),
    },
    {
        "doc_id": "DOC-04",
        "topic": "reverse_pickup_eligibility",
        "title": "Reverse-Pickup Eligibility and Self-Ship Reimbursement",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Doorstep reverse pickup is available across more than 18,000 serviceable Indian PIN codes and is scheduled within 48 hours of return request approval. "
            "Up to two courier pickup attempts are made, and all items must be handed over with the original invoice and protective packaging. "
            "If a customer's PIN code is not serviceable for reverse pickup, the customer may self-ship the parcel via a tracked courier and upload the receipt to receive up to ₹150 shipping reimbursement to the demonstration wallet."
        ),
    },
    {
        "doc_id": "DOC-05",
        "topic": "warranty_terms_by_category",
        "title": "Warranty Terms by Product Category",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Electronics and hair-styling appliances sold by the hypothetical retailer carry a 1-year to 2-year manufacturer warranty covering internal motor, heating element, and circuit defects, claimed using the tax invoice. "
            "Footwear and Apparel items include a 30-day warranty against sole delamination, zipper failure, and seam stitching defects under normal wear. "
            "Consumable Beauty cosmetics and skincare products do not carry a post-opening warranty, though the example assumes 100% brand authenticity and at least 6 months of remaining shelf life at delivery."
        ),
    },
    {
        "doc_id": "DOC-06",
        "topic": "order_cancellation_policy",
        "title": "Order and Item Cancellation Policy",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Customers can cancel an entire order or individual items directly from the My Orders section of the demonstration app while the order status is Placed and before warehouse packing completes. "
            "Once an order transitions to Shipped status and receives an airway bill number, self-service cancellation is disabled because the parcel is already in transit with the courier partner. "
            "For orders already in Shipped status, customers may politely refuse the package at the doorstep, and a full refund is automatically triggered once the courier marks Return-to-Origin (RTO)."
        ),
    },
    {
        "doc_id": "DOC-07",
        "topic": "loyalty_points_redemption_policy",
        "title": "Illustrative Loyalty-Points Redemption Example",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Illustrative loyalty points can be redeemed at checkout at a fixed conversion rate where 100 loyalty points equal ₹1 INR off the payable total. "
            "A minimum payable cart value of ₹499 after coupon discounts is required to apply loyalty points, and points cannot be combined with instant bank-card flash-sale offers. "
            "Unredeemed loyalty points expire exactly 12 months from the date they were credited, and points spent on a cancelled or returned order are restored within 24 hours."
        ),
    },
    {
        "doc_id": "DOC-08",
        "topic": "payment_failure_retry_policy",
        "title": "Payment-Failure and Auto-Reversal Policy",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "When an online payment via UPI, credit card, debit card, or netbanking fails at checkout, the hypothetical retailer reserves the inventory and applied promo codes for 30 minutes so the customer can retry payment from the order details screen. "
            "If the customer's bank account or card is debited during a failed transaction, the payment gateway reconciles the ledger within 24 hours and either confirms the order or initiates an automatic refund. "
            "Auto-reversed payments settle back to the original UPI handle within 1 to 2 business days or to the issuing card or bank account within 5 to 7 business days."
        ),
    },
    {
        "doc_id": "DOC-09",
        "topic": "size_exchange_policy",
        "title": "Apparel and Footwear Size-Exchange Policy",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Apparel and Footwear items qualify for a one-time free size exchange for a different size of the exact same product SKU within 30 days of delivery. "
            "The replacement size is dispatched immediately once the courier agent completes the doorstep pickup and tag inspection of the original item. "
            "If the requested replacement size is out of stock at the time of exchange, the hypothetical retailer automatically converts the request into a full refund to the customer's original payment method."
        ),
    },
    {
        "doc_id": "DOC-10",
        "topic": "damaged_item_claim_process",
        "title": "Damaged, Leaked, or Missing Item Claim Process",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Claims for shattered glass, leaked liquids, damaged outer cartons, or missing items must be submitted within 48 hours of delivery through the demonstration help center. "
            "The customer must upload clear photos of the outer shipping label, the printed product batch code, and the damaged contents for verification. "
            "Once verified by the hypothetical quality-assurance team within 24 hours, a free replacement or full refund is issued immediately, and customers are never asked to ship back hazardous broken glass."
        ),
    },
    {
        "doc_id": "DOC-11",
        "topic": "international_shipping_restrictions",
        "title": "International Shipping Restrictions and Foreign Card Acceptance",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "the example retailer delivers only to domestic Indian PIN codes and does not offer international shipping outside India. "
            "International air export is restricted because beauty catalogs frequently contain pressurized aerosols, alcohol-based perfumes, and flammable nail lacquers governed by IATA dangerous-goods rules. "
            "Customers residing abroad may still place gift orders using 3D-Secure enabled international Visa or Mastercard cards provided the delivery address and recipient phone number are within India."
        ),
    },
    {
        "doc_id": "DOC-12",
        "topic": "customer_support_escalation_matrix",
        "title": "Customer-Support Escalation Matrix and Resolution SLAs",
        "text": (
            "Illustrative educational example only; this is not an actual retailer policy or customer commitment. "
            "Tier-1 automated and frontline support resolves routine order-tracking and policy inquiries immediately upon contact. "
            "Orders whose computed escalation score is at or above 0.65, or tickets unresolved after 24 hours, are escalated to Tier-2 Senior Resolution Specialists with a 4-hour response SLA. "
            "Disputes remaining open beyond 72 hours or involving high-value orders above ₹10,000 are routed to the Tier-3 Nodal Grievance Officer, with a target response time of 24 business hours."
        ),
    },
]


def validate_knowledge_base() -> None:
    assert len(KNOWLEDGE_BASE) >= 12, f"Expected >= 12 docs, got {len(KNOWLEDGE_BASE)}"
    for doc in KNOWLEDGE_BASE:
        sentences = [s.strip() for s in doc["text"].split(".") if s.strip()]
        assert 2 <= len(sentences) <= 5, (
            f"{doc['doc_id']} has {len(sentences)} sentences (must be 2–5)"
        )


if __name__ == "__main__":
    validate_knowledge_base()
    print(f"Validated {len(KNOWLEDGE_BASE)} knowledge base documents (all 2–5 sentences).")
    for d in KNOWLEDGE_BASE:
        sc = len([s.strip() for s in d["text"].split(".") if s.strip()])
        print(f"  - {d['doc_id']} ({d['topic']}): {sc} sentences | {d['title']}")
