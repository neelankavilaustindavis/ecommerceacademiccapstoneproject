"""
Part 4, Tasks 14, 15, 16 — Resilience & Governance:
  - Task 14: 2-Agent Autogen Review Stage (`RoundRobinGroupChat`, `max_turns=2`,
    `custom_message_types=[StructuredMessage[ReviewVerdict]]`)
  - Task 15: Four-Layer AI Governance Model (Application-layer least-autonomy tool
    enforcement, Medium Risk classification & justification, Runtime token/cost budget cap)
  - Task 16: Normalized Query In-Memory Response Caching with before/after evidence
Track: E-commerce & Retail (hypothetical retailer)
"""

from __future__ import annotations
import _offline_compat  # noqa: F401

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from rag_core import RetailRAGEngine
from tools_and_crew import CheckOrderStatusTool, RetailSupportCrewOrchestrator

try:
    from autogen_agentchat.agents import AssistantAgent  # type: ignore
    from autogen_agentchat.conditions import MaxMessageTermination  # type: ignore
    from autogen_agentchat.messages import StructuredMessage  # type: ignore
    from autogen_agentchat.teams import RoundRobinGroupChat  # type: ignore
except ImportError:
    from _offline_compat import (
        AssistantAgent,
        MaxMessageTermination,
        RoundRobinGroupChat,
        StructuredMessage,
    )


class ReviewVerdict(BaseModel):
    """Pydantic structured verdict emitted by Final_Editor (Task 14)."""

    approved: bool = Field(
        ..., description="True if the CrewAI Composer's draft answer is 100% grounded and approved unchanged."
    )
    final_answer: str = Field(
        ..., description="The approved unchanged answer or the revised grounded answer."
    )
    reason: str = Field(
        ..., description="Explanation of why the draft was approved unchanged or what ungrounded claim was removed."
    )


@dataclass
class MockAutogenCompletionResult:
    content: str


class MockAutogenModelClient:
    def __init__(self) -> None:
        self.call_count: int = 0

    @staticmethod
    def _parse_task_payload(task_text: str) -> Dict[str, str]:
        draft_match = re.search(r"DRAFT_ANSWER:\s*(.*?)\nRETRIEVED_CONTEXT:", task_text, re.DOTALL)
        ctx_match = re.search(r"RETRIEVED_CONTEXT:\s*(.*)", task_text, re.DOTALL)
        return {
            "draft": draft_match.group(1).strip() if draft_match else task_text,
            "context": ctx_match.group(1).strip() if ctx_match else "",
        }

    @staticmethod
    def _audit_sentences(draft: str, context: str) -> Tuple[bool, List[str], List[str]]:
        raw_sents = [
            s.strip() + ("." if not s.strip().endswith(".") else "")
            for s in re.split(r"\.\s+", draft.strip())
            if s.strip()
        ]
        ctx_lower = context.lower().replace(",", "")
        ctx_numbers = {float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", ctx_lower)}
        ctx_words = set(re.findall(r"[a-z]{4,}", ctx_lower))

        kept: List[str] = []
        flagged: List[str] = []

        for sent in raw_sents:
            # Strip citation header prefix (e.g. "According to illustrative demo policy (DOC-02):") before numeric audit
            sent_body = re.sub(r"^According to illustrative demo policy \([A-Z0-9,\s\-]+\):\s*", "", sent, flags=re.IGNORECASE)
            s_lower = re.sub(r"tier-\d+", "tier", sent_body.lower().replace(",", ""))
            s_nums = {float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", s_lower)}
            s_words = set(re.findall(r"[a-z]{4,}", s_lower)) - {
                "according", "retail_demo", "policy", "customers", "order", "orders",
                "currently", "status", "value", "created", "delayed", "shipment",
                "escalation", "score", "recommended", "standard", "handling"
            }

            unbacked_nums = s_nums - ctx_numbers
            word_overlap = len(s_words & ctx_words) / max(1, len(s_words))

            if unbacked_nums or word_overlap < 0.30:
                flagged.append(sent)
            else:
                kept.append(sent)

        is_clean = len(flagged) == 0
        return is_clean, kept, flagged

    async def create(
        self,
        messages: List[Any],
        agent_name: str = "",
        system_message: str = "",
        output_content_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> MockAutogenCompletionResult:
        self.call_count += 1
        first_msg = messages[0].content if hasattr(messages[0], "content") else str(messages[0])
        payload = self._parse_task_payload(first_msg)
        is_clean, kept_sents, flagged_sents = self._audit_sentences(
            payload["draft"], payload["context"]
        )

        if output_content_type is None and agent_name == "Policy_Compliance_Reviewer":
            if is_clean:
                critique = (
                    "COMPLIANCE_AUDIT: PASS. Every policy rule, timeline, and numeric figure in the "
                    "CrewAI Composer draft is directly supported by the retrieved knowledge-base context."
                )
            else:
                critique = (
                    "COMPLIANCE_AUDIT: REVISE_REQUIRED. Detected ungrounded claim(s) not present in "
                    f"retrieved context: {flagged_sents}. Retain only verified sentences: {' '.join(kept_sents)}"
                )
            return MockAutogenCompletionResult(content=critique)

        if is_clean:
            verdict = ReviewVerdict(
                approved=True,
                final_answer=payload["draft"],
                reason=(
                    "Approved unchanged: All facts, figures, and citations in the CrewAI draft "
                    "are verified against the retrieved illustrative policy context."
                ),
            )
        else:
            revised_text = " ".join(kept_sents).strip()
            verdict = ReviewVerdict(
                approved=False,
                final_answer=revised_text,
                reason=(
                    "Revised draft: Removed ungrounded claim(s) not supported by retrieved context: "
                    + "; ".join(flagged_sents)
                ),
            )
        return MockAutogenCompletionResult(content=verdict.model_dump_json())


class AutogenReviewStage:
    def __init__(self) -> None:
        self.model_client = MockAutogenModelClient()

    def build_team(self) -> RoundRobinGroupChat:
        reviewer = AssistantAgent(
            name="Policy_Compliance_Reviewer",
            model_client=self.model_client,
            system_message=(
                "You are the Illustrative Policy Compliance Reviewer. Audit the CrewAI Composer's "
                "draft answer against the retrieved context and flag any ungrounded claims."
            ),
        )
        final_editor = AssistantAgent(
            name="Final_Editor",
            model_client=self.model_client,
            system_message=(
                "You are the Retail Demo Final Editor. Based on the reviewer's audit, either approve "
                "the draft unchanged or remove ungrounded claims and emit a structured ReviewVerdict."
            ),
            output_content_type=ReviewVerdict,
        )
        return RoundRobinGroupChat(
            participants=[reviewer, final_editor],
            max_turns=2,
            custom_message_types=[StructuredMessage[ReviewVerdict]],
        )

    def review(self, draft_answer: str, retrieved_context: str) -> Dict[str, Any]:
        team = self.build_team()
        task_prompt = (
            f"Review the following CrewAI Composer draft against the retrieved context.\n"
            f"DRAFT_ANSWER: {draft_answer}\n"
            f"RETRIEVED_CONTEXT: {retrieved_context}"
        )
        task_result = asyncio.run(team.run(task=task_prompt))

        verdict_obj: Optional[ReviewVerdict] = None
        transcript: List[Dict[str, str]] = []
        for msg in task_result.messages:
            src = getattr(msg, "source", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, ReviewVerdict):
                verdict_obj = content
                transcript.append({"source": src, "type": msg.type, "content": content.model_dump_json()})
            else:
                transcript.append({"source": src, "type": getattr(msg, "type", "TextMessage"), "content": str(content)})

        assert verdict_obj is not None, "Final_Editor did not produce a ReviewVerdict"
        validated = ReviewVerdict.model_validate(verdict_obj.model_dump())
        return {
            "verdict": validated,
            "transcript": transcript,
            "stop_reason": task_result.stop_reason,
        }


SYSTEM_RISK_CLASSIFICATION: Dict[str, str] = {
    "risk_level": "Medium",
    "scheme": "Low / Medium / High",
    "justification": (
        "Under the three-tier AI risk classification scheme (Low: passive summarization/transcription; "
        "Medium: code generation and customer-support ticket resolution; High: medical diagnosis, "
        "automated hiring decisions, or autonomous financial execution), the Retail Demo Support Agent "
        "is classified as MEDIUM RISK. While the agent interacts directly with customers regarding "
        "order statuses, return eligibility, COD refund timelines, and tier escalations—where an "
        "inaccurate statement could cause customer dissatisfaction or improper ticket routing—it has "
        "read-only tool access (`rag_lookup` and `check_order_status`), cannot autonomously execute "
        "bank transfers or mutate database state, masks fixed-format PII on ingress, and subjects every "
        "draft response to both an output groundedness guardrail and a two-agent Autogen compliance review."
    ),
}

LEAST_AUTONOMY_EXPLANATION: str = (
    "At the Application governance layer, we enforce the principle of least autonomy by restricting "
    "`check_order_status` exclusively to the `Order Lookup Specialist` (`Lookup Agent`). The "
    "`Policy Retrieval Specialist` (`Retrieval Agent`) is wired only with `rag_lookup`, and the "
    "`Support Response Composer` has zero external data tools. In `CheckOrderStatusTool.__init__` and "
    "`CheckOrderStatusTool._run`, we validate `bound_agent_role == 'Order Lookup Specialist'`; any "
    "attempt in code or at runtime to bind or invoke `check_order_status` from another agent immediately "
    "raises a `PermissionError`, preventing cross-privilege data leakage."
)


class BudgetExceededError(RuntimeError):
    """Raised when a request exceeds the Runtime Layer token or cost budget cap."""


class RuntimeBudgetGuard:
    def __init__(
        self,
        max_tokens_per_request: int = 450,
        cost_per_1k_tokens_inr: float = 1.20,
        max_cost_inr_per_request: float = 0.54,
    ) -> None:
        self.max_tokens_per_request = max_tokens_per_request
        self.cost_per_1k_tokens_inr = cost_per_1k_tokens_inr
        self.max_cost_inr_per_request = max_cost_inr_per_request

    def check_request_budget(self, query_text: str, estimated_completion_tokens: int = 180) -> Dict[str, Any]:
        prompt_tokens = max(1, int(len(query_text.split()) * 1.35))
        total_estimated_tokens = prompt_tokens + estimated_completion_tokens
        estimated_cost_inr = round((total_estimated_tokens / 1000.0) * self.cost_per_1k_tokens_inr, 4)

        if (
            total_estimated_tokens > self.max_tokens_per_request
            or estimated_cost_inr > self.max_cost_inr_per_request
        ):
            raise BudgetExceededError(
                f"Runtime Governance Budget Cap Exceeded: Estimated request size is "
                f"{total_estimated_tokens} tokens (₹{estimated_cost_inr:.4f} INR), which exceeds "
                f"the per-request cap of {self.max_tokens_per_request} tokens "
                f"(₹{self.max_cost_inr_per_request:.4f} INR). Request rejected before LLM execution."
            )

        return {
            "allowed": True,
            "prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "total_estimated_tokens": total_estimated_tokens,
            "estimated_cost_inr": estimated_cost_inr,
            "max_tokens_cap": self.max_tokens_per_request,
            "max_cost_inr_cap": self.max_cost_inr_per_request,
        }


def demonstrate_response_caching(rag_engine: Optional[RetailRAGEngine] = None) -> Dict[str, Any]:
    engine = rag_engine or RetailRAGEngine()
    engine.response_cache.clear()
    engine.cache_hits = 0
    engine.cache_misses = 0
    engine.llm_generation_calls = 0

    q_first = "How many illustrative loyalty points equal 1 INR at checkout?"
    calls_before_first = engine.llm_generation_calls
    res1 = engine.grounded_generate(q_first, strategy="sentence", top_k=3, use_cache=True)
    calls_after_first = engine.llm_generation_calls

    q_repeat = "  how many illustrative loyalty points equal 1 inr at checkout???  "
    calls_before_second = engine.llm_generation_calls
    res2 = engine.grounded_generate(q_repeat, strategy="sentence", top_k=3, use_cache=True)
    calls_after_second = engine.llm_generation_calls

    return {
        "first_call": {
            "query": q_first,
            "normalized_key": res1["normalized_query"],
            "cache_hit": res1["cache_hit"],
            "llm_calls_before": calls_before_first,
            "llm_calls_after": calls_after_first,
            "llm_calls_delta": calls_after_first - calls_before_first,
            "latency_ms": res1["latency_ms"],
        },
        "second_call": {
            "query": q_repeat,
            "normalized_key": res2["normalized_query"],
            "cache_hit": res2["cache_hit"],
            "llm_calls_before": calls_before_second,
            "llm_calls_after": calls_after_second,
            "llm_calls_delta": calls_after_second - calls_before_second,
            "latency_ms": res2["latency_ms"],
        },
        "total_cache_hits": engine.cache_hits,
        "total_cache_misses": engine.cache_misses,
    }


def print_part4_governance_report() -> None:
    print("=" * 72)
    print("PART 4 TASKS 14, 15, 16 — AUTOGEN REVIEW, AI GOVERNANCE & CACHING")
    print("=" * 72)

    orchestrator = RetailSupportCrewOrchestrator()
    review_stage = AutogenReviewStage()

    print("\n--- Task 14: 2-Agent Autogen Review Stage (Approve vs. Revise) ---")
    q1 = "What is the Cash on Delivery (COD) refund timeline for illustrative wallet versus NEFT?"
    crew_res1 = orchestrator.run_crew(q1)
    ctx1 = orchestrator.rag_engine.doc_map["DOC-02"]["text"]
    rev1 = review_stage.review(draft_answer=crew_res1.final_answer, retrieved_context=ctx1)
    v1: ReviewVerdict = rev1["verdict"]
    print(
        f"  [Case 14A - Grounded Draft Approved Unchanged]\n"
        f"    Crew Draft  : {crew_res1.final_answer}\n"
        f"    Verdict JSON: {v1.model_dump_json()}"
    )

    injected_draft = (
        f"{crew_res1.final_answer} "
        "Additionally, hypothetical retailer pays ₹2,500 bonus compensation if your COD refund takes over 48 hours."
    )
    rev2 = review_stage.review(draft_answer=injected_draft, retrieved_context=ctx1)
    v2: ReviewVerdict = rev2["verdict"]
    print(
        f"\n  [Case 14B - Injected Ungrounded Claim Caught & Revised]\n"
        f"    Tainted Draft: {injected_draft}\n"
        f"    Reviewer Msg : {rev2['transcript'][1]['content']}\n"
        f"    Verdict JSON : {v2.model_dump_json()}"
    )

    print("\n--- Task 15A: Application-Layer Least Autonomy Enforcement ---")
    print(f"  Policy: {LEAST_AUTONOMY_EXPLANATION}")
    try:
        _illegal_tool = CheckOrderStatusTool(bound_agent_role="Policy Retrieval Specialist")
    except PermissionError as exc:
        print(f"  [PASS - Unauthorized Tool Wiring Blocked]: {exc}")

    print("\n--- Task 15B: System Risk Classification & Justification ---")
    print(f"  Risk Level   : {SYSTEM_RISK_CLASSIFICATION['risk_level']} ({SYSTEM_RISK_CLASSIFICATION['scheme']})")
    print(f"  Justification: {SYSTEM_RISK_CLASSIFICATION['justification']}")

    print("\n--- Task 15C: Runtime-Layer Token & Cost Budget Cap ---")
    budget_guard = RuntimeBudgetGuard(max_tokens_per_request=450, max_cost_inr_per_request=0.54)
    normal_check = budget_guard.check_request_budget(
        "What is the return window for Footwear order ORD-1001?"
    )
    print(f"  [Normal Request Budget Check]: {normal_check}")
    oversized_query = "Please explain every detail of hypothetical retailer policy. " * 60
    try:
        budget_guard.check_request_budget(oversized_query)
    except BudgetExceededError as exc:
        print(f"  [PASS - Oversized Request Rejected]: {exc}")

    print("\n--- Task 16: Normalized In-Memory Response Caching ---")
    cache_demo = demonstrate_response_caching(orchestrator.rag_engine)
    print(f"  1st Call (Cold Miss): {json.dumps(cache_demo['first_call'])}")
    print(f"  2nd Call (Warm Hit) : {json.dumps(cache_demo['second_call'])}")
    print(
        f"  Summary: Cache Hits={cache_demo['total_cache_hits']}, "
        f"Cache Misses={cache_demo['total_cache_misses']}, "
        f"LLM Calls Saved on Repeat={1 - cache_demo['second_call']['llm_calls_delta']}"
    )
    print("=" * 72)


if __name__ == "__main__":
    print_part4_governance_report()
