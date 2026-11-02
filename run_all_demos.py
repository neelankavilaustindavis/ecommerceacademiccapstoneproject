"""
Master Verification & Graded Transcript Runner (`run_all_demos.py`)
Track: E-commerce & Retail (hypothetical retailer)

Runs all 16 tasks across Parts 1–4 using local Ollama by default. Pass `--mock`
to select deterministic `MOCK_LLM=true` for offline acceptance runs. The
complete output is saved to `transcripts/full_execution_transcript.txt`.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
TRANSCRIPT_PATH = PROJECT_ROOT / "transcripts" / "full_execution_transcript.txt"
STATUS_UPDATE_INTERVAL_SECONDS = 60


def configure_runtime(argv: Optional[list[str]] = None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic mock LLM instead of the configured Ollama model.",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    if args.mock:
        os.environ["MOCK_LLM"] = "true"
    else:
        os.environ.setdefault("MOCK_LLM", "false")
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    return os.environ["MOCK_LLM"].strip().lower() in {"1", "true", "yes", "on"}


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    overall_start_at = datetime.now(timezone.utc)
    overall_started = perf_counter()
    logger.info(
        "overall run started_at=%s",
        overall_start_at.isoformat(timespec="microseconds"),
    )
    configure_runtime(argv)
    import _offline_compat  # noqa: F401

    from app import verify_fastapi_and_logging
    from dataset import print_dataset_report
    from evaluation import print_part3_evaluation_report
    from governance_and_review import print_part4_governance_report
    from knowledge_base import validate_knowledge_base
    from rag_core import print_part1_rag_report
    from tools_and_crew import print_part2_crew_report

    TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("HYPOTHETICAL RETAILER DOMAIN SUPPORT AGENT — COMPLETE GRADED EXECUTION TRANSCRIPT")
        print("Track: E-commerce & Retail (hypothetical retailer) | Domain: E-commerce")
        print(
            f"Environment: MOCK_LLM={os.environ['MOCK_LLM']} | "
            f"OLLAMA_MODEL={os.getenv('OLLAMA_MODEL', 'gemma4:latest')} | "
            f"CREWAI_DISABLE_TELEMETRY={os.environ['CREWAI_DISABLE_TELEMETRY']} | "
            f"OTEL_SDK_DISABLED={os.environ['OTEL_SDK_DISABLED']}"
        )
        print()
        print("RUN TIMING (timestamps are UTC; durations use a monotonic clock)")
        print(f"  Overall start: {overall_start_at.isoformat(timespec='microseconds')}")

        steps: list[tuple[str, Callable[[], None]]] = [
            ("Knowledge-base validation", validate_knowledge_base),
            ("Dataset validation report", print_dataset_report),
            ("Part 1 — RAG, calibration, and chunking evaluation", print_part1_rag_report),
            ("Part 2 — CrewAI, memory, schema, and guardrails", print_part2_crew_report),
            ("Part 3 — FastAPI, WebSocket, and JSONL logging", verify_fastapi_and_logging),
            ("Part 3 — End-to-end evaluation", print_part3_evaluation_report),
            ("Part 4 — Governance, review, and caching", print_part4_governance_report),
        ]
        current_step = "Preparing verification stages"
        current_step_lock = threading.Lock()
        status_stop = threading.Event()

        def log_running_status(step_name: str) -> None:
            elapsed = int(perf_counter() - overall_started)
            elapsed_minutes, elapsed_seconds = divmod(elapsed, 60)
            logger.info(
                "status=running current_step=%r elapsed=%dm %02ds (%.1f seconds)",
                step_name,
                elapsed_minutes,
                elapsed_seconds,
                perf_counter() - overall_started,
            )

        def log_periodic_status() -> None:
            while not status_stop.wait(STATUS_UPDATE_INTERVAL_SECONDS):
                with current_step_lock:
                    step_name = current_step
                log_running_status(step_name)

        status_thread = threading.Thread(
            target=log_periodic_status,
            name="demo-status-updater",
            daemon=True,
        )
        status_thread.start()
        log_running_status(current_step)

        try:
            for step_name, step in steps:
                with current_step_lock:
                    current_step = step_name
                log_running_status(step_name)
                step_start_at = datetime.now(timezone.utc)
                step_started = perf_counter()
                step_status = "failed"
                try:
                    step()
                    step_status = "completed"
                finally:
                    step_ended = perf_counter()
                    step_end_at = datetime.now(timezone.utc)
                    step_duration = step_ended - step_started
                    logger.info(
                        "stage=%r status=%s start_at=%s end_at=%s time_taken_seconds=%.6f",
                        step_name,
                        step_status,
                        step_start_at.isoformat(timespec="microseconds"),
                        step_end_at.isoformat(timespec="microseconds"),
                        step_duration,
                    )
                print(f"\nSTEP TIMING — {step_name}")
                print(f"  Start: {step_start_at.isoformat(timespec='microseconds')}")
                print(f"  End:   {step_end_at.isoformat(timespec='microseconds')}")
                print(f"  Time taken: {step_duration:.6f} seconds")
                print()
        finally:
            status_stop.set()
            status_thread.join()

        mode = "MOCK_LLM" if os.environ["MOCK_LLM"].lower() == "true" else "Ollama"
        print(f"\nALL 16 TASKS & ACCEPTANCE CRITERIA COMPLETED UNDER {mode}.")
        overall_ended = perf_counter()
        overall_end_at = datetime.now(timezone.utc)
        overall_duration = overall_ended - overall_started
        logger.info(
            "overall run status=completed start_at=%s end_at=%s time_taken_seconds=%.6f",
            overall_start_at.isoformat(timespec="microseconds"),
            overall_end_at.isoformat(timespec="microseconds"),
            overall_duration,
        )
        print("\nOVERALL RUN TIMING")
        print(f"  Overall start: {overall_start_at.isoformat(timespec='microseconds')}")
        print(f"  Overall end:   {overall_end_at.isoformat(timespec='microseconds')}")
        print(f"  Time taken: {overall_duration:.6f} seconds")

    transcript_text = buf.getvalue()
    TRANSCRIPT_PATH.write_text(transcript_text, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(transcript_text)


if __name__ == "__main__":
    main()
