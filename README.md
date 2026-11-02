# Retail Customer-Support Agent — Educational Demonstration

> **Unofficial project disclaimer:** This is an independent educational software demonstration. It is not affiliated with, endorsed by, sponsored by, or operated by Nykaa, FSN E-Commerce Ventures Limited, or any other retailer. The retailer name was used only to identify the academic domain selected for the exercise.
>
> **Synthetic content notice:** All orders, order IDs, customers, interactions, policy examples, refund amounts, delivery timelines, warranty statements, loyalty rules, and escalation procedures in this repository are fictional test fixtures. They are not actual or verified company policies and must never be presented to customers as real commitments.

## Project overview

A local-first demonstration of an AI-powered retail customer-support agent using Python, FastAPI, CrewAI-style orchestration, LangGraph, retrieval-augmented generation (RAG), ChromaDB, Pydantic, and Ollama. The application demonstrates policy retrieval, synthetic order lookup, response grounding, input checks, structured output, and evaluation.

## Data and policy boundaries

- `dataset.py` generates 50 synthetic order records with a fixed random seed.
- `knowledge_base.py` contains 12 **fictional illustrative policy examples**. Each record is explicitly labelled as an educational example.
- Do not replace these fixtures with real customer records or proprietary company policy documents unless you have documented permission and an appropriate data-protection basis.
- Do not use this demo as a real customer-support channel or rely on its policy answers for actual purchases, returns, refunds, warranties, or grievances.

## Architecture and files

| File | Purpose |
|---|---|
| `app.py` | FastAPI demonstration API and WebSocket endpoint |
| `dataset.py` | Synthetic order generator |
| `knowledge_base.py` | Fictional policy examples |
| `rag_core.py` | Chunking, embeddings, retrieval, calibration, and cache |
| `tools_and_crew.py` | Agent orchestration, local Ollama adapter, order/RAG tools, and guardrails |
| `evaluation.py` | Controlled demonstration evaluation cases |
| `governance_and_review.py` | Demonstration governance and review report |
| `_offline_compat.py` | Optional local compatibility implementations |
| `run_all_demos.py` | Runs the verification/demo sequence |
| `PROJECT_GUIDE.md` | Architecture, API, and operating guide |
| `requirements.txt` | Python dependencies |

## Run locally

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The application uses local Ollama by default. Install and start Ollama separately, then ensure the configured model is available. For deterministic offline demonstrations, set `MOCK_LLM=true` or run:

```bash
python run_all_demos.py --mock
```

Start the API locally only:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

The API is a demonstration and does not include production authentication, authorization, durable storage, rate limiting, or multi-tenant isolation. **Do not expose it to the public internet or connect it to real customer data without implementing and independently reviewing those controls.** In particular, protect `/add-document` and the WebSocket endpoint with authentication and authorization before any non-local deployment.

## Privacy and security notes

- Use synthetic test data only. Never commit `.env`, access tokens, API keys, passwords, private keys, customer records, logs, or generated transcripts.
- PII masking is pattern-based and incomplete; it is not a guarantee of anonymization.
- Prompt-injection detection is a limited pattern check, not a general security boundary.
- Bind the demo to `127.0.0.1` unless a security review has approved a protected deployment.
- Review every dependency, its license, provenance, and version before deployment. Dependencies retain their own licenses; this repository does not relicense them.
- No license is granted to third-party trademarks, logos, product names, datasets, or proprietary materials.

## Trademark and affiliation

Any third-party company or product name is used only for nominative identification of the example domain. No logo, brand artwork, or claim of official status is included or intended. All sample policy content is independently authored for this demonstration and should not be interpreted as a reproduction of actual retailer terms.

## Limitations

This project is a capstone/demo implementation, not a production customer-support system. Synthetic evaluation scores measure behavior against controlled fixtures and should not be interpreted as independent production-quality validation.
