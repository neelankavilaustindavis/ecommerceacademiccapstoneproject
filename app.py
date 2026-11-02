"""Local-only FastAPI wrapper for the educational retail support demonstration.

This API intentionally has no production authentication. Keep it bound to
127.0.0.1 and never expose it to an untrusted network.
"""
from __future__ import annotations
import json, logging, time, uuid
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from tools_and_crew import RetailSupportCrewOrchestrator, mask_pii

app = FastAPI(title="Retail Support Agent — Educational Demo", version="1.0.0")
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PATH = LOG_DIR / "requests.jsonl"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("retail_demo")
orchestrator = RetailSupportCrewOrchestrator()

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None

class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=100)
    topic: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=20, max_length=10000)


def write_log(endpoint: str, raw_text: str, status: str, latency_ms: float, session_id: Optional[str] = None) -> None:
    safe_text, _ = mask_pii(raw_text)
    entry = {"timestamp": time.time(), "trace_id": str(uuid.uuid4()), "endpoint": endpoint,
             "session_id": session_id, "request_text": safe_text, "status": status,
             "latency_ms": round(latency_ms, 3)}
    with LOG_FILE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "mode": "educational-demo", "data": "synthetic-only"}

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    result = orchestrator.run_crew(req.query)
    elapsed = (time.perf_counter() - started) * 1000
    write_log("POST /ask", req.query, "success", elapsed, req.session_id)
    return result.model_dump()

@app.post("/add-document")
def add_document(req: AddDocumentRequest) -> Dict[str, Any]:
    # Local demo endpoint only. Add authentication/authorization before deployment.
    result = orchestrator.rag_engine.add_document(req.doc_id, req.topic, req.title, req.text)
    write_log("POST /add-document", f"Add {req.doc_id}: {req.text}", "indexed", 0.0)
    return result

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                query = str(payload.get("query", ""))
                session_id = str(payload.get("session_id", session_id))
            except (ValueError, AttributeError):
                query = raw
            started = time.perf_counter()
            result = orchestrator.run_crew(query)
            elapsed = (time.perf_counter() - started) * 1000
            write_log("WS /ws/chat", query, "success", elapsed, session_id)
            await websocket.send_json({"session_id": session_id, **result.model_dump()})
    except WebSocketDisconnect:
        write_log("WS /ws/chat", "client disconnected", "disconnected", 0.0, session_id)

def verify_fastapi_and_logging() -> None:
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    print("FastAPI health endpoint verified.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
