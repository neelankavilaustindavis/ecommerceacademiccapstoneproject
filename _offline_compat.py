"""Compatibility helpers for offline/local embedding and vector-store fallback.

CrewAI fallback orchestration is intentionally not emulated here. Install the
pinned CrewAI and LangGraph dependencies from requirements.txt for the full
multi-agent workflow. This avoids silently substituting a partial fake agent
runtime for the real framework.
"""
from __future__ import annotations
import hashlib
import re
from typing import Any, Iterable, List
import numpy as np

class LocalSentenceTransformer:
    """Small deterministic hashing embedder; not a semantic production model."""
    def __init__(self, model_name_or_path: str = "local-hash-demo", **kwargs: Any) -> None:
        self.model_name_or_path = model_name_or_path
    def encode(self, sentences: Iterable[str], normalize_embeddings: bool = True, **kwargs: Any) -> np.ndarray:
        rows=[]
        for sentence in sentences:
            vec=np.zeros(384,dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", str(sentence).lower()):
                idx=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%384
                vec[idx]+=1.0
            norm=np.linalg.norm(vec)
            if normalize_embeddings and norm: vec/=norm
            rows.append(vec)
        return np.asarray(rows,dtype=np.float32)

class _Collection:
    def __init__(self) -> None: self.rows={}
    def upsert(self, ids, embeddings, documents, metadatas):
        for i,e,d,m in zip(ids,embeddings,documents,metadatas): self.rows[i]=(np.asarray(e),d,m)
    def count(self): return len(self.rows)
    def query(self, query_embeddings, n_results=3):
        q=np.asarray(query_embeddings[0]); scored=[]
        for key,(e,d,m) in self.rows.items():
            dist=1-float(np.dot(q,e)/(max(np.linalg.norm(q)*np.linalg.norm(e),1e-12)))
            scored.append((dist,key,d,m))
        scored.sort(key=lambda x:x[0]); scored=scored[:n_results]
        return {"ids":[[x[1] for x in scored]],"documents":[[x[2] for x in scored]],
                "metadatas":[[x[3] for x in scored]],"distances":[[x[0] for x in scored]]}

class LocalChromaClient:
    def __init__(self): self.collections={}
    def get_or_create_collection(self, name, metadata=None):
        return self.collections.setdefault(name,_Collection())

# Minimal import-time compatibility types. The actual agent runtime is not
# simulated: install CrewAI to execute the multi-agent workflow.
try:
    from pydantic import BaseModel
    class BaseTool(BaseModel):
        name: str = "tool"
        description: str = ""
        args_schema: Any = None
        class Config:
            arbitrary_types_allowed = True
        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Install CrewAI to invoke agent tools in this demo.")
    class BaseLLM:
        def __init__(self, model: str = "local-demo", temperature: float = 0.0, **kwargs: Any):
            self.model = model
            self.temperature = temperature
    class Agent:
        def __init__(self, **kwargs: Any): self.__dict__.update(kwargs)
    class Task:
        def __init__(self, **kwargs: Any): self.__dict__.update(kwargs)
    class Crew:
        def __init__(self, **kwargs: Any): self.__dict__.update(kwargs)
        def kickoff(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("CrewAI is required for the multi-agent workflow. Install requirements.txt.")
except ImportError:
    pass
