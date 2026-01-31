from collections import defaultdict
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import RAGStore
from backend.ingest import build_docs
from backend.recommender import recommend_with_llm

app = FastAPI(title="NBA Coach RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
store = RAGStore()
DEFAULT_TEAM = "BOS"
DEFAULT_SEASON = "2024-25"


class IngestRequest(BaseModel):
    team: str
    season: str = DEFAULT_SEASON
    last: int = 12


class RecommendRequest(BaseModel):
    opponent: str
    team: str = DEFAULT_TEAM
    season: str = DEFAULT_SEASON
    limit: int = 5


@app.post("/ingest")
def ingest_team(payload: IngestRequest) -> Dict[str, Any]:
    docs = build_docs(payload.team, payload.season, last_n=payload.last)
    store.add_documents(docs)
    return {"status": "ok", "docs_added": len(docs)}


@app.post("/recommend")
def recommend_lineup(payload: RecommendRequest) -> Dict[str, Any]:
    return recommend_with_llm(store, payload)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "loaded_docs": str(len(store.meta))}
