import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@dataclass
class Doc:
    text: str
    meta: Dict[str, Any]


class RAGStore:
    """
    Simple in-process vector store using SentenceTransformer + numpy.
    Embeddings are saved to disk to avoid recompute on restart.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "backend/data"):
        self.model_name = model_name
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.model = SentenceTransformer(self.model_name)
        self.emb_path = os.path.join(self.data_dir, "embeddings.npy")
        self.meta_path = os.path.join(self.data_dir, "meta.json")
        self.embeddings: Optional[np.ndarray] = None
        self.meta: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.emb_path):
            self.embeddings = np.load(self.emb_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def _persist(self) -> None:
        if self.embeddings is not None:
            np.save(self.emb_path, self.embeddings)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add_documents(self, docs: List[Doc]) -> None:
        if not docs:
            return
        texts = [d.text for d in docs]
        new_emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # L2 normalize for cosine similarity
        new_emb = new_emb / np.clip(np.linalg.norm(new_emb, axis=1, keepdims=True), 1e-12, None)
        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])
        self.meta.extend([d.meta for d in docs])
        self._persist()

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.meta:
            return []
        filters = filters or {}
        candidates = [i for i, m in enumerate(self.meta) if all(m.get(k) == v for k, v in filters.items())]
        if not candidates:
            candidates = list(range(len(self.meta)))
        sub_meta = [self.meta[i] for i in candidates]
        sub_emb = self.embeddings[candidates]
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
        sims = cosine_similarity(q_emb, sub_emb)[0]
        top_idx = np.argsort(sims)[::-1][: min(k, len(sub_meta))]
        results = []
        for idx in top_idx:
            results.append({"score": float(sims[idx]), "meta": sub_meta[idx]})
        return results


def format_game_sentence(game: Dict[str, Any]) -> str:
    """Convert a game log row into a compact natural sentence for embedding."""
    return (
        f"{game['game_date']} vs {game['opponent']}: {game['minutes']} MIN, "
        f"{game['pts']} PTS on {game['fgm']}/{game['fga']} FG, {game['reb']} REB, "
        f"{game['ast']} AST, {game['stl']} STL, {game['blk']} BLK; "
        f"3P {game['fg3m']}/{game['fg3a']}, FT {game['ftm']}/{game['fta']}"
    )
