# NBA Coach RAG — Minimal Demo

End-to-end: pull nba.com/stats data → build vector store → FastAPI serves rotation suggestions → simple frontend to query.

## Prereqs
- Python 3.10+
- Install deps:
  ```bash
  pip install -r backend/requirements.txt
  ```
- Run API:
  ```bash
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
  ```

## How to Use
1) Serve the frontend in another terminal:
   ```bash
   cd frontend && python -m http.server 3000
   ```
   Open http://localhost:3000

2) Set Gemini key (same terminal as uvicorn):
   ```bash
   export GOOGLE_API_KEY="<your_gemini_key>"
   # optional: silence tokenizers warning
   export TOKENIZERS_PARALLELISM=false
   ```

3) Run backend:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

4) In the page, fill team abbr (e.g., BOS), season (e.g., 2024-25), opponent abbr (e.g., LAL), then click:
   - “Ingest Data”: fetch roster + recent N games per player and build embeddings.
   - “Build Rotation”: calls Gemini 2.5 Flash with retrieved stats; returns LLM suggestion plus rule-based fallback lineup (rendered on page).

## API Quick Ref
- `POST /ingest` `{team, season, last}` — fetch and store.
- `POST /recommend` `{team, season, opponent, limit?}` — suggested rotation and sample games.
- `GET /health` — status.

## Notes
- Uses SentenceTransformer `all-MiniLM-L6-v2`; first run downloads the model.
- Data source: public nba.com/stats. Add more stats (PER, BPM, etc.) in `backend/ingest.py` and update `format_game_sentence` if desired.
- Ranking logic: prefer players with opponent samples, then by recent minutes and points. Tweak in `backend/main.py`.
