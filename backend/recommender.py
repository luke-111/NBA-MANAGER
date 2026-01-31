import os
from collections import defaultdict
from typing import Dict, Any, List

import google.generativeai as genai

# Expect env var GOOGLE_API_KEY already set


def _aggregate_recent(meta_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    player_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    player_counts: Dict[str, int] = defaultdict(int)
    for meta in meta_list:
        player = meta["player"]
        player_counts[player] += 1
        for k in ("pts", "reb", "ast", "minutes"):
            player_stats[player][k] += float(meta.get(k, 0))
    for player, agg in player_stats.items():
        count = player_counts[player]
        for k in list(agg.keys()):
            agg[k] = round(agg[k] / max(count, 1), 2)
    return player_stats


def _build_prompt(opponent: str, team: str, season: str, player_stats, opponent_hits):
    lines = [
        f"You are an NBA assistant coach. Build a recommended starting 5 and rotation for {team} vs {opponent} in season {season}.",
        "Use only the provided stats; be concise.",
        "Players with opponent_history=True have prior games vs this opponent.",
        "Return JSON with keys: lineup (array of 5 player names), bench (array of 3-5 names), reasons (array of strings)."
    ]
    lines.append("\nPlayer recent form:")
    for name, stats in player_stats.items():
        lines.append(f"- {name}: min {stats.get('minutes',0)}, pts {stats.get('pts',0)}, reb {stats.get('reb',0)}, ast {stats.get('ast',0)}")
    if opponent_hits:
        lines.append("\nOpponent history samples:")
        for h in opponent_hits:
            meta = h["meta"]
            lines.append(
                f"- {meta['player']} vs {meta.get('opponent')} on {meta.get('game_date')}: {meta.get('minutes')} MIN, {meta.get('pts')} PTS"
            )
    return "\n".join(lines)


def recommend_with_llm(store, payload):
    if store.embeddings is None:
        return {"error": "Store empty. Run /ingest first."}

    team_docs = [m for m in store.meta if m.get("team") == payload.team and m.get("season") == payload.season]
    if not team_docs:
        return {"error": "No docs for team/season. Run /ingest."}

    player_stats = _aggregate_recent(team_docs)

    hits = store.search(
        query=f"rotation vs {payload.opponent}",
        k=payload.limit * 3,
        filters={"team": payload.team, "opponent": payload.opponent},
    )

    # Rule-based ranking (fallback + display) — same as earlier logic
    opponent_players = {h["meta"]["player"] for h in hits}
    ranked = sorted(
        player_stats.items(),
        key=lambda kv: (kv[0] not in opponent_players, -kv[1].get("minutes", 0), -kv[1].get("pts", 0)),
    )
    top_players = ranked[: payload.limit]
    suggested_lineup = [
        {
            "player": name,
            "avg_minutes": stats.get("minutes", 0),
            "avg_pts": stats.get("pts", 0),
            "avg_reb": stats.get("reb", 0),
            "avg_ast": stats.get("ast", 0),
            "opponent_history": name in opponent_players,
        }
        for name, stats in top_players
    ]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "error": "Missing GOOGLE_API_KEY env var for Gemini.",
            "suggested_lineup": suggested_lineup,
        }

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = _build_prompt(payload.opponent, payload.team, payload.season, player_stats, hits)
    resp = model.generate_content(prompt)

    return {
        "team": payload.team,
        "season": payload.season,
        "opponent": payload.opponent,
        "suggested_lineup": suggested_lineup,
        "llm_response": resp.text,
        "player_stats": player_stats,
    }
