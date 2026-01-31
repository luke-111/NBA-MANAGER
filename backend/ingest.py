"""Fetch NBA stats and build local RAG store."""
import argparse
import datetime as dt
import os
from typing import List, Dict, Any

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.library.parameters import Season
from nba_api.stats.static import teams

from backend.rag import RAGStore, Doc, format_game_sentence


def resolve_team_id(abbr: str) -> int:
    team = teams.find_team_by_abbreviation(abbr.upper())
    if not team:
        raise ValueError(f"Unknown team abbreviation: {abbr}")
    return team["id"]


def fetch_roster(team_abbr: str, season: str) -> List[Dict[str, Any]]:
    team_id = resolve_team_id(team_abbr)
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]
    return df.to_dict("records")


def fetch_player_games(player_id: int, season: str, last_n: int = 12) -> List[Dict[str, Any]]:
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star="Regular Season")
    df = gl.get_data_frames()[0]
    # keep most recent games
    df = df.sort_values("GAME_DATE", ascending=False).head(last_n)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "game_date": row["GAME_DATE"],
                "opponent": row["MATCHUP"].split(" ")[-1],
                "minutes": float(row["MIN"]),
                "pts": float(row["PTS"]),
                "reb": float(row["REB"]),
                "ast": float(row["AST"]),
                "stl": float(row["STL"]),
                "blk": float(row["BLK"]),
                "fgm": float(row["FGM"]),
                "fga": float(row["FGA"]),
                "fg3m": float(row["FG3M"]),
                "fg3a": float(row["FG3A"]),
                "ftm": float(row["FTM"]),
                "fta": float(row["FTA"]),
            }
        )
    return records


def build_docs(team_abbr: str, season: str, last_n: int = 12) -> List[Doc]:
    roster = fetch_roster(team_abbr, season)
    docs: List[Doc] = []
    for player in roster:
        pid = int(player["PLAYER_ID"])
        games = fetch_player_games(pid, season, last_n=last_n)
        for game in games:
            meta = {
                "player_id": pid,
                "player": player["PLAYER"],
                "team": team_abbr,
                "season": season,
                "position": player.get("POSITION", ""),
                "opponent": game["opponent"],
                "game_date": game["game_date"],
                "context_type": "game_log",
                "minutes": game["minutes"],
                "pts": game["pts"],
                "reb": game["reb"],
                "ast": game["ast"],
            }
            docs.append(Doc(text=format_game_sentence(game), meta=meta))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest team data into FAISS store")
    parser.add_argument("team", help="Team abbreviation, e.g., BOS")
    parser.add_argument("season", help="Season format YYYY-YY, e.g., 2024-25", default=Season.current_season, nargs="?")
    parser.add_argument("--last", type=int, default=12, help="Number of recent games per player")
    args = parser.parse_args()

    store = RAGStore()
    docs = build_docs(args.team, args.season, last_n=args.last)
    store.add_documents(docs)
    print(f"Ingested {len(docs)} documents for team {args.team} {args.season}")


if __name__ == "__main__":
    main()
