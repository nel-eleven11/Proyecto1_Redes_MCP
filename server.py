# server.py
from __future__ import annotations

# FastMCP server
from mcp.server.fastmcp import FastMCP

# Data & analytics stack
from statsbombpy import sb
import pandas as pd
import numpy as np
import socceraction.spadl as spadl
import socceraction.xthreat as xthreat

# Instantiate the FastMCP server
mcp = FastMCP("SoccerBestPlayMCP")

# Simple in-process caches to avoid refetching during one session
_COMP_CACHE: pd.DataFrame | None = None
_MATCH_ROW_CACHE: dict[int, pd.Series] = {}
_ACTIONS_CACHE: dict[int, pd.DataFrame] = {}

# Load public xT model (12x8 grid). If the URL is unavailable, raise a clear error.
_XT_MODEL_URL = "https://karun.in/blog/data/open_xt_12x8_v1.json"
try:
    XT_MODEL = xthreat.load_model(_XT_MODEL_URL)
except Exception as e:
    # If you prefer, you can bake a local JSON of the xT grid and load from disk as a fallback.
    raise RuntimeError(
        "Failed to load xT model from URL. Check your internet connection "
        "or replace the URL with a local JSON file."
    ) from e


def _list_competitions() -> pd.DataFrame:
    """Return and cache the list of competitions (StatsBomb Open Data)."""
    global _COMP_CACHE
    if _COMP_CACHE is None:
        _COMP_CACHE = sb.competitions()
        if not isinstance(_COMP_CACHE, pd.DataFrame):
            raise RuntimeError("StatsBombPy returned invalid competitions payload.")
    return _COMP_CACHE


def _find_match_row(match_id: int) -> pd.Series:
    """
    Find the row for a given match_id across all open-data competitions/seasons.

    StatsBombPy's API requires (competition_id, season_id) to list matches.
    Since we are called with only match_id, we scan known competitions once and cache the hit.
    """
    if match_id in _MATCH_ROW_CACHE:
        return _MATCH_ROW_CACHE[match_id]

    comps = _list_competitions()
    for _, comp in comps.iterrows():
        comp_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception:
            continue  # skip this competition on transient/network issues

        if not isinstance(matches, pd.DataFrame) or matches.empty:
            continue

        hit = matches[matches["match_id"] == match_id]
        if not hit.empty:
            row = hit.iloc[0]
            _MATCH_ROW_CACHE[match_id] = row
            return row

    raise ValueError(
        f"match_id {match_id} not found in open-data catalog. "
        f"Pick a valid match_id from sb.matches(competition_id, season_id)."
    )


def _safe_name(df: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    """
    Utility to normalize id->name mapping from events frame.
    Some events rows may have NaNs in player/team; we drop those safely.
    """
    cols = [c for c in (id_col, name_col) if c in df.columns]
    if len(cols) < 2:
        # If the events frame doesn't have both columns, return empty mapping
        return pd.DataFrame(columns=[id_col, name_col]).astype({id_col: "Int64", name_col: "string"})
    mapping = (
        df[cols]
        .dropna()
        .drop_duplicates()
        .rename(columns={id_col: id_col, name_col: name_col})
    )
    # enforce dtypes for safer merges
    mapping[id_col] = mapping[id_col].astype("int64", errors="ignore")
    mapping[name_col] = mapping[name_col].astype("string")
    return mapping


def _load_actions_with_xt(match_id: int) -> pd.DataFrame:
    """
    Fetch events for match_id using statsbombpy, convert to SPADL actions,
    orient left-to-right, and compute per-action xT using the public grid model.
    The resulting DataFrame includes player/team names where possible.
    """
    if match_id in _ACTIONS_CACHE:
        return _ACTIONS_CACHE[match_id]

    # Fetch raw events
    events = sb.events(match_id=match_id)
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError(f"No events found for match_id={match_id}")

    # Resolve match row to get home_team (needed by SPADL orientation)
    mrow = _find_match_row(match_id)
    # Column names in matches typically include: 'home_team' and 'away_team'
    # Fall back gracefully if column naming differs
    if "home_team" not in mrow.index:
        raise RuntimeError("Expected 'home_team' in matches row, but it was not found.")
    home_team = int(mrow["home_team"])

    # Convert events -> SPADL actions for StatsBomb schema
    # This produces standard SPADL fields including player_id, team_id, type_id/name, result_name, etc.
    actions = spadl.statsbomb.convert_to_actions(events, home_team)
    actions = spadl.add_names(actions)

    # Add player/team names by merging from events where possible
    player_map = _safe_name(events, "player_id", "player")
    team_map = _safe_name(events, "team_id", "team")

    if not player_map.empty and "player_id" in actions.columns:
        actions = actions.merge(player_map.rename(columns={"player": "player_name"}), on="player_id", how="left")
    else:
        actions["player_name"] = pd.NA

    if not team_map.empty and "team_id" in actions.columns:
        actions = actions.merge(team_map.rename(columns={"team": "team_name"}), on="team_id", how="left")
    else:
        actions["team_name"] = pd.NA

    # Ensure we can compute a readable minute later
    # SPADL may include 'time_seconds'; if not, approximate from events timing
    if "time_seconds" not in actions.columns:
        # Best-effort approximation: use cumulative seconds within each period if available
        if "period_id" in actions.columns and "seconds" in actions.columns:
            # some pipelines produce 'seconds' as within-period seconds; convert to match clock
            # Standard halves are 45 minutes; this is a rough heuristic.
            actions["time_seconds"] = np.where(
                actions["period_id"] == 1, actions["seconds"],
                np.where(actions["period_id"] == 2, actions["seconds"] + 45 * 60, actions["seconds"])
            )
        else:
            actions["time_seconds"] = np.nan

    # Orient L->R before xT rating (xT is field-position dependent)
    actions_ltr = spadl.play_left_to_right(actions, home_team)

    # Compute action-level xT contribution (this is ΔxT per action)
    # The public model returns an array-like of the same length as actions_ltr
    xt_values = XT_MODEL.rate(actions_ltr)
    actions["xT"] = pd.Series(xt_values, index=actions.index)

    # Cache and return
    _ACTIONS_CACHE[match_id] = actions
    return actions


def _int_minute_from_seconds(sec: float | int | None) -> int | None:
    """Helper to render minutes from seconds (floor), guarding None/NaN."""
    if sec is None:
        return None
    try:
        if np.isnan(sec):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return int(float(sec) // 60)


@mcp.tool()
def best_play(match_id: int) -> dict:
    """
    Return the single best play (highest xT contribution) in the match.
    Output is a compact JSON with player, team, action type, minute, and xT value.
    """
    actions = _load_actions_with_xt(match_id)

    if actions["xT"].notna().any():
        idx = actions["xT"].idxmax()
        row = actions.loc[idx]
    else:
        # If xT is entirely NaN (shouldn't happen with valid data), bail gracefully
        return {"error": f"xT values are all NaN for match_id={match_id}"}

    minute = _int_minute_from_seconds(row.get("time_seconds"))
    return {
        "match_id": match_id,
        "player": (row.get("player_name") if pd.notna(row.get("player_name")) else None),
        "team": (row.get("team_name") if pd.notna(row.get("team_name")) else None),
        "action_type": (row.get("type_name") if "type_name" in actions.columns else None),
        "minute": minute,
        "xT": float(row["xT"]) if pd.notna(row["xT"]) else None,
        "spadl_index": int(idx),
    }


@mcp.tool()
def player_performance(match_id: int) -> list[dict]:
    """
    Aggregate per-player performance for the given match, using xT as the primary signal.
    Returns a list of player dicts with: player, team, total_xT, positive_actions, shots, goals, assists.
    """
    actions = _load_actions_with_xt(match_id)

    # Initialize basic aggregates
    grouped = actions.groupby("player_id", dropna=True)

    rows: list[dict] = []
    for pid, grp in grouped:
        # Most frequent team and player names (robust if there are NaNs)
        team_name = grp["team_name"].dropna().mode().iloc[0] if grp["team_name"].notna().any() else None
        player_name = grp["player_name"].dropna().mode().iloc[0] if grp["player_name"].notna().any() else None

        # Sum xT and count positive actions
        total_xt = float(grp["xT"].fillna(0.0).sum())
        positive_actions = int((grp["xT"].fillna(0.0) > 0).sum())

        # Shots and goals from SPADL semantics:
        # type_name == "shot" is a shot. A goal is typically a shot with a "success" result in SPADL.
        shots = int((grp.get("type_name") == "shot").sum()) if "type_name" in grp.columns else 0

        # Goals: prefer 'result_name' == 'success' for shot rows; else 0
        if "type_name" in grp.columns and "result_name" in grp.columns:
            goals = int(((grp["type_name"] == "shot") & (grp["result_name"] == "success")).sum())
        else:
            goals = 0

        # Simple assist heuristic:
        # Count an assist when a shot-success is immediately preceded by a pass/cross from same team.
        assists = 0
        if "type_name" in actions.columns and "result_name" in actions.columns:
            # find indices of goal shots in the full actions frame
            goal_rows = actions.index[(actions["type_name"] == "shot") & (actions["result_name"] == "success")]
            for gidx in goal_rows:
                prev_idx = gidx - 1
                if prev_idx in actions.index:
                    prev_row = actions.loc[prev_idx]
                    g_row = actions.loc[gidx]
                    if (
                        prev_row.get("team_id") == g_row.get("team_id")
                        and prev_row.get("type_name") in ("pass", "cross")
                        and int(prev_row.get("player_id")) == int(pid)
                    ):
                        assists += 1

        rows.append(
            {
                "player_id": int(pid),
                "player": player_name,
                "team": team_name,
                "total_xT": round(total_xt, 4),
                "positive_actions": positive_actions,
                "shots": shots,
                "goals": assists + (goals - assists),  # store goals as-is; keep assists separate
                "assists": assists,
            }
        )

    # Sort by total_xT desc for convenience
    rows.sort(key=lambda r: r["total_xT"], reverse=True)
    return rows


if __name__ == "__main__":
    
    mcp.run()
