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
    """Find the row for a given match_id across all open-data competitions/seasons."""
    if match_id in _MATCH_ROW_CACHE:
        return _MATCH_ROW_CACHE[match_id]

    comps = _list_competitions()
    for _, comp in comps.iterrows():
        comp_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception:
            continue
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


def _infer_home_team_id_from_events(match_row: pd.Series, events: pd.DataFrame) -> int:
    """
    Infer numeric home team id when matches row doesn't include 'home_team_id'.
    We map the home team NAME from sb.matches() to the (team, team_id) pairs found in events.
    """
    home_name = None
    for candidate in ("home_team", "home_team_name"):
        if candidate in match_row.index and pd.notna(match_row[candidate]):
            home_name = str(match_row[candidate])
            break
    if not home_name:
        if "home_team" in match_row.index and isinstance(match_row["home_team"], dict):
            home_name = match_row["home_team"].get("home_team_name")
    if not home_name:
        raise RuntimeError("Could not resolve home team name from matches row.")

    if "team" not in events.columns or "team_id" not in events.columns:
        raise RuntimeError("Events dataframe lacks 'team'/'team_id' to infer home team id.")

    pairs = events[["team", "team_id"]].dropna().drop_duplicates()
    hit = pairs[pairs["team"].astype(str) == home_name]
    if not hit.empty:
        return int(hit.iloc[0]["team_id"])
    home_norm = home_name.strip().lower()
    hit = pairs[pairs["team"].astype(str).str.strip().str.lower() == home_norm]
    if not hit.empty:
        return int(hit.iloc[0]["team_id"])

    raise RuntimeError(
        f"Could not infer home team id from events. Home name '{home_name}' not found among event teams."
    )


def _get_home_team_id(match_row: pd.Series, events: pd.DataFrame) -> int:
    """
    Robust home team id resolver:
    1) Use 'home_team_id' if present (flattened).
    2) Use nested dict in 'home_team' if present.
    3) Otherwise infer from events mapping (team name -> team_id).
    """
    if "home_team_id" in match_row.index and pd.notna(match_row["home_team_id"]):
        return int(match_row["home_team_id"])
    if "home_team" in match_row.index and isinstance(match_row["home_team"], dict):
        ht = match_row["home_team"]
        if "home_team_id" in ht:
            return int(ht["home_team_id"])
    if "home_team" in match_row.index and isinstance(match_row["home_team"], (int, np.integer)):
        return int(match_row["home_team"])
    return _infer_home_team_id_from_events(match_row, events)


def _safe_name(df: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    """Normalize id->name mapping from events, dropping NaNs safely."""
    if id_col not in df.columns or name_col not in df.columns:
        out = pd.DataFrame(columns=[id_col, name_col])
        return out.astype({id_col: "Int64", name_col: "string"})
    mapping = df[[id_col, name_col]].dropna().drop_duplicates()
    try:
        mapping[id_col] = mapping[id_col].astype("Int64")
    except Exception:
        pass
    try:
        mapping[name_col] = mapping[name_col].astype("string")
    except Exception:
        pass
    return mapping


def _ensure_events_type_name(events: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'type_name' column exists in the StatsBomb events dataframe.
    Some statsbombpy versions provide only 'type' (string or dict); SPADL expects 'type_name'.
    """
    if "type_name" in events.columns:
        return events
    if "type" not in events.columns:
        raise RuntimeError("Events dataframe has neither 'type_name' nor 'type' column.")

    ev = events.copy()

    def _extract_type_name(x):
        # x can be a dict like {"id": 30, "name": "Pass"} or already a string like "Pass"
        if isinstance(x, dict):
            return x.get("name")
        return x

    ev["type_name"] = ev["type"].apply(_extract_type_name).astype("string")
    return ev


def _ensure_action_names(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'type_name', 'result_name', 'bodypart_name' exist by joining
    SPADL lookup tables on *_id columns when necessary.
    """
    a = actions.copy()
    at = spadl.actiontypes_df.copy()
    rt = spadl.results_df.copy()
    bt = spadl.bodyparts_df.copy()

    for col in ("type_id", "result_id", "bodypart_id"):
        if col in a.columns:
            try:
                a[col] = a[col].astype("Int64")
            except Exception:
                pass

    if "type_name" not in a.columns and "type_id" in a.columns:
        a = a.merge(at[["type_id", "type_name"]], on="type_id", how="left")
    if "result_name" not in a.columns and "result_id" in a.columns:
        a = a.merge(rt[["result_id", "result_name"]], on="result_id", how="left")
    if "bodypart_name" not in a.columns and "bodypart_id" in a.columns:
        a = a.merge(bt[["bodypart_id", "bodypart_name"]], on="bodypart_id", how="left")

    return a


def _load_actions_with_xt(match_id: int) -> pd.DataFrame:
    """
    Fetch events for match_id, normalize columns, convert to SPADL actions,
    orient left-to-right, and compute per-action xT.
    """
    if match_id in _ACTIONS_CACHE:
        return _ACTIONS_CACHE[match_id]

    # Raw events
    events = sb.events(match_id=match_id)
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError(f"No events found for match_id={match_id}")

    # Make sure 'type_name' exists for SPADL
    events = _ensure_events_type_name(events)

    # Resolve home team id (needed to orient L->R)
    mrow = _find_match_row(match_id)
    home_team_id = _get_home_team_id(mrow, events)

    # Convert events -> SPADL actions
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id)

    # Try to add names via SPADL; if that fails, ensure via lookups
    try:
        actions = spadl.add_names(actions)
    except Exception:
        pass
    actions = _ensure_action_names(actions)

    # Add player/team names from events (if available)
    player_map = _safe_name(events, "player_id", "player")
    team_map = _safe_name(events, "team_id", "team")
    if "player_id" in actions.columns:
        if not player_map.empty:
            actions = actions.merge(
                player_map.rename(columns={"player": "player_name"}),
                on="player_id",
                how="left",
            )
        else:
            actions["player_name"] = pd.NA
    if "team_id" in actions.columns:
        if not team_map.empty:
            actions = actions.merge(
                team_map.rename(columns={"team": "team_name"}),
                on="team_id",
                how="left",
            )
        else:
            actions["team_name"] = pd.NA

    # Best-effort time in seconds for pretty minutes
    if "time_seconds" not in actions.columns:
        if "period_id" in actions.columns and "seconds" in actions.columns:
            actions["time_seconds"] = np.where(
                actions["period_id"] == 1,
                actions["seconds"],
                np.where(actions["period_id"] == 2, actions["seconds"] + 45 * 60, actions["seconds"]),
            )
        else:
            actions["time_seconds"] = np.nan

    # Orient L->R for xT
    actions_ltr = spadl.play_left_to_right(actions, home_team_id)

    # ΔxT per action
    xt_values = XT_MODEL.rate(actions_ltr)
    actions["xT"] = pd.Series(xt_values, index=actions.index)

    _ACTIONS_CACHE[match_id] = actions
    return actions


def _int_minute_from_seconds(sec: float | int | None) -> int | None:
    """Render minutes from seconds (floor), guarding None/NaN."""
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
    """
    actions = _load_actions_with_xt(match_id)

    if actions["xT"].notna().any():
        idx = actions["xT"].idxmax()
        row = actions.loc[idx]
    else:
        return {"error": f"xT values are all NaN for match_id={match_id}"}

    minute = _int_minute_from_seconds(row.get("time_seconds"))
    action_type = None
    if "type_name" in actions.columns and pd.notna(row.get("type_name")):
        action_type = str(row.get("type_name"))
    elif "type_id" in actions.columns and pd.notna(row.get("type_id")):
        action_type = f"type_{int(row.get('type_id'))}"

    return {
        "match_id": match_id,
        "player": (row.get("player_name") if pd.notna(row.get("player_name")) else None),
        "team": (row.get("team_name") if pd.notna(row.get("team_name")) else None),
        "action_type": action_type,
        "minute": minute,
        "xT": float(row["xT"]) if pd.notna(row["xT"]) else None,
        "spadl_index": int(idx),
    }


@mcp.tool()
def player_performance(match_id: int) -> list[dict]:
    """
    Aggregate per-player performance for the given match using xT as the primary signal.
    """
    actions = _load_actions_with_xt(match_id)
    grouped = actions.groupby("player_id", dropna=True)

    rows: list[dict] = []
    for pid, grp in grouped:
        team_name = grp["team_name"].dropna().mode().iloc[0] if grp["team_name"].notna().any() else None
        player_name = grp["player_name"].dropna().mode().iloc[0] if grp["player_name"].notna().any() else None

        total_xt = float(grp["xT"].fillna(0.0).sum())
        positive_actions = int((grp["xT"].fillna(0.0) > 0).sum())

        shots = int((grp.get("type_name") == "shot").sum()) if "type_name" in grp.columns else 0
        if "type_name" in grp.columns and "result_name" in grp.columns:
            goals = int(((grp["type_name"] == "shot") & (grp["result_name"] == "success")).sum())
        else:
            goals = 0

        # Simple assist heuristic: previous action same team is a pass/cross by this player.
        assists = 0
        if "type_name" in actions.columns and "result_name" in actions.columns:
            goal_rows = actions.index[(actions["type_name"] == "shot") & (actions["result_name"] == "success")]
            for gidx in goal_rows:
                prev_idx = gidx - 1
                if prev_idx in actions.index:
                    prev_row = actions.loc[prev_idx]
                    g_row = actions.loc[gidx]
                    prev_pid = prev_row.get("player_id")
                    if (
                        prev_row.get("team_id") == g_row.get("team_id")
                        and prev_row.get("type_name") in ("pass", "cross")
                        and pd.notna(prev_pid)
                        and int(prev_pid) == int(pid)
                    ):
                        assists += 1

        rows.append(
            {
                "player_id": int(pid) if pd.notna(pid) else None,
                "player": player_name,
                "team": team_name,
                "total_xT": round(total_xt, 4),
                "positive_actions": positive_actions,
                "shots": shots,
                "goals": goals,
                "assists": assists,
            }
        )

    rows.sort(key=lambda r: r["total_xT"], reverse=True)
    return rows


if __name__ == "__main__":
    mcp.run()
