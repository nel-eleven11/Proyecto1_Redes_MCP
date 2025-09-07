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
from socceraction.spadl import actiontypes_df, results_df, bodyparts_df, config as spadlconfig
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # silence fidelity_version message

# Instantiate the FastMCP server
mcp = FastMCP("SoccerBestPlayMCP")

# Simple in-process caches
_COMP_CACHE: pd.DataFrame | None = None
_MATCH_ROW_CACHE: dict[int, pd.Series] = {}
_ACTIONS_CACHE: dict[int, pd.DataFrame] = {}

# Public xT model
_XT_MODEL_URL = "https://karun.in/blog/data/open_xt_12x8_v1.json"
try:
    XT_MODEL = xthreat.load_model(_XT_MODEL_URL)
except Exception as e:
    raise RuntimeError(
        "Failed to load xT model from URL. Check your internet connection "
        "or replace the URL with a local JSON file."
    ) from e


def _list_competitions() -> pd.DataFrame:
    """Return and cache StatsBomb open-data competitions."""
    global _COMP_CACHE
    if _COMP_CACHE is None:
        _COMP_CACHE = sb.competitions()
        if not isinstance(_COMP_CACHE, pd.DataFrame):
            raise RuntimeError("StatsBombPy returned invalid competitions payload.")
    return _COMP_CACHE


def _find_match_row(match_id: int) -> pd.Series:
    """Find the sb.matches() row for match_id by scanning all open competitions/seasons."""
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


def _safe_name(df: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
    """Create id->name mapping, dropping NaNs safely."""
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


def _normalize_events_for_spadl(events: pd.DataFrame) -> pd.DataFrame:
    """Ensure StatsBomb 'events' have the columns SPADL conversion expects."""
    ev = events.copy()

    # 1) Ensure type_name exists
    if "type_name" not in ev.columns:
        def _tname(x): return x.get("name") if isinstance(x, dict) else x
        ev["type_name"] = ev["type"].apply(_tname).astype("string")

    # 2) Ensure 'extra' column exists
    if "extra" not in ev.columns:
        ev["extra"] = [{} for _ in range(len(ev))]
    else:
        ev["extra"] = ev["extra"].apply(lambda x: x if isinstance(x, dict) else {})

    # 3) Ensure match_id column exists
    if "match_id" not in ev.columns:
        ev["match_id"] = pd.NA

    return ev


def _ensure_game_id(actions: pd.DataFrame, match_id: int, events: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a numeric 'game_id' column on actions."""
    a = actions.copy()
    if "game_id" not in a.columns:
        gid = None
        if "match_id" in events.columns and events["match_id"].notna().any():
            try:
                gid = int(pd.Series(events["match_id"]).dropna().iloc[0])
            except Exception:
                gid = None
        if gid is None:
            gid = int(match_id)
        a["game_id"] = int(gid)
    try:
        a["game_id"] = a["game_id"].astype("int64")
    except Exception:
        pass
    return a


def _ensure_action_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable names for type/bodypart/result if missing."""
    a = actions.copy()
    if "type_name" not in a.columns and "type_id" in a.columns:
        a = a.merge(actiontypes_df[["type_id", "type_name"]], on="type_id", how="left")
    if "bodypart_name" not in a.columns and "bodypart_id" in a.columns:
        a = a.merge(bodyparts_df[["bodypart_id", "bodypart_name"]], on="bodypart_id", how="left")
    if "result_name" not in a.columns and "result_id" in a.columns:
        a = a.merge(results_df[["result_id", "result_name"]], on="result_id", how="left")
    return a


def _orient_left_to_right_manual(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Flip X coordinates for away-team actions so that the home team always attacks left->right."""
    a = actions.copy()
    if "team_id" not in a.columns or "start_x" not in a.columns or "end_x" not in a.columns:
        return a
    flen = float(spadlconfig.field_length)
    mask_away = a["team_id"].astype("Int64") != int(home_team_id)
    a.loc[mask_away, "start_x"] = flen - a.loc[mask_away, "start_x"].astype(float)
    a.loc[mask_away, "end_x"]   = flen - a.loc[mask_away, "end_x"].astype(float)
    return a


def _load_actions_with_xt(match_id: int) -> pd.DataFrame:
    """Fetch events, normalize them, convert to actions, guarantee game_id, orient, compute xT."""
    if match_id in _ACTIONS_CACHE:
        return _ACTIONS_CACHE[match_id]

    # 1) Raw events
    events = sb.events(match_id=match_id)
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError(f"No events found for match_id={match_id}")

    # 2) Normalize events
    events = _normalize_events_for_spadl(events)

    # 3) Resolve home team id
    mrow = _find_match_row(match_id)
    home_team_name = mrow["home_team"]
    try:
        home_team_id = int(events.loc[events["team"] == home_team_name, "team_id"].iloc[0])
    except Exception:
        home_team_id = int(events["team_id"].dropna().mode().iloc[0])

    # 4) Convert to SPADL actions
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id)

    # 5) Guarantee game_id
    actions = _ensure_game_id(actions, match_id, events)

    # 6) Attach names
    try:
        actions = spadl.add_names(actions)
    except Exception:
        pass
    actions = _ensure_action_names(actions)

    # 7) Ensure time_seconds
    if "time_seconds" not in actions.columns:
        if {"period_id", "seconds"}.issubset(actions.columns):
            actions["time_seconds"] = np.where(
                actions["period_id"] == 1, actions["seconds"],
                np.where(actions["period_id"] == 2, actions["seconds"] + 45 * 60, actions["seconds"])
            )
        else:
            actions["time_seconds"] = np.nan

    # 8) Manual orientation
    actions_ltr = _orient_left_to_right_manual(actions, home_team_id)

    # 9) Compute xT
    xt_values = XT_MODEL.rate(actions_ltr)
    actions["xT"] = pd.Series(xt_values, index=actions.index)

    # 10) Attach team/player names
    def _safe_map(df: pd.DataFrame, id_col: str, name_col: str) -> pd.DataFrame:
        cols = [c for c in (id_col, name_col) if c in df.columns]
        if len(cols) < 2:
            return pd.DataFrame(columns=[id_col, name_col])
        return df[cols].dropna().drop_duplicates()

    pmap = _safe_map(events, "player_id", "player")
    tmap = _safe_map(events, "team_id", "team")
    if not pmap.empty and "player_id" in actions.columns:
        actions = actions.merge(pmap.rename(columns={"player": "player_name"}), on="player_id", how="left")
    else:
        actions["player_name"] = pd.NA
    if not tmap.empty and "team_id" in actions.columns:
        actions = actions.merge(tmap.rename(columns={"team": "team_name"}), on="team_id", how="left")
    else:
        actions["team_name"] = pd.NA

    _ACTIONS_CACHE[match_id] = actions
    return actions


def _int_minute_from_seconds(sec: float | int | None) -> int | None:
    """Convert seconds to integer minute (floor)."""
    if sec is None:
        return None
    try:
        if np.isnan(sec):
            return None
    except Exception:
        pass
    return int(float(sec) // 60)


@mcp.tool()
def best_play(match_id: int) -> dict:
    """Return the single best play (highest ΔxT) in the match."""
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
    """Aggregate per-player performance using ΔxT as the main signal."""
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
