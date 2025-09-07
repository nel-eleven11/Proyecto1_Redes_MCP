# server.py
from mcp.server.fastmcp import FastMCP
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.xthreat as xthreat

# Initialize MCP server with FastMCP
mcp = FastMCP("SoccerAnalysis")  # Descriptive server name

# Load xT (Expected Threat) model at server startup.
# We use a public pre-trained model (12x8 grid).
xT_model_url = "https://karun.in/blog/data/open_xt_12x8_v1.json"
xT_model = xthreat.load_model(xT_model_url)

# Initialize StatsBomb data loader
SBL = StatsBombLoader()  # StatsBomb Open Data loader

# (Optional) In-memory cache for analyzed matches, for efficiency
_match_cache = {}


def _get_match_info(match_id: int):
    for comp in SBL.competitions().itertuples():
        games = SBL.games(comp.competition_id, comp.season_id)
        if match_id in games["game_id"].values:
            return games.set_index("game_id").loc[match_id]
    raise ValueError(f"Match {match_id} not found in StatsBomb open data.")


def _load_match_data(match_id: int):
    """Internal helper to load and process match data."""
    # Return cached result if already processed
    if match_id in _match_cache:
        return _match_cache[match_id]
    # Fetch events, players, and teams for the match
    df_events = SBL.events(match_id)
    df_players = SBL.players(match_id)
    df_teams = SBL.teams(match_id)
    # Get home team ID for field orientation (SPADL expects home/away)
    game_row = _get_match_info(match_id)
    home_team_id = int(game_row["home_team_id"])
    # Convert events to SPADL actions
    df_actions = spadl.statsbomb.convert_to_actions(df_events, home_team_id)
    df_actions = spadl.add_names(df_actions)               # Add descriptive names
    df_actions = df_actions.merge(df_players, how="left").merge(df_teams, how="left")
    # Compute xT for each action: orient actions left-to-right and apply the model
    df_actions_ltr = spadl.play_left_to_right(df_actions, home_team_id)
    df_actions["xT_value"] = xT_model.rate(df_actions_ltr)  # xT value per action
    # Cache and return
    _match_cache[match_id] = (df_actions, df_players, df_teams)
    return _match_cache[match_id]

@mcp.tool()
def best_play(match_id: int) -> dict:
    """Identify the action with the highest expected threat (xT) increase for the specified match."""
    # Load and prepare match data
    df_actions, df_players, df_teams = _load_match_data(match_id)
    if df_actions.empty:
        return {"error": "No se encontraron acciones para el partido ID %d" % match_id}
    # Find the action with maximum xT value
    idx_max = df_actions["xT_value"].idxmax()
    best_action = df_actions.loc[idx_max]
    player_name = best_action["player_name"]
    team_name = best_action["team_name"]
    action_type = best_action["type_name"]
    xT_increase = float(best_action["xT_value"]) if best_action["xT_value"] == best_action["xT_value"] else 0.0  # handle NaN
    # Compute (approximate) match minute from time_seconds
    minute = int(best_action["time_seconds"] // 60)
    # Return structured information of the best play
    return {
        "player": player_name,
        "team": team_name,
        "action_type": action_type,
        "minute": minute,
        "xT_increase": round(xT_increase, 4)
    }

@mcp.tool()
def player_performance(match_id: int) -> list:
    """Compute each player's performance in the match: total xT and key actions (important passes/shots)."""
    df_actions, df_players, df_teams = _load_match_data(match_id)
    if df_actions.empty:
        return []  # empty match or not found
    # Initialize per-player accumulator dict
    performance = {}  # key: player_id, value: stats dict
    for _, action in df_actions.iterrows():
        pid = action["player_id"]
        if pid not in performance:
            performance[pid] = {
                "player": action["player_name"],
                "team": action["team_name"],
                "total_xT": 0.0,
                "positive_actions": 0,   # actions with xT > 0
                "shots": 0,
                "goals": 0,
                "assists": 0
            }
        val = action["xT_value"]
        # Accumulate xT (if not NaN)
        if val == val:  # NaN check
            performance[pid]["total_xT"] += float(val)
            if val > 0:
                performance[pid]["positive_actions"] += 1
        # Count shots and goals
        if action["type_name"] == "shot":
            performance[pid]["shots"] += 1
            # If the shot was a goal (success)
            if action["result_name"] == "success":
                performance[pid]["goals"] += 1
                # Credit assist to the previous action's player if same team
                prev_idx = action.name - 1  # previous index
                if prev_idx in df_actions.index:
                    prev_action = df_actions.loc[prev_idx]
                    if prev_action["team_id"] == action["team_id"] and prev_action["type_name"] in ["pass", "cross"]:
                        prev_pid = prev_action["player_id"]
                        if prev_pid in performance:
                            performance[prev_pid]["assists"] += 1
    # Convert to a list of per-player stats
    players_stats = list(performance.values())
    # Optional: round total_xT to 4 decimals for readability
    for stats in players_stats:
        stats["total_xT"] = round(stats["total_xT"], 4)
    return players_stats

if __name__ == "__main__":
    # Run MCP server 
    mcp.run()
