# SoccerAnalysis MCP Server

A minimal **Model Context Protocol (MCP)** server that analyzes **StatsBomb Open Data** using the
[`socceraction`](https://socceraction.readthedocs.io/) toolkit. The server exposes two tools:

- **`best_play(match_id)`** – returns the single action with the highest Expected Threat (xT) in a match.
- **`player_performance(match_id)`** – aggregates per-player performance (total xT, positive actions, shots, goals, assists).

The server runs locally (STDIO) and is intended to be used by MCP-compatible clients (e.g., desktop assistants) or via the `mcp` CLI for local testing.

---

## Author

Nelson García Bravatti 22434

For Networks course

---

## Features

- Uses `StatsBombLoader` to fetch **open** competitions, games, players, teams, and events.
- Converts events to **SPADL** actions via `socceraction.spadl`.
- Computes **Expected Threat (xT)** per action with a public 12×8 xT model.
- Fast local development with **FastMCP** and the `mcp` CLI.
- Lightweight in-memory caching per match during a server session.

---

## Project Structure

```
.
├── server.py          # MCP server (FastMCP) with two tools
├── pyproject.toml     # Project metadata and dependencies (uv / PEP 621)
├── requirements.txt   # Equivalent dependency list for pip users
└── README.md
```

---

## Requirements

- **Python 3.11+**
- Internet access (to fetch StatsBomb Open Data and the public xT model)
- One of:
  - **[uv](https://github.com/astral-sh/uv)** (recommended)
  - or **pip** / virtualenv

---

## Installation

### Option A: Using `uv` (recommended)

```bash
# From the project root
uv sync
```

This creates/updates a virtual environment and installs all dependencies declared in `pyproject.toml`.

### Option B: Using `pip`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Server

### Quick dev run (with `uv`)

```bash
uv run mcp dev server.py
```

- This launches the server in **dev mode** via the `mcp` CLI.
- The CLI will print logs and allow a connected MCP client to discover the tools.

### Direct run (Python)

```bash
uv run python server.py
# or, if you used pip/venv:
python server.py
```

> The server speaks MCP over STDIO. Most users will prefer `uv run mcp dev server.py` because the `mcp` CLI
> handles environment and dependency resolution neatly for development.

---

## Using the Tools

Any MCP-compatible client can call the tools once connected. Below are example payloads to illustrate usage.

### `best_play`

**Request**
```json
{
  "method": "tools/call",
  "params": {
    "name": "best_play",
    "arguments": { "match_id": 3895302 }
  }
}
```

**Response (example)**
```json
{
  "player": "Player Name",
  "team": "Team Name",
  "action_type": "pass",
  "minute": 72,
  "xT_increase": 0.1457
}
```

### `player_performance`

**Request**
```json
{
  "method": "tools/call",
  "params": {
    "name": "player_performance",
    "arguments": { "match_id": 3895302 }
  }
}
```

**Response (example)**
```json
[
  {
    "player": "Player A",
    "team": "Home Team",
    "total_xT": 0.4892,
    "positive_actions": 12,
    "shots": 3,
    "goals": 1,
    "assists": 1
  },
  ...
]
```

---

## How to Find a Valid `match_id`

This server uses **StatsBomb Open Data**. To discover valid IDs:

```python
from socceraction.data.statsbomb import StatsBombLoader
SBL = StatsBombLoader()
comps = SBL.competitions()
# Pick a competition/season pair
for comp in comps.itertuples():
    games = SBL.games(comp.competition_id, comp.season_id)
    print(games[["game_id", "home_team_id", "away_team_id"]].head())
    break  # inspect and choose a game_id from this table
```

Use any `game_id` shown as the `match_id` argument.

---

## License

This project uses StatsBomb **Open Data** for demonstration purposes. Please review and respect the
StatsBomb Open Data license and terms of use.

License at:

[`statsBomb Open-data`](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf)
