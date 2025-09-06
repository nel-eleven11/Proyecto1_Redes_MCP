# server.py
from mcp.server.fastmcp import FastMCP
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.xthreat as xthreat

# Inicializar servidor MCP con FastMCP
mcp = FastMCP("SoccerAnalysis")  # Nombre descriptivo del servidor

# Cargar modelo xT (Expected Threat) global al iniciar el servidor.
# Usamos un modelo público pre-entrenado (grid 12x8):contentReference[oaicite:6]{index=6}.
xT_model_url = "https://karun.in/blog/data/open_xt_12x8_v1.json"
xT_model = xthreat.load_model(xT_model_url)

# Inicializar cargador de datos de StatsBomb
SBL = StatsBombLoader()  # Loader de datos públicos de StatsBomb:contentReference[oaicite:7]{index=7}

# (Opcional) Caché en memoria para datos de partidos analizados, para eficiencia
_match_cache = {}

def _load_match_data(match_id: int):
    """Función interna para cargar y procesar datos de un partido."""
    # Si ya fue procesado antes, retornar de la caché
    if match_id in _match_cache:
        return _match_cache[match_id]
    # Obtener eventos, jugadores y equipos del partido
    df_events = SBL.events(match_id)        # Eventos del partido:contentReference[oaicite:8]{index=8}
    df_players = SBL.players(match_id)      # Jugadores que participaron
    df_teams = SBL.teams(match_id)          # Equipos que jugaron
    # Obtener ID del equipo local para referencia de orientación del campo
    # (StatsBomb designa home/away; necesitamos home_team_id para SPADL)
    game = SBL.games(competition_id=None, season_id=None)  # all games loaded
    home_team_id = game.set_index("game_id").at[match_id, "home_team_id"]
    # Convertir eventos a acciones en formato SPADL:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}
    df_actions = spadl.statsbomb.convert_to_actions(df_events, home_team_id)
    df_actions = spadl.add_names(df_actions)               # Agregar nombres descriptivos
    df_actions = df_actions.merge(df_players, how="left").merge(df_teams, how="left")
    # Calcular xT de cada acción: orientamos acciones izquierda->derecha y aplicamos modelo:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}
    df_actions_ltr = spadl.play_left_to_right(df_actions, home_team_id)
    df_actions["xT_value"] = xT_model.rate(df_actions_ltr)  # Valor xT para cada acción
    # Almacenar en caché y retornar
    _match_cache[match_id] = (df_actions, df_players, df_teams)
    return _match_cache[match_id]

@mcp.tool()
def best_play(match_id: int) -> dict:
    """Identifica la jugada con mayor aumento de amenaza esperada (xT) en el partido especificado."""
    # Cargar y preparar datos del partido
    df_actions, df_players, df_teams = _load_match_data(match_id)
    if df_actions.empty:
        return {"error": "No se encontraron acciones para el partido ID %d" % match_id}
    # Encontrar la acción con valor xT máximo
    idx_max = df_actions["xT_value"].idxmax()
    best_action = df_actions.loc[idx_max]
    player_name = best_action["player_name"]
    team_name = best_action["team_name"]
    action_type = best_action["type_name"]
    xT_increase = float(best_action["xT_value"]) if best_action["xT_value"] == best_action["xT_value"] else 0.0  # handle NaN
    # Calcular minuto de juego (aproximado) a partir de time_seconds
    minute = int(best_action["time_seconds"] // 60)
    # Devolver información estructurada de la mejor jugada
    return {
        "player": player_name,
        "team": team_name,
        "action_type": action_type,
        "minute": minute,
        "xT_increase": round(xT_increase, 4)
    }

@mcp.tool()
def player_performance(match_id: int) -> list:
    """Calcula el rendimiento de cada jugador en el partido: suma de xT y acciones clave (pases/tiros importantes)."""
    df_actions, df_players, df_teams = _load_match_data(match_id)
    if df_actions.empty:
        return []  # partido vacío o no encontrado
    # Inicializar diccionario de acumuladores por jugador
    performance = {}  # key: player_id, value: stats dict
    for _, action in df_actions.iterrows():
        pid = action["player_id"]
        if pid not in performance:
            performance[pid] = {
                "player": action["player_name"],
                "team": action["team_name"],
                "total_xT": 0.0,
                "positive_actions": 0,   # acciones con xT > 0
                "shots": 0,
                "goals": 0,
                "assists": 0
            }
        val = action["xT_value"]
        # Sumar xT (si no es NaN)
        if val == val:  # comprobación de NaN
            performance[pid]["total_xT"] += float(val)
            if val > 0:
                performance[pid]["positive_actions"] += 1
        # Contabilizar tiros y goles
        if action["type_name"] == "shot":
            performance[pid]["shots"] += 1
            # Si el tiro fue gol (éxito)
            if action["result_name"] == "success":
                performance[pid]["goals"] += 1
                # Marcar asistencia para el jugador de la acción anterior si es mismo equipo
                prev_idx = action.name - 1  # índice anterior
                if prev_idx in df_actions.index:
                    prev_action = df_actions.loc[prev_idx]
                    if prev_action["team_id"] == action["team_id"] and prev_action["type_name"] in ["pass", "cross"]:
                        prev_pid = prev_action["player_id"]
                        if prev_pid in performance:
                            performance[prev_pid]["assists"] += 1
    # Convertir a lista de estadísticas por jugador
    players_stats = list(performance.values())
    # Opcional: redondear total_xT a 4 decimales por claridad
    for stats in players_stats:
        stats["total_xT"] = round(stats["total_xT"], 4)
    return players_stats

if __name__ == "__main__":
    # Ejecutar servidor MCP (modo desarrollo por defecto)
    mcp.run()
