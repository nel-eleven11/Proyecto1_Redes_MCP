from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from mcp.server import Tool
from futboliq_mcp.data_providers.statsbomb import load_events
from futboliq_mcp.utils.xt import compute_xt_delta, FirstHighImpactError

class AnalyzeMatchParams(BaseModel):
    match_id: Optional[int] = Field(default=None, description="StatsBomb Open Data match_id")
    events_path: Optional[str] = Field(default=None, description="Ruta a JSON/CSV de eventos")
    focus_team: Optional[str] = Field(default=None, description="Nombre del equipo a analizar")
    dt_seconds: int = Field(default=12, description="Ventana temporal para ΔxT")
    threshold: float = Field(default=-0.05, description="Umbral de ΔxT para error de alto impacto")

class AnalyzeMatchResult(BaseModel):
    minute: float
    team: str
    player: str
    action: str
    xt_before: float
    xt_after: float
    d_xt: float
    context: Dict[str, Any]
    suggested_option: Optional[Dict[str, Any]] = None

async def _handler(params: AnalyzeMatchParams) -> Dict[str, Any]:
    df_events = load_events(match_id=params.match_id, events_path=params.events_path)
    err: FirstHighImpactError = compute_xt_delta(
        df_events,
        focus_team=params.focus_team,
        dt_seconds=params.dt_seconds,
        threshold=params.threshold,
    )
    # Serializar resultado mínimo
    return AnalyzeMatchResult(
        minute=err.minute,
        team=err.team,
        player=err.player,
        action=err.action,
        xt_before=err.xt_before,
        xt_after=err.xt_after,
        d_xt=err.d_xt,
        context=err.context,
        suggested_option=err.suggested_option,
    ).model_dump()

analyze_match_tool = Tool(
    name="analyze_match",
    description="Detecta el primer error de alto impacto (ΔxT negativo) en un partido.",
    input_schema=AnalyzeMatchParams.model_json_schema(),
    handler=lambda kwargs: _handler(AnalyzeMatchParams(**kwargs)),
)
