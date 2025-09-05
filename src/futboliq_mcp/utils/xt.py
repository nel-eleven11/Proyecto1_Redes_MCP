from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class FirstHighImpactError:
    minute: float
    team: str
    player: str
    action: str
    xt_before: float
    xt_after: float
    d_xt: float
    context: Dict[str, Any]
    suggested_option: Optional[Dict[str, Any]] = None

def _ensure_xt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula xT por acción si no existe.
    Sugerencia: usar socceraction (xThreat) para grid y expectativa.
    Aquí dejamos un placeholder simple; reemplázalo con el pipeline real.
    """
    if "xt_start" not in df.columns or "xt_end" not in df.columns:
        # TODO: integrar socceraction.xthreat para computar xt_start/xt_end reales
        df = df.copy()
        df["xt_start"] = 0.0
        df["xt_end"] = 0.0
    return df

def compute_xt_delta(df_events: pd.DataFrame, focus_team: Optional[str], dt_seconds: int, threshold: float) -> FirstHighImpactError:
    df = _ensure_xt_columns(df_events)

    # ΔxT por acción
    df["d_xt"] = df["xt_end"] - df["xt_start"]

    if focus_team:
        df = df[df["team"] == focus_team]

    # Buscar primera acción con ΔxT <= umbral negativo (error de alto impacto)
    idx = df[df["d_xt"] <= threshold].index.min()
    if pd.isna(idx):
        raise RuntimeError("No se encontró un error de alto impacto con el umbral dado.")

    row = df.loc[idx]
    minute = float(row.get("minute", 0)) + float(row.get("second", 0))/60.0

    # TODO: generar 'suggested_option' con heurística (vecinos con mayor xT esperado)
    suggestion = None

    return FirstHighImpactError(
        minute=minute,
        team=str(row.get("team","")),
        player=str(row.get("player","")),
        action=str(row.get("type","")),
        xt_before=float(row.get("xt_start",0.0)),
        xt_after=float(row.get("xt_end",0.0)),
        d_xt=float(row.get("d_xt",0.0)),
        context={"index": int(idx)},
        suggested_option=suggestion,
    )
