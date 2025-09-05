from typing import Optional
import pandas as pd

def load_events(match_id: Optional[int]=None, events_path: Optional[str]=None) -> pd.DataFrame:
    """
    Carga eventos desde StatsBomb Open Data (statsbombpy) o un archivo local.
    Devuelve un DataFrame normalizado con columnas mínimas: ['minute','second','team','player','type','x','y','end_x','end_y', ...]
    """
    if events_path:
        # TODO: autodetect JSON/CSV y normalizar columnas
        if events_path.endswith(".json"):
            return pd.read_json(events_path)
        return pd.read_csv(events_path)
    elif match_id is not None:
        try:
            from statsbombpy import sb
        except Exception as e:
            raise RuntimeError("Instala statsbombpy para usar match_id") from e
        df = sb.events(match_id=match_id)
        # TODO: mapear a columnas mínimas esperadas por compute_xt_delta
        return df
    else:
        raise ValueError("Proporciona 'events_path' o 'match_id'")
