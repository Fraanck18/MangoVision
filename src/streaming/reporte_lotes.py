"""
Reporte BATCH por lote/cargamento (rollup del stream).

Rol correcto del batch (ver docs: streaming vs batch): NO re-clasificar por
"mas precision" (el modelo pesado no supera al brillo trivial -> re-verificar seria
redundante). Su trabajo legitimo es el AGREGADO por envio: al llegar el camion a
destino, resumir cuantos mangos pasaron, la tasa de anomalia a lo largo del viaje
(para detectar ruptura de cadena de frio) y la fraccion degradada.

Agrega lo que el STREAM ya computo (SQLite `live_evaluations`, con su verificacion
de anomalia y `lote_ingesta`); no vuelve a inferir. Reporta SIEMPRE la tasa de
anomalia del detector Y del baseline trivial (guardarrail de honestidad).
"""
import sqlite3
from pathlib import Path

from src.config import EVALUACIONES_DB


def _conexion():
    con = sqlite3.connect(EVALUACIONES_DB, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def calcular_reporte_lotes():
    """
    Devuelve un dict con el reporte agregado por `lote_ingesta` + un resumen global.
    Si no hay datos aun, devuelve estado 'sin_datos'.
    """
    if not Path(EVALUACIONES_DB).exists():
        return {"estado": "sin_datos", "lotes": []}

    con = _conexion()
    filas = con.execute("""
        SELECT lote_ingesta,
               COUNT(*)                         AS n,
               AVG(es_anomalia)                 AS tasa_anom_det,
               AVG(es_anomalia_trivial)         AS tasa_anom_triv,
               AVG(degradada)                   AS frac_degradada,
               AVG(is_correct)                  AS accuracy_3clases,
               MIN(ts)                          AS ts_min
        FROM live_evaluations
        WHERE lote_ingesta IS NOT NULL
        GROUP BY lote_ingesta
        ORDER BY ts_min ASC
    """).fetchall()

    lotes = []
    for f in filas:
        # Distribucion de clases predichas dentro del lote.
        dist = {r["y_pred"]: r["c"] for r in con.execute(
            "SELECT y_pred, COUNT(*) AS c FROM live_evaluations WHERE lote_ingesta = ? GROUP BY y_pred",
            (f["lote_ingesta"],),
        )}
        lotes.append({
            "lote_ingesta": f["lote_ingesta"],
            "n_imagenes": f["n"],
            "tasa_anomalia_detector_pct": round((f["tasa_anom_det"] or 0.0) * 100, 1),
            "tasa_anomalia_trivial_pct": round((f["tasa_anom_triv"] or 0.0) * 100, 1),
            "fraccion_degradada_pct": round((f["frac_degradada"] or 0.0) * 100, 1),
            "accuracy_3clases_vs_humano_pct": round((f["accuracy_3clases"] or 0.0) * 100, 1),
            "distribucion_clases": dist,
        })
    con.close()

    if not lotes:
        return {"estado": "sin_datos", "lotes": []}

    total = sum(l["n_imagenes"] for l in lotes)
    # Tendencia de anomalia (trivial) a lo largo del viaje: sirve para detectar una
    # ruptura de cadena de frio (la tasa sube en lotes tardios).
    tendencia = [{"lote": l["lote_ingesta"], "anomalia_trivial_pct": l["tasa_anomalia_trivial_pct"]}
                 for l in lotes]
    delta = (lotes[-1]["tasa_anomalia_trivial_pct"] - lotes[0]["tasa_anomalia_trivial_pct"]) if len(lotes) > 1 else 0.0

    return {
        "estado": "ok",
        "n_lotes": len(lotes),
        "total_imagenes": total,
        "tendencia_anomalia_trivial": tendencia,
        "delta_anomalia_viaje_pct": round(delta, 1),
        "alerta_cadena_frio": bool(delta >= 20.0),   # subida marcada de anomalia = sospecha
        "lotes": lotes,
        "nota_honestidad": (
            "Rollup del stream; NO re-clasifica. La tasa del detector PEDTE se muestra "
            "junto al baseline trivial de brillo (referencia honesta). El accuracy 3-clases "
            "es contra etiqueta humana y no supera al trivial: usar como contexto, no como KPI."
        ),
    }
