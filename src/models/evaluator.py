"""
Auditoria dinamica de metricas + GATE de produccion RELATIVO Y HONESTO.

[Fallo critico #1 — el gate ya no es teatro]
Antes: `assert exactitud > 0.90`. Ese 0.90 estaba POR ENCIMA de la linea base
honesta alcanzable (~0.60-0.72), asi que en datos reales SIEMPRE lanzaba
AssertionError, los scripts la ATRAPABAN y la descartaban -> una alarma
desconectada. Un umbral por encima de lo alcanzable no es rigor, es una mentira
estructural.

Ahora el gate:
  * Deriva su umbral de la LINEA BASE HONESTA DOCUMENTADA en disco
    (`web/data/cv_honesta.json`, producido por el LOGO restringido a verdad
    humana), no de un numero aspiracional quemado. Umbral = baseline + margen.
  * Devuelve un VEREDICTO ESTRUCTURADO (dict) en vez de lanzar una excepcion que
    los callers se tragan. Asi el gate realmente puede bloquear, o dejar pasar con
    aviso, y el resultado es inspeccionable.
  * Se mide SOLO sobre muestras REALES (regla de oro de las tecnicas de expansion:
    se entrena con sinteticos/pseudo, se MIDE contra verdad humana). El caller pasa
    una mascara `es_real` cuando el conjunto mezcla procedencias.

Todas las cifras se calculan al vuelo desde los tensores, sin diccionarios
estaticos ni valores quemados (ver tests/test_metrics.py).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

RAIZ = Path(__file__).resolve().parents[2]
BASELINE_JSON = RAIZ / "web" / "data" / "cv_honesta.json"

# Margen exigido por encima de la baseline documentada (env-configurable). 0.0 =
# "no regresar por debajo de lo ya conseguido"; subelo para exigir mejora real.
MARGEN_DEFECTO = float(os.getenv("MANGO_GATE_MARGEN", "0.0"))
# Solo si NO hay baseline documentada en disco ni env: piso conservador (no un
# 0.90 inalcanzable). Es deliberadamente bajo para no bloquear por accidente
# cuando aun no se ha medido la linea base honesta.
BASELINE_FALLBACK = float(os.getenv("MANGO_GATE_FALLBACK", "0.5"))


def leer_baseline_honesta(ruta=BASELINE_JSON):
    """
    Devuelve (baseline, fuente). Prioridad:
      1. env MANGO_BASELINE_ACCURACY (override explicito),
      2. `accuracy_media` de cv_honesta.json (LOGO restringido a verdad humana),
      3. BASELINE_FALLBACK (sin baseline documentada).
    """
    env = os.getenv("MANGO_BASELINE_ACCURACY")
    if env is not None:
        return float(env), "env:MANGO_BASELINE_ACCURACY"
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        if datos.get("accuracy_media") is not None:
            grupos = datos.get("n_grupos_humanos_evaluables", "?")
            return float(datos["accuracy_media"]), f"cv_honesta.json (LOGO, {grupos} grupos humanos)"
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    return BASELINE_FALLBACK, "fallback (sin baseline documentada en disco)"


def _a_numpy(t):
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def evaluar_gate(y_verdadero, y_predicho, baseline=None, margen=MARGEN_DEFECTO, es_real=None):
    """
    VEREDICTO ESTRUCTURADO del gate de produccion. No lanza excepciones.

    y_verdadero, y_predicho : tensores/arrays de clases.
    baseline : si None, se lee de disco (leer_baseline_honesta).
    margen   : cuanto hay que superar la baseline para aprobar.
    es_real  : mascara booleana opcional; si se da, la metrica se calcula SOLO sobre
               las muestras reales (regla de oro: medir contra verdad humana).

    Devuelve dict: {aprobado, accuracy, f1_weighted, umbral_usado, baseline,
                    margen, fuente_baseline, n_muestras, n_reales, motivo}.
    """
    y_true = _a_numpy(y_verdadero)
    y_pred = _a_numpy(y_predicho)
    n_total = int(len(y_true))

    if es_real is not None:
        es_real = _a_numpy(es_real).astype(bool)
        y_true, y_pred = y_true[es_real], y_pred[es_real]

    n_reales = int(len(y_true))
    if n_reales == 0:
        return {"aprobado": False, "accuracy": None, "f1_weighted": None,
                "umbral_usado": None, "baseline": None, "margen": margen,
                "fuente_baseline": None, "n_muestras": n_total, "n_reales": 0,
                "motivo": "No hay muestras REALES para medir el gate (todo era expansion)."}

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))

    fuente = "explicito"
    if baseline is None:
        baseline, fuente = leer_baseline_honesta()
    umbral = float(baseline) + float(margen)
    aprobado = accuracy >= umbral

    motivo = (
        f"APROBADO: accuracy real {accuracy:.3f} >= umbral {umbral:.3f} "
        f"(baseline {baseline:.3f} + margen {margen:.3f}, fuente: {fuente})."
        if aprobado else
        f"BLOQUEADO: accuracy real {accuracy:.3f} < umbral {umbral:.3f} "
        f"(baseline {baseline:.3f} + margen {margen:.3f}, fuente: {fuente})."
    )
    return {
        "aprobado": bool(aprobado),
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "umbral_usado": round(umbral, 4),
        "baseline": round(float(baseline), 4),
        "margen": float(margen),
        "fuente_baseline": fuente,
        "n_muestras": n_total,
        "n_reales": n_reales,
        "motivo": motivo,
    }


def evaluar_convergencia_real(y_verdadero_tensor, y_predicho_tensor, es_real=None):
    """
    Calcula el reporte de clasificacion dinamico y ejecuta el gate relativo. Ya NO
    lanza AssertionError: imprime el veredicto estructurado del gate y devuelve el
    DataFrame de metricas (contrato que esperan los scripts de entrenamiento).

    Para obtener el veredicto como dato (no solo impreso), usa `evaluar_gate`.
    """
    y_true = _a_numpy(y_verdadero_tensor)
    y_pred = _a_numpy(y_predicho_tensor)

    veredicto = evaluar_gate(y_true, y_pred, es_real=es_real)
    estado = "APROBADO" if veredicto["aprobado"] else "BLOQUEADO"
    print(f"[GATE {estado}] {veredicto['motivo']}")
    if veredicto["f1_weighted"] is not None:
        print(f"[AUDITORIA] F1 ponderado (dinamico): {veredicto['f1_weighted'] * 100:.2f}%")

    reporte_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(reporte_dict).transpose()
