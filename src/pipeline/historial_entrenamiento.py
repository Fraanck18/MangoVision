"""
Persistencia del estado de entrenamiento en curso y del historial
append-only de corridas pasadas.

Separado de `scripts/train_resnet50.py` porque tanto el script de
entrenamiento como la API (que necesita poder detenerlo y exponer el
historial vía endpoints) comparten esta misma fuente de verdad; antes cada
corrida sobreescribia `estado_entrenamiento.json` sin dejar rastro de
corridas anteriores.
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
ESTADO_ENTRENAMIENTO = RAIZ / "dataset_mangos" / "estado_entrenamiento.json"
HISTORIAL_ENTRENAMIENTOS = RAIZ / "dataset_mangos" / "historial_entrenamientos.json"


def escribir_estado(datos):
    """Estado de la corrida EN CURSO (se sobreescribe en cada época, es efímero por diseño)."""
    ESTADO_ENTRENAMIENTO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def leer_estado():
    if not ESTADO_ENTRENAMIENTO.exists():
        return {"estado": "sin_iniciar"}
    return json.loads(ESTADO_ENTRENAMIENTO.read_text(encoding="utf-8"))


def leer_historial():
    if not HISTORIAL_ENTRENAMIENTOS.exists():
        return []
    return json.loads(HISTORIAL_ENTRENAMIENTOS.read_text(encoding="utf-8"))


def agregar_a_historial(resumen):
    """
    Append-only: cada corrida (completada o detenida manualmente) se agrega
    como entrada nueva, nunca se sobreescribe una entrada previa. Permite
    comparar entrenamientos a traves del tiempo en vez de solo ver los
    candidatos de la corrida mas reciente.
    """
    historial = leer_historial()
    historial.append(resumen)
    HISTORIAL_ENTRENAMIENTOS.write_text(json.dumps(historial, ensure_ascii=False, indent=2), encoding="utf-8")
