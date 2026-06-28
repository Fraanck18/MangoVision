"""
Daemon de evaluacion online (Online Evaluation Pipeline).

Consume el topic Kafka de imagenes en continuo y, por cada evento:
  1. Carga la imagen referenciada en el Data Lake.
  2. Infiere ŷ con el modelo de produccion.
  3. Lee el ground-truth y_true del propio evento (Oraculo: el productor
     sintetico conoce la etiqueta real -> Opcion A de la arquitectura).
  4. Calcula is_correct = (ŷ == y_true) y lo persiste en SQLite.

La metrica que se muestra NO es la exactitud historica acumulada (que se
congela y oculta caidas), sino una VENTANA DESLIZANTE sobre las ultimas N
muestras -> la metrica "baila" con el flujo reciente. Ademas detecta DRIFT:
si la accuracy de ventana cae bajo el umbral Y la confianza media de la red
tambien, levanta una alerta (p.ej. condensacion en lente FLIR).

El calculo pesado vive aqui (daemon en segundo plano), nunca en el dashboard:
el frontend solo hace fetch del resultado ya calculado (endpoint /metricas/vivo).
"""
import logging
import sqlite3
import threading
import time
from pathlib import Path

from PIL import Image

from src.config import (
    DRIFT_UMBRAL_ACCURACY,
    DRIFT_UMBRAL_CONFIANZA,
    EVALUACIONES_DB,
    VENTANA_CONFIANZA,
    VENTANA_DESLIZANTE,
)
from src.inference.diagnostico import diagnosticar_imagen_pil
from src.streaming.kafka_cliente import crear_consumidor

logger = logging.getLogger("mangovision.evaluador")

_detener = threading.Event()


def _conexion():
    Path(EVALUACIONES_DB).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(EVALUACIONES_DB, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def inicializar_db():
    con = _conexion()
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_evaluations (
            ts            REAL    NOT NULL,
            id_imagen     TEXT,
            y_true        TEXT,
            y_pred        TEXT,
            confianza     REAL,
            is_correct    INTEGER,
            degradada     INTEGER,
            lote_ingesta  TEXT
        )
    """)
    con.commit()
    con.close()


def registrar_evaluacion(fila):
    con = _conexion()
    con.execute(
        "INSERT INTO live_evaluations (ts, id_imagen, y_true, y_pred, confianza, is_correct, degradada, lote_ingesta) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        fila,
    )
    con.commit()
    con.close()


def calcular_metricas_vivo():
    """Lee SQLite y calcula la metrica de ventana deslizante + estado de drift."""
    con = _conexion()
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM live_evaluations")
    total = cur.fetchone()[0]
    if total == 0:
        con.close()
        return {"estado": "sin_datos", "muestras_totales": 0}

    cur.execute(
        "SELECT AVG(is_correct) FROM (SELECT is_correct FROM live_evaluations ORDER BY ts DESC LIMIT ?)",
        (VENTANA_DESLIZANTE,),
    )
    rolling_accuracy = cur.fetchone()[0] or 0.0

    cur.execute(
        "SELECT AVG(confianza) FROM (SELECT confianza FROM live_evaluations ORDER BY ts DESC LIMIT ?)",
        (VENTANA_CONFIANZA,),
    )
    confianza_media = cur.fetchone()[0] or 0.0

    # Tendencia: comparar la mitad reciente de la ventana con la mitad anterior.
    cur.execute(
        "SELECT is_correct FROM live_evaluations ORDER BY ts DESC LIMIT ?",
        (VENTANA_DESLIZANTE,),
    )
    valores = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT ts FROM live_evaluations ORDER BY ts DESC LIMIT 1")
    ultima_ts = cur.fetchone()[0]
    con.close()

    mitad = max(1, len(valores) // 2)
    acc_reciente = sum(valores[:mitad]) / mitad
    acc_previa = sum(valores[mitad:]) / max(1, len(valores) - mitad)
    delta = acc_reciente - acc_previa
    if delta <= -0.10:
        trend = "DOWNWARD_SHARP"
    elif delta < -0.02:
        trend = "downward"
    elif delta > 0.02:
        trend = "upward"
    else:
        trend = "stable"

    # Deteccion de drift con DOS señales complementarias (OR, no AND):
    #  - Accuracy de ventana: señal AUTORITATIVA porque el stream trae el
    #    ground-truth (oraculo). Es la que manda cuando hay etiquetas.
    #  - Confianza media (entropia del softmax): proxy NO supervisado, util
    #    cuando en produccion no llegan etiquetas. Nota: las CNN suelen ser
    #    sobreconfiadas en entradas fuera de distribucion, por lo que la
    #    accuracy etiquetada detecta la degradacion mucho antes que la confianza.
    drift_por_accuracy = rolling_accuracy < DRIFT_UMBRAL_ACCURACY
    drift_por_confianza = confianza_media < DRIFT_UMBRAL_CONFIANZA
    drift = drift_por_accuracy or drift_por_confianza

    resultado = {
        "estado": "ok",
        "rolling_accuracy": round(rolling_accuracy * 100, 1),
        "confianza_media": round(confianza_media, 3),
        "ventana": VENTANA_DESLIZANTE,
        "muestras_totales": total,
        "trend": trend,
        "drift_alert": drift,
        "ultima_actualizacion": ultima_ts,
    }
    if drift:
        causa = "accuracy de ventana" if drift_por_accuracy else "confianza de la red"
        resultado["causa_drift"] = causa
        resultado["msg"] = (
            f"Posible condensacion en lente FLIR o perdida de cadena de frio: caida de {causa} "
            f"(acc={resultado['rolling_accuracy']}%, conf={resultado['confianza_media']})."
        )
    return resultado


def _evaluar_evento(evento, estado_modelo):
    ruta = Path(evento["ruta_imagen"])
    if not ruta.exists():
        return
    try:
        imagen = Image.open(ruta).convert("RGB")
    except Exception:
        logger.warning("Imagen ilegible en evento: %s", ruta)
        return

    diag = diagnosticar_imagen_pil(
        imagen, estado_modelo["modelo"], estado_modelo["device"], estado_modelo["preprocess"]
    )
    y_true = evento.get("true_label")
    y_pred = diag["etiqueta"]
    registrar_evaluacion((
        time.time(),
        evento.get("id_lectura"),
        y_true,
        y_pred,
        diag["confianza"],
        int(y_pred == y_true),
        int(bool(evento.get("degradada"))),
        evento.get("lote_ingesta"),
    ))


def _bucle_consumidor(estado_modelo):
    inicializar_db()
    while not _detener.is_set():
        if estado_modelo.get("modelo") is None:
            _detener.wait(2)
            continue
        consumidor = crear_consumidor(timeout_ms=1500)
        if consumidor is None:
            logger.warning("Evaluador online: Kafka no disponible, reintentando en 10s.")
            _detener.wait(10)
            continue
        try:
            for mensaje in consumidor:
                if _detener.is_set():
                    break
                try:
                    _evaluar_evento(mensaje.value, estado_modelo)
                except Exception:
                    logger.exception("Fallo evaluando un evento del stream.")
        except Exception:
            logger.exception("Error en el bucle del consumidor Kafka.")
        finally:
            try:
                consumidor.close()
            except Exception:
                pass


def iniciar_evaluador_online(estado_modelo):
    """Arranca el daemon evaluador en segundo plano (no bloqueante). Idempotente."""
    if any(h.name == "evaluador-online" for h in threading.enumerate()):
        return
    inicializar_db()
    hilo = threading.Thread(target=_bucle_consumidor, args=(estado_modelo,), name="evaluador-online", daemon=True)
    hilo.start()
    logger.info("Daemon de evaluacion online iniciado (ventana=%d).", VENTANA_DESLIZANTE)


def detener_evaluador_online():
    _detener.set()
