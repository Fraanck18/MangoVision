"""
Simulador de llegada en tiempo real del sensor termico del camion.

Sin un camion fisico ni un cluster HDFS real disponible para esta demo, este
modulo reproduce el mismo flujo que documentan `edge_buffer.py` y
`uploader_batch.py` (buffer local -> staging -> HDFS particionado por
lote_ingesta), pero disparado por un hilo en segundo plano cada
`INTERVALO_INGESTA_SEGUNDOS` en vez de por un sensor real. Las imagenes que
"llegan" se toman del split de test del dataset real (nunca visto en
entrenamiento), por lo que la clasificacion posterior sigue siendo honesta.

Cada imagen capturada tambien se aumenta con Volteo Horizontal (Horizontal
Flip / Reflexion Especular Horizontal): la version espejada conserva la
misma etiqueta real (un mango sigue siendo el mismo mango volteado), por lo
que no introduce ruido en la evaluacion -- es la misma augmentacion que ya
usa `obtener_transformaciones_entrenamiento` para el entrenamiento. El
resultado es que cada lote aterriza en HDFS con el doble de imagenes
(original + espejo), y `/hdfs/analizar` las procesa a todas igual.
"""
import io
import json
import logging
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.config import (
    FRACCION_DEGRADADAS,
    IMAGENES_MAX_POR_LOTE,
    IMAGENES_MIN_POR_LOTE,
    INTERVALO_INGESTA_SEGUNDOS,
    POOL_IMAGENES_SENSOR,
)
from src.ingestion.edge_buffer import capturar_lectura
from src.ingestion.hdfs_client import leer_bytes, listar_archivos
from src.ingestion.uploader_batch import drenar_buffer_a_staging, subir_staging_a_hdfs
from src.streaming.kafka_cliente import publicar_evento_imagen

logger = logging.getLogger("mangovision.ingestion")

_detener = threading.Event()

# Modo "fallo de sensor" para demo: cuando _fallo_lotes_restantes > 0, los
# proximos lotes llegan casi todos degradados (condensacion en lente / perdida
# de cadena de frio), tumbando la metrica y disparando el drift; luego se
# auto-recupera. Se activa via POST /simular/fallo-sensor.
_fallo_lotes_restantes = {"n": 0}
FRACCION_DEGRADADAS_FALLO = 0.9


def activar_fallo_sensor(n_lotes=3):
    _fallo_lotes_restantes["n"] = int(n_lotes)
    logger.warning("Modo FALLO DE SENSOR activado para los proximos %d lotes.", n_lotes)


def _fraccion_degradadas_actual():
    return FRACCION_DEGRADADAS_FALLO if _fallo_lotes_restantes["n"] > 0 else FRACCION_DEGRADADAS


def _pool_imagenes():
    raiz = Path(POOL_IMAGENES_SENSOR)
    return list(raiz.glob("*/*.jpg")) + list(raiz.glob("*/*.jpeg")) + list(raiz.glob("*/*.png"))


def _volteo_horizontal_bytes(imagen_bytes):
    """Devuelve la version espejada (izquierda-derecha) de la imagen, re-codificada a JPEG."""
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    volteada = imagen.transpose(Image.FLIP_LEFT_RIGHT)
    buffer_salida = io.BytesIO()
    volteada.save(buffer_salida, format="JPEG")
    return buffer_salida.getvalue()


def _degradar_imagen_bytes(imagen_bytes, severa=False):
    """
    Simula un FALLO DE SENSOR (condensacion en la lente FLIR / perdida de
    cadena de frio): desenfoque + ruido + shift de brillo.

    - Leve (goteo normal): degradacion moderada; el modelo suele acertar igual,
      pero baja algo la confianza -> la metrica de ventana sube y baja.
    - Severa (modo fallo activado): ademas corrompe el balance de canales RGB,
      que es la señal que el modelo usa para leer la temperatura via la paleta
      Ironbow. Esto hace que falle de verdad y con baja confianza -> dispara la
      deteccion de drift.
    """
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    if severa:
        # Lente totalmente empañada / perdida total de cadena de frio: la imagen
        # queda dominada por ruido y se corrompe el color, de modo que la red
        # pierde la señal -> prediccion casi aleatoria y BAJA confianza (softmax
        # plano) -> dispara el detector de drift.
        imagen = imagen.filter(ImageFilter.GaussianBlur(radius=random.uniform(9.0, 15.0)))
        arreglo = np.asarray(imagen, dtype=np.float32)
        escala = np.array([random.uniform(0.4, 1.6) for _ in range(3)], dtype=np.float32)
        arreglo *= escala[None, None, :]
        ruido_puro = np.random.uniform(0.0, 255.0, arreglo.shape)
        alpha = random.uniform(0.65, 0.85)  # fraccion de ruido que ahoga la imagen
        arreglo = (1.0 - alpha) * arreglo + alpha * ruido_puro
    else:
        imagen = imagen.filter(ImageFilter.GaussianBlur(radius=random.uniform(3.5, 6.5)))
        arreglo = np.asarray(imagen, dtype=np.float32)
        arreglo += np.random.normal(0.0, random.uniform(18.0, 34.0), arreglo.shape)
        arreglo += random.uniform(-45.0, 45.0)
    arreglo = np.clip(arreglo, 0, 255).astype(np.uint8)
    buffer_salida = io.BytesIO()
    Image.fromarray(arreglo, mode="RGB").save(buffer_salida, format="JPEG")
    return buffer_salida.getvalue()


def _capturar_una(imagen_bytes, sensor_id, clase_real, aumentado):
    """Captura una lectura al buffer, degradandola segun la fraccion y severidad vigentes."""
    en_fallo = _fallo_lotes_restantes["n"] > 0
    degradada = random.random() < _fraccion_degradadas_actual()
    bytes_final = _degradar_imagen_bytes(imagen_bytes, severa=en_fallo) if degradada else imagen_bytes
    capturar_lectura(bytes_final, {
        "sensor_id": sensor_id,
        "clase_real_origen": clase_real,
        "aumentado": aumentado,
        "degradada": degradada,
    })


def _publicar_lote_a_kafka(carpeta_destino, lote_ingesta):
    """
    Publica un evento por imagen aterrizada al topic Kafka (referencia + ground-truth).

    `carpeta_destino` es la ruta en HDFS (WebHDFS) donde el uploader dejo el
    lote: las imagenes ya NO viven en disco local (se subieron y se borraron del
    staging), por lo que la metadata se lee desde HDFS y el evento referencia la
    ruta HDFS de la imagen (patron correcto: Kafka transporta el evento ligero,
    la imagen vive en el Data Lake).
    """
    base = carpeta_destino.rstrip("/")
    nombres_json = [n for n in listar_archivos(base) if n.endswith(".json")]
    publicados = 0
    for nombre_json in sorted(nombres_json):
        nombre_imagen = nombre_json[: -len(".json")] + ".jpg"
        try:
            meta = json.loads(leer_bytes(f"{base}/{nombre_json}").decode("utf-8"))
        except Exception:
            logger.warning("Metadata ilegible en HDFS: %s/%s", base, nombre_json)
            continue
        ok = publicar_evento_imagen({
            "id_lectura": meta.get("id_lectura"),
            "ruta_imagen": f"{base}/{nombre_imagen}",
            "true_label": meta.get("clase_real_origen"),
            "aumentado": meta.get("aumentado"),
            "degradada": meta.get("degradada", False),
            "lote_ingesta": lote_ingesta,
            "timestamp": meta.get("timestamp_captura_utc"),
        })
        publicados += int(ok)
    if publicados:
        logger.info("Publicados %d eventos al topic Kafka del lote %s", publicados, lote_ingesta)


def _simular_llegada_lote(sensor_id="camion-07-sensor-termico"):
    """Una 'parada del camion con señal': captura un lote nuevo (con volteo horizontal y posible degradacion), lo aterriza en HDFS y publica sus eventos a Kafka."""
    pool = _pool_imagenes()
    if not pool:
        logger.warning("Pool de imagenes del sensor vacio en %s", POOL_IMAGENES_SENSOR)
        return None

    tamano_lote = random.randint(IMAGENES_MIN_POR_LOTE, IMAGENES_MAX_POR_LOTE)
    elegidas = random.sample(pool, min(tamano_lote, len(pool)))

    for ruta_imagen in elegidas:
        clase_real = ruta_imagen.parent.name
        imagen_bytes = ruta_imagen.read_bytes()
        _capturar_una(imagen_bytes, sensor_id, clase_real, "original")
        _capturar_una(_volteo_horizontal_bytes(imagen_bytes), sensor_id, clase_real, "horizontal_flip")

    if _fallo_lotes_restantes["n"] > 0:
        _fallo_lotes_restantes["n"] -= 1

    drenar_buffer_a_staging(verificador_conexion=lambda: True)
    # Cada llegada simulada es su propio lote_ingesta (granularidad de segundo,
    # no de dia) para que /hdfs/analizar pueda distinguir lotes consecutivos
    # separados por INTERVALO_INGESTA_SEGUNDOS en vez de fusionarlos por fecha.
    lote_ingesta = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    resultado = subir_staging_a_hdfs(lote_ingesta=lote_ingesta)
    logger.info(
        "Lote de ingesta %s aterrizado en HDFS con %d imagenes",
        resultado["lote_ingesta"], len(resultado["subidas"]),
    )
    _publicar_lote_a_kafka(resultado["carpeta_destino"], resultado["lote_ingesta"])
    return resultado


def _bucle_ingesta():
    while not _detener.is_set():
        try:
            _simular_llegada_lote()
        except Exception:
            logger.exception("Fallo simulando la llegada de un lote del sensor termico.")
        _detener.wait(INTERVALO_INGESTA_SEGUNDOS)


def iniciar_scheduler_ingesta():
    """Arranca el hilo de ingesta en segundo plano (no bloqueante). Idempotente."""
    if any(hilo.name == "ingesta-hdfs" for hilo in threading.enumerate()):
        return
    hilo = threading.Thread(target=_bucle_ingesta, name="ingesta-hdfs", daemon=True)
    hilo.start()
    logger.info("Scheduler de ingesta HDFS iniciado (cada %ds).", INTERVALO_INGESTA_SEGUNDOS)


def detener_scheduler_ingesta():
    _detener.set()
