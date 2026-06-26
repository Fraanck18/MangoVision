"""
Buffer local en el gateway de borde del sensor termico (camion de la cadena
de frio).

Un camion en ruta no tiene conexion de red sostenida hacia el NameNode de
HDFS, por lo que el sensor NUNCA escribe directo al Data Lake: cada lectura
se acumula en disco local (este modulo) y solo se sube por lotes cuando el
camion recupera señal (ver `uploader_batch.py`). Esto es deliberado, no un
parche: si se intentara escribir directo a HDFS y la conexion cae a mitad de
una escritura, se corrompe el archivo; el buffer local hace que la captura
del sensor sea independiente de la disponibilidad de red.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import BUFFER_LOCAL_PATH


def capturar_lectura(imagen_bytes, metadata_sensor, carpeta_buffer=None):
    """
    Escritura local pura: nunca falla por falta de conexion porque no
    depende de la red. Devuelve el id de la lectura bufferizada.

    metadata_sensor: dict con al menos {"sensor_id": str}. Se le agrega
    timestamp de captura en UTC automaticamente.
    """
    carpeta_buffer = Path(carpeta_buffer or BUFFER_LOCAL_PATH)
    carpeta_buffer.mkdir(parents=True, exist_ok=True)

    id_lectura = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

    ruta_imagen = carpeta_buffer / f"{id_lectura}.jpg"
    ruta_metadata = carpeta_buffer / f"{id_lectura}.json"

    ruta_imagen.write_bytes(imagen_bytes)

    metadata_completa = dict(metadata_sensor)
    metadata_completa.update({
        "id_lectura": id_lectura,
        "timestamp_captura_utc": datetime.now(timezone.utc).isoformat(),
        "subido": False,
    })
    ruta_metadata.write_text(json.dumps(metadata_completa, ensure_ascii=False, indent=2))

    return id_lectura


def listar_pendientes(carpeta_buffer=None):
    """Lecturas en el buffer local que aun no se han subido (toda la cola si el camion sigue sin señal)."""
    carpeta_buffer = Path(carpeta_buffer or BUFFER_LOCAL_PATH)
    if not carpeta_buffer.exists():
        return []

    pendientes = []
    for ruta_metadata in sorted(carpeta_buffer.glob("*.json")):
        metadata = json.loads(ruta_metadata.read_text())
        ruta_imagen = ruta_metadata.with_suffix(".jpg")
        if ruta_imagen.exists():
            pendientes.append({"metadata": metadata, "ruta_imagen": ruta_imagen, "ruta_metadata": ruta_metadata})
    return pendientes
