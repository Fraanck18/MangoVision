"""
Subida por lotes desde el buffer local del camion hasta el Data Lake.

Flujo: buffer local (disco del gateway) -> staging (en produccion: cola
MQTT/Kafka + bucket de staging) -> HDFS, particionado por `lote_ingesta`.
Solo se ejecuta cuando `hay_conexion()` es verdadero -- nunca se asume
conectividad, se verifica antes de cada intento de subida.

El particionado por `lote_ingesta` no es incidental: es lo que permite a
`src/pipeline/spark_inference.py` filtrar y procesar unicamente el lote
nuevo en cada corrida batch (partition pruning), en vez de reescanear todo
el Data Lake cada vez que un camion sube datos.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import BUFFER_LOCAL_PATH, STAGING_PATH, HDFS_RAW_PATH_WEBHDFS
from src.ingestion.edge_buffer import listar_pendientes
from src.ingestion.hdfs_client import subir_archivo


def hay_conexion(verificador_conexion=None):
    """
    Punto unico de verificacion de conectividad. En produccion: ping al
    gateway de telemetria de la flota o chequeo de socket. En este repo se
    inyecta un callback (o se asume False) para no asumir conectividad por
    defecto -- igual que en la realidad, sin verificacion explicita no hay
    subida.
    """
    if verificador_conexion is None:
        return False
    return bool(verificador_conexion())


def drenar_buffer_a_staging(carpeta_buffer=None, carpeta_staging=None, verificador_conexion=None):
    """
    Mueve las lecturas pendientes del buffer local a staging (simulacro de
    publicar a la cola MQTT/Kafka). No borra nada del buffer si no hay
    conexion: el camion sigue acumulando hasta la proxima parada con señal.
    """
    if not hay_conexion(verificador_conexion):
        return []

    carpeta_buffer = Path(carpeta_buffer or BUFFER_LOCAL_PATH)
    carpeta_staging = Path(carpeta_staging or STAGING_PATH)
    carpeta_staging.mkdir(parents=True, exist_ok=True)

    movidas = []
    for lectura in listar_pendientes(carpeta_buffer):
        id_lectura = lectura["metadata"]["id_lectura"]
        shutil.move(str(lectura["ruta_imagen"]), carpeta_staging / lectura["ruta_imagen"].name)
        shutil.move(str(lectura["ruta_metadata"]), carpeta_staging / lectura["ruta_metadata"].name)
        movidas.append(id_lectura)

    return movidas


def subir_staging_a_hdfs(lote_ingesta=None, carpeta_staging=None, carpeta_hdfs_raiz=None):
    """
    Aterriza lo que esta en staging dentro del Data Lake en HDFS real,
    particionado por `lote_ingesta` (por defecto la fecha UTC de la subida).
    Equivalente a `hdfs dfs -put` hacia `raw/lote_ingesta=<fecha>/`, pero via
    WebHDFS (ver `src/ingestion/hdfs_client.py`).
    """
    carpeta_staging = Path(carpeta_staging or STAGING_PATH)
    lote_ingesta = lote_ingesta or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    carpeta_destino_hdfs = f"{(carpeta_hdfs_raiz or HDFS_RAW_PATH_WEBHDFS).rstrip('/')}/lote_ingesta={lote_ingesta}"

    subidas = []
    for ruta_metadata in sorted(carpeta_staging.glob("*.json")):
        ruta_imagen = ruta_metadata.with_suffix(".jpg")
        if not ruta_imagen.exists():
            continue
        subir_archivo(ruta_imagen, f"{carpeta_destino_hdfs}/{ruta_imagen.name}")
        subir_archivo(ruta_metadata, f"{carpeta_destino_hdfs}/{ruta_metadata.name}")
        ruta_imagen.unlink()
        ruta_metadata.unlink()
        subidas.append(ruta_imagen.stem)

    return {"lote_ingesta": lote_ingesta, "subidas": subidas, "carpeta_destino": carpeta_destino_hdfs}
