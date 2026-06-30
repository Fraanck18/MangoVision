"""
Cliente real de HDFS via WebHDFS (REST), sin dependencias nativas (libhdfs/
Java) en el host. Apunta al NameNode levantado en `docker-compose.hdfs.yml`.

Se usa la API REST en vez del binding nativo de pyarrow porque la capa de
ingesta corre en el host (camion/gateway de borde, potencialmente Windows
sin Hadoop instalado); WebHDFS solo necesita HTTP.
"""
from hdfs import InsecureClient

from src.config import HDFS_WEBHDFS_URL, HDFS_WEBHDFS_USER

_cliente = None


def obtener_cliente():
    """Cliente WebHDFS compartido (lazy, una sola conexion por proceso)."""
    global _cliente
    if _cliente is None:
        _cliente = InsecureClient(HDFS_WEBHDFS_URL, user=HDFS_WEBHDFS_USER)
    return _cliente


def subir_archivo(ruta_local, ruta_hdfs, sobrescribir=True):
    """Sube un archivo local al HDFS real (equivalente a `hdfs dfs -put`)."""
    cliente = obtener_cliente()
    cliente.upload(ruta_hdfs, str(ruta_local), overwrite=sobrescribir)


def existe(ruta_hdfs):
    cliente = obtener_cliente()
    return cliente.status(ruta_hdfs, strict=False) is not None


def listar_subcarpetas(ruta_hdfs):
    """Nombres de subdirectorios directos de `ruta_hdfs` (vacio si no existe)."""
    cliente = obtener_cliente()
    if cliente.status(ruta_hdfs, strict=False) is None:
        return []
    return sorted(
        nombre for nombre, detalle in cliente.list(ruta_hdfs, status=True) if detalle["type"] == "DIRECTORY"
    )


def listar_archivos(ruta_hdfs):
    """Nombres de archivos directos de `ruta_hdfs` (vacio si no existe)."""
    cliente = obtener_cliente()
    if cliente.status(ruta_hdfs, strict=False) is None:
        return []
    return sorted(nombre for nombre, detalle in cliente.list(ruta_hdfs, status=True) if detalle["type"] == "FILE")


def leer_bytes(ruta_hdfs):
    """Lee el contenido completo de un archivo en HDFS."""
    cliente = obtener_cliente()
    with cliente.read(ruta_hdfs) as flujo:
        return flujo.read()
