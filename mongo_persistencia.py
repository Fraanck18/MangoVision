"""
Capa de persistencia en MongoDB Atlas para la DATA DE CLASIFICACION.

Guarda un documento por cada diagnostico realizado (subida manual /diagnostico
o lote procesado por los workers /hdfs/analizar). Es ADITIVO: no reemplaza el
SQLite de evaluacion online ni toca la arquitectura Kafka/HDFS/Spark; solo
suma un almacen consultable y durable en la nube.

La URI se lee de la variable de entorno MONGO_URI (nunca hardcodeada). Si no
esta definida o Atlas no responde, las funciones degradan sin romper la app
(la clasificacion sigue funcionando; solo no se persiste en Mongo).
"""
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("mangovision.mongo")

_DB_NOMBRE = os.getenv("MONGO_DB", "mango-vision")
_COL_NOMBRE = os.getenv("MONGO_COLLECTION", "clasificacion_mangos")

_coleccion = None
_intento_hecho = False


def _get_coleccion():
    """Inicializa el cliente una sola vez (lazy). Devuelve None si no hay Mongo."""
    global _coleccion, _intento_hecho
    if _coleccion is not None:
        return _coleccion
    if _intento_hecho:
        return None
    _intento_hecho = True

    uri = os.getenv("MONGO_URI")
    if not uri:
        logger.warning("MONGO_URI no definida: la clasificacion NO se persistira en MongoDB.")
        return None
    try:
        from pymongo import MongoClient
        cliente = MongoClient(uri, serverSelectionTimeoutMS=5000)
        cliente.admin.command("ping")
        _coleccion = cliente[_DB_NOMBRE][_COL_NOMBRE]
        logger.info("MongoDB conectado: %s.%s", _DB_NOMBRE, _COL_NOMBRE)
        return _coleccion
    except Exception:
        logger.exception("No se pudo conectar a MongoDB; se continua sin persistencia.")
        return None


def guardar_clasificacion(diagnostico, origen):
    """
    Inserta un diagnostico como documento. `origen` = 'manual' | 'lote'.
    Nunca lanza: si Mongo falla, loguea y sigue (la API no debe caerse por esto).
    """
    coleccion = _get_coleccion()
    if coleccion is None:
        return None
    try:
        doc = {
            "ts": datetime.now(timezone.utc),
            "origen": origen,
            "etiqueta": diagnostico.get("etiqueta"),
            "clase_id": diagnostico.get("clase_id"),
            "confianza": diagnostico.get("confianza"),
            "categoria": diagnostico.get("categoria"),
            "indice_itr": diagnostico.get("indice_itr"),
            # Campos presentes solo en resultados de lote (worker_inference):
            "clase_real_origen": diagnostico.get("clase_real_origen"),
            "lote_ingesta": diagnostico.get("lote_ingesta"),
            "aumentado": diagnostico.get("aumentado"),
        }
        return coleccion.insert_one(doc).inserted_id
    except Exception:
        logger.exception("Fallo al guardar clasificacion en MongoDB.")
        return None


def ultimas_clasificaciones(limite=50):
    """Devuelve las ultimas N clasificaciones guardadas (mas recientes primero)."""
    coleccion = _get_coleccion()
    if coleccion is None:
        return {"disponible": False, "total": 0, "documentos": []}
    try:
        cursor = coleccion.find({}, {"_id": 0}).sort("ts", -1).limit(int(limite))
        docs = []
        for d in cursor:
            if isinstance(d.get("ts"), datetime):
                d["ts"] = d["ts"].isoformat()
            docs.append(d)
        return {"disponible": True, "total": coleccion.estimated_document_count(), "documentos": docs}
    except Exception:
        logger.exception("Fallo al leer clasificaciones de MongoDB.")
        return {"disponible": False, "total": 0, "documentos": []}
