"""
Capa de streaming sobre Kafka real (KRaft, en Docker).

Productor y consumidor del topic de imagenes termicas. Cada mensaje es un
EVENTO que referencia la imagen ya aterrizada en el Data Lake HDFS REAL
(NameNode+DataNode de docker-compose.hdfs.yml, verificado E2E) y lleva su
ground-truth -- patron industrial correcto: Kafka transporta el evento ligero,
no el blob pesado; la imagen vive en el lake. Esto desacopla
productor (sensor del camion) de consumidor (evaluador online), de modo que el
broker es intercambiable sin tocar la logica de negocio.

Se degrada con elegancia: si el broker no esta disponible, las funciones de
publicacion no rompen el pipeline (la ingesta a HDFS sigue), solo se omite el
evento de streaming y se registra un warning.
"""
import json
import logging

from src.config import KAFKA_BOOTSTRAP, KAFKA_GRUPO_EVALUADOR, KAFKA_TOPIC_STREAM

logger = logging.getLogger("mangovision.kafka")

_productor = None


def _crear_productor():
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        retries=2,
        acks=1,
        request_timeout_ms=4000,
        api_version_auto_timeout_ms=4000,
    )


def obtener_productor():
    """Productor singleton; devuelve None si el broker no esta disponible."""
    global _productor
    if _productor is not None:
        return _productor
    try:
        _productor = _crear_productor()
        logger.info("Productor Kafka conectado a %s", KAFKA_BOOTSTRAP)
    except Exception:
        logger.warning("Kafka no disponible en %s; el pipeline sigue sin streaming.", KAFKA_BOOTSTRAP)
        _productor = None
    return _productor


def publicar_evento_imagen(evento):
    """
    Publica un evento de imagen al topic. `evento` es un dict con al menos:
    {id_lectura, ruta_imagen, true_label, lote_ingesta, degradada}.
    No lanza si el broker esta caido (degradacion elegante).
    """
    productor = obtener_productor()
    if productor is None:
        return False
    try:
        productor.send(KAFKA_TOPIC_STREAM, evento)
        return True
    except Exception:
        logger.exception("Fallo publicando evento en Kafka.")
        return False


def crear_consumidor(timeout_ms=1500):
    """
    Crea un consumidor del topic para el daemon evaluador. Devuelve None si el
    broker no esta disponible. `auto_offset_reset='earliest'`: en el primer
    arranque procesa el historico para poblar la ventana de inmediato; luego,
    gracias a los offsets commiteados del group_id, reanuda donde se quedo (no
    reprocesa), comportandose como evaluacion online continua.
    """
    from kafka import KafkaConsumer

    try:
        return KafkaConsumer(
            KAFKA_TOPIC_STREAM,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=KAFKA_GRUPO_EVALUADOR,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=timeout_ms,
            api_version_auto_timeout_ms=4000,
        )
    except Exception:
        logger.warning("No se pudo crear consumidor Kafka en %s.", KAFKA_BOOTSTRAP)
        return None
