"""
[Fallo #3] Etapa ETL FALTANTE raw -> raw_parquet, con DATA AUGMENTATION fisica.

Antes, `src/pipeline/spark_inference.py` leia `raw_parquet/` pero NADIE lo escribia
-> el pipeline batch de Spark fallaba en la primera lectura y el nucleo
"distribuido" del proyecto no corria E2E. Este script produce ese Parquet.

Que hace, en Spark:
  1. FUENTE del lote nuevo: URIs desde Kafka (topic `mango-etl-raw`) o, si no hay
     Kafka, desde un manifiesto/carpeta local (modo demo). Particion natural por
     `lote_ingesta` -> partition pruning gratis aguas abajo.
  2. Por cada imagen real emite (via `mapInPandas` -> `expandir_particion`):
       * 1 fila original (provenance="real"),
       * K filas fisicamente aumentadas (label-preserving, monotonas en ITR),
       * opcionalmente muestras sinteticas auto-etiquetadas por firma termica
         (provenance="synthetic_physics"), para "anabolizar" clases minoritarias.
  3. ESCRIBE `raw_parquet/` con la imagen como columna `imagen_bytes` +
     `lote_ingesta` + `grupo_fisico` + `clase` + `provenance`, particionado por
     `lote_ingesta` (anti small-file: cientos de imagenes por archivo Parquet).

Con esto `ejecutar_pipeline_batch()` corre, y el volumen que alimenta a Spark se
multiplica x(K+1) con muestras fisicamente validas -> palanca directa del fallo #1.

Uso:
    # Desde Kafka (topic mango-etl-raw con {uri,lote_ingesta,grupo_fisico,clase}):
    spark-submit scripts/etl_raw_a_parquet_spark.py --fuente kafka --lote 20260628T044016

    # Demo local sin Kafka (recorre processed/ como si fuera el raw aterrizado):
    spark-submit scripts/etl_raw_a_parquet_spark.py --fuente local --k-aug 3
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import (
    CLASS_NAMES, HDFS_RAW_PARQUET_PATH, KAFKA_BOOTSTRAP, RANDOM_SEED,
)
from src.pipeline.etl_augmentation import COLUMNAS, expandir_particion

KAFKA_TOPIC_ETL = "mango-etl-raw"
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")
PROCESADOS = RAIZ / "dataset_mangos" / "processed"


def _grupo_de(nombre):
    m = PATRON_GRUPO.search(nombre)
    return m.group(1) if m else Path(nombre).stem


# --------------------------------------------------------------------------
# FUENTES del lote nuevo (devuelven registros {uri, lote_ingesta, grupo_fisico,
# clase, imagen_bytes}). Se mantienen desacopladas del Spark para poder testear.
# --------------------------------------------------------------------------
def registros_desde_local(lote_ingesta="demo_local"):
    """Modo demo: recorre processed/train/<clase> como si fuera el raw aterrizado."""
    registros = []
    for clase in CLASS_NAMES:
        carpeta = PROCESADOS / "train" / clase
        if not carpeta.exists():
            continue
        for ruta in carpeta.glob("*"):
            if ruta.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            registros.append({
                "uri": str(ruta), "lote_ingesta": lote_ingesta,
                "grupo_fisico": _grupo_de(ruta.name), "clase": clase,
                "imagen_bytes": ruta.read_bytes(),
            })
    return registros


def registros_desde_kafka(lote_ingesta, bootstrap=KAFKA_BOOTSTRAP, timeout_ms=8000):
    """
    Consume del topic `mango-etl-raw` los eventos del lote indicado. Cada evento
    referencia una imagen en el Data Lake (WebHDFS): {uri, lote_ingesta,
    grupo_fisico, clase}. Se leen los bytes reales via el cliente HDFS.
    """
    from kafka import KafkaConsumer
    from src.ingestion.hdfs_client import leer_bytes

    consumidor = KafkaConsumer(
        KAFKA_TOPIC_ETL, bootstrap_servers=bootstrap, auto_offset_reset="earliest",
        consumer_timeout_ms=timeout_ms, value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    registros = []
    for mensaje in consumidor:
        ev = mensaje.value
        if lote_ingesta and ev.get("lote_ingesta") != lote_ingesta:
            continue  # partition pruning por lote
        registros.append({
            "uri": ev["uri"], "lote_ingesta": ev.get("lote_ingesta", "sin_lote"),
            "grupo_fisico": ev.get("grupo_fisico") or _grupo_de(Path(ev["uri"]).name),
            "clase": ev.get("clase", ""), "imagen_bytes": leer_bytes(ev["uri"]),
        })
    consumidor.close()
    return registros


# --------------------------------------------------------------------------
# Driver Spark
# --------------------------------------------------------------------------
def ejecutar_etl(registros, k_aug=3, sinteticos_por_base=0, destino=HDFS_RAW_PARQUET_PATH):
    """
    Distribuye `expandir_particion` sobre los registros con `mapInPandas` y escribe
    el Parquet curado particionado por `lote_ingesta`.
    """
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, BinaryType

    spark = SparkSession.builder.appName("MangoQC-ETL-raw-a-parquet").getOrCreate()

    esquema = StructType([
        StructField("uri_origen", StringType(), False),
        StructField("lote_ingesta", StringType(), False),
        StructField("grupo_fisico", StringType(), False),
        StructField("clase", StringType(), True),
        StructField("provenance", StringType(), False),
        StructField("imagen_bytes", BinaryType(), False),
    ])

    df_in = spark.createDataFrame(
        [(r["uri"], r["lote_ingesta"], r["grupo_fisico"], r.get("clase", ""), r["imagen_bytes"])
         for r in registros],
        schema="uri string, lote_ingesta string, grupo_fisico string, clase string, imagen_bytes binary",
    )

    def _expandir(iterador):
        for pdf in iterador:
            registros_part = pdf.rename(columns={"imagen_bytes": "imagen_bytes"}).to_dict("records")
            filas = expandir_particion(
                [{"imagen_bytes": r["imagen_bytes"], "uri": r["uri"],
                  "lote_ingesta": r["lote_ingesta"], "grupo_fisico": r["grupo_fisico"],
                  "clase": r["clase"]} for r in registros_part],
                k_aug=k_aug, sinteticos_por_base=sinteticos_por_base, seed_base=RANDOM_SEED,
            )
            yield pd.DataFrame(filas, columns=COLUMNAS)

    df_out = df_in.repartition("lote_ingesta").mapInPandas(_expandir, schema=esquema)
    (df_out.write.mode("append").partitionBy("lote_ingesta").parquet(destino))
    print(f"ETL: escrito Parquet particionado por lote_ingesta en {destino} "
          f"(K={k_aug}, sinteticos/base={sinteticos_por_base}).")
    spark.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", choices=["kafka", "local"], default="local")
    ap.add_argument("--lote", default="demo_local")
    ap.add_argument("--k-aug", type=int, default=3)
    ap.add_argument("--sinteticos-por-base", type=int, default=0)
    args = ap.parse_args()

    if args.fuente == "kafka":
        registros = registros_desde_kafka(args.lote)
    else:
        registros = registros_desde_local(args.lote)
    print(f"Registros raw del lote '{args.lote}': {len(registros)}")
    if not registros:
        print("Sin registros: nada que materializar.")
        return
    ejecutar_etl(registros, k_aug=args.k_aug, sinteticos_por_base=args.sinteticos_por_base)


if __name__ == "__main__":
    main()
