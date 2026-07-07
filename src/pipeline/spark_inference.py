"""
Pipeline distribuido de inferencia batch sobre el Data Lake en HDFS.

[Opcion C — VERIFICADO E2E, 2026-07-07] Ya NO es "diseñado pero nunca probado":
se ejecuto contra el cluster real (`docker-compose.hdfs.yml`, Spark 3.5.1 +
Hadoop) leyendo `raw_parquet` real de HDFS, corriendo ResNet50 en el executor y
escribiendo `curated_results` real en HDFS. La carga del modelo admite un
checkpoint LOCAL via `MANGO_MODEL_PATH` (arquitectura-consciente), de modo que el
pipeline corre SIN depender de un MLflow Registry levantado.
NOTA de honestidad: que la TUBERIA corra E2E no dice nada de la CALIDAD del
modelo (este ResNet50 no supera al baseline trivial de brillo; ver docs/INTEGRIDAD.md).

Explota la localidad de los datos de Apache Spark: cada particion del
DataFrame es procesada por un Executor que carga el modelo ResNet50 UNA SOLA
VEZ por particion (no por fila) mediante `mapInPandas`, evitando tanto la
recarga repetida del modelo como el colapso de memoria de los nodos.

Decisiones de escala para >10k imagenes:

1. Estrategia de paralelismo: Data Parallelism, no Model Parallelism.
   ResNet50 cabe entero en la memoria de cualquier executor (~98MB de
   pesos); particionar el propio modelo entre nodos (tecnica reservada
   para modelos que no caben en un solo dispositivo, p.ej. LLMs) no aplica
   aqui. El cuello de botella es el volumen de imagenes, no el tamano de
   la red.
2. Anti "small file problem" de HDFS: las imagenes NUNCA se leen como
   archivos sueltos en este pipeline. Se leen empaquetadas como bytes
   dentro de columnas de Parquet (`raw_parquet/`), agrupando cientos de
   imagenes por archivo, evitando la sobrecarga del NameNode al gestionar
   millones de archivos pequenos.
3. Partition pruning por lote de ingesta: `raw_parquet/` se espera
   particionado en disco por `lote_ingesta` (fecha o id de carga batch).
   Filtrar por ese lote antes de procesar permite a Spark leer solo los
   datos nuevos en cada corrida, sin reescanear todo el Data Lake.
4. Calibracion de `repartition(N)`: N se deriva del paralelismo por
   defecto del cluster (nucleos de executor disponibles), no de la
   cantidad de imagenes. Demasiadas particiones repiten el costo de
   cargar el modelo por particion; muy pocas subutilizan el cluster.
"""
import os

import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from src.config import HDFS_RAW_PARQUET_PATH, HDFS_CURATED_RESULTS_PATH, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE
from src.models.resnet50_custom import (
    cargar_modelo_mlflow_registry,
    cargar_pesos_arquitectura_consciente,
    cargar_imagen_hdfs,
    obtener_transformaciones_validacion,
)
from src.inference.tta import probabilidades_tta

esquema_salida = StructType([
    StructField("uri_origen", StringType(), nullable=False),
    StructField("provenance", StringType(), nullable=True),
    StructField("diagnostico_metabolico", IntegerType(), nullable=False),
])

PARTICIONES_POR_NUCLEO = 3
IMAGENES_POR_PARTICION_OBJETIVO = 300

# [#9 TTA] Test-Time Augmentation en el pipeline batch. Barato aqui (ya vamos por
# lotes) y reduce la varianza por frame. Opt-in por env (MANGO_TTA=1).
USAR_TTA = os.getenv("MANGO_TTA", "0") == "1"

# [Opcion C] Ruta a un checkpoint LOCAL (best_model.pth) para cargar el modelo sin
# depender de un MLflow Registry levantado. Si esta seteada, tiene prioridad y la
# carga es ARQUITECTURA-CONSCIENTE (detecta ResNet50 del checkpoint). Asi el
# pipeline distribuido corre E2E aunque no haya servidor MLflow.
MANGO_MODEL_PATH = os.getenv("MANGO_MODEL_PATH", "")


def _cargar_modelo_inferencia(device):
    """
    Carga el modelo de inferencia en el executor. Prioridad:
      1. MANGO_MODEL_PATH (checkpoint local, sin MLflow) -> arquitectura-consciente.
      2. MLflow Model Registry (produccion).
    """
    if MANGO_MODEL_PATH and os.path.exists(MANGO_MODEL_PATH):
        return cargar_pesos_arquitectura_consciente(MANGO_MODEL_PATH, device)
    return cargar_modelo_mlflow_registry(MLFLOW_MODEL_NAME, stage=MLFLOW_MODEL_STAGE).to(device).eval()


def _imagen_de_fila(fila):
    """
    Obtiene la imagen de una fila del Parquet curado. Prioriza la columna
    `imagen_bytes` (imagenes EMPAQUETADAS dentro del Parquet, anti small-file, tal
    como describe el docstring de este modulo y produce scripts/etl_raw_a_parquet_spark.py),
    y cae a leer por URI desde HDFS solo si no hay bytes inline.
    """
    import io
    from PIL import Image

    datos = fila.get("imagen_bytes")
    if datos is not None:
        return Image.open(io.BytesIO(datos)).convert("RGB")
    return cargar_imagen_hdfs(fila["uri_origen"])


def batch_resnet50_inference(iterator):
    """Inicializacion unica del modelo por particion en el nodo Executor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = _cargar_modelo_inferencia(device)
    modelo.eval()

    preprocess = obtener_transformaciones_validacion()

    for pdf in iterator:
        tensores = [preprocess(_imagen_de_fila(fila)) for _, fila in pdf.iterrows()]
        lote_apilado = torch.stack(tensores).to(device)

        with torch.no_grad():
            if USAR_TTA:
                probabilidades = probabilidades_tta(modelo, lote_apilado)
                predicciones = torch.argmax(probabilidades, dim=1).cpu().numpy()
            else:
                logits = modelo(lote_apilado)
                predicciones = torch.argmax(logits, dim=1).cpu().numpy()

        pdf["diagnostico_metabolico"] = predicciones
        if "provenance" not in pdf.columns:
            pdf["provenance"] = "real"
        yield pdf[["uri_origen", "provenance", "diagnostico_metabolico"]]


def calcular_particiones_optimas(spark, num_filas):
    """
    Deriva N para repartition() del paralelismo real del cluster (nucleos de
    executor disponibles), acotado por un tamano de particion objetivo, en
    vez de un numero fijo independiente de la escala del dataset.
    """
    nucleos_disponibles = spark.sparkContext.defaultParallelism
    particiones_por_nucleos = nucleos_disponibles * PARTICIONES_POR_NUCLEO
    particiones_por_volumen = max(1, num_filas // IMAGENES_POR_PARTICION_OBJETIVO)
    return max(1, min(particiones_por_nucleos, particiones_por_volumen) or particiones_por_nucleos)


def ejecutar_pipeline_batch(lote_ingesta=None):
    """
    lote_ingesta: si se especifica, filtra `raw_parquet/` por esa columna de
    particion (partition pruning) para procesar solo el lote nuevo en vez de
    reescanear todo el Data Lake.
    """
    spark = SparkSession.builder.appName("MangoQC-BatchPipeline").getOrCreate()

    df_imagenes = spark.read.parquet(HDFS_RAW_PARQUET_PATH)
    if lote_ingesta is not None:
        df_imagenes = df_imagenes.filter(df_imagenes.lote_ingesta == lote_ingesta)

    num_particiones = calcular_particiones_optimas(spark, df_imagenes.count())

    df_resultados = df_imagenes.repartition(num_particiones).mapInPandas(
        batch_resnet50_inference, schema=esquema_salida
    )

    escritor = df_resultados.write.mode("overwrite")
    if lote_ingesta is not None:
        escritor = escritor.option("partitionOverwriteMode", "dynamic")
    escritor.parquet(HDFS_CURATED_RESULTS_PATH)

    spark.stop()


if __name__ == "__main__":
    import sys

    lote = sys.argv[1] if len(sys.argv) > 1 else None
    ejecutar_pipeline_batch(lote_ingesta=lote)
