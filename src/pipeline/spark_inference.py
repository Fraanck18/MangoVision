"""
Pipeline distribuido de inferencia batch sobre el Data Lake en HDFS.

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
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from src.config import HDFS_RAW_PARQUET_PATH, HDFS_CURATED_RESULTS_PATH, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE
from src.models.resnet50_custom import (
    cargar_modelo_mlflow_registry,
    cargar_imagen_hdfs,
    obtener_transformaciones_validacion,
)

esquema_salida = StructType([
    StructField("uri_hdfs", StringType(), nullable=False),
    StructField("diagnostico_metabolico", IntegerType(), nullable=False),
])

PARTICIONES_POR_NUCLEO = 3
IMAGENES_POR_PARTICION_OBJETIVO = 300


def batch_resnet50_inference(iterator):
    """Inicializacion unica del modelo por particion en el nodo Executor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = cargar_modelo_mlflow_registry(MLFLOW_MODEL_NAME, stage=MLFLOW_MODEL_STAGE).to(device)
    modelo.eval()

    preprocess = obtener_transformaciones_validacion()

    for pdf in iterator:
        tensores = [preprocess(cargar_imagen_hdfs(uri)) for uri in pdf["uri_hdfs"]]
        lote_apilado = torch.stack(tensores).to(device)

        with torch.no_grad():
            logits = modelo(lote_apilado)
            predicciones = torch.argmax(logits, dim=1).cpu().numpy()

        pdf["diagnostico_metabolico"] = predicciones
        yield pdf[["uri_hdfs", "diagnostico_metabolico"]]


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
