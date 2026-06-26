"""
Pipeline distribuido de inferencia batch sobre el Data Lake en HDFS.

Explota la localidad de los datos de Apache Spark: cada particion del
DataFrame es procesada por un Executor que carga el modelo ResNet18 UNA SOLA
VEZ por particion (no por fila) mediante `mapInPandas`, evitando tanto la
recarga repetida del modelo como el colapso de memoria de los nodos.
"""
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from src.config import HDFS_RAW_PARQUET_PATH, HDFS_CURATED_RESULTS_PATH, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE
from src.models.resnet18_custom import (
    cargar_modelo_mlflow_registry,
    cargar_imagen_hdfs,
    obtener_transformaciones_validacion,
)

esquema_salida = StructType([
    StructField("uri_hdfs", StringType(), nullable=False),
    StructField("diagnostico_metabolico", IntegerType(), nullable=False),
])


def batch_resnet18_inference(iterator):
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


def ejecutar_pipeline_batch(num_particiones=16):
    spark = SparkSession.builder.appName("MangoQC-BatchPipeline").getOrCreate()

    df_imagenes = spark.read.parquet(HDFS_RAW_PARQUET_PATH)

    df_resultados = df_imagenes.repartition(num_particiones).mapInPandas(
        batch_resnet18_inference, schema=esquema_salida
    )

    df_resultados.write.mode("overwrite").parquet(HDFS_CURATED_RESULTS_PATH)
    spark.stop()


if __name__ == "__main__":
    ejecutar_pipeline_batch()
