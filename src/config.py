"""Configuracion centralizada de URIs HDFS, MLflow y variables de entorno del sistema MangoVision."""
import os

HDFS_NAMENODE = os.getenv("HDFS_NAMENODE", "hdfs://nodo-maestro:9000")
HDFS_RAW_PATH = os.getenv("HDFS_RAW_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/raw/")
HDFS_RAW_PARQUET_PATH = os.getenv("HDFS_RAW_PARQUET_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/raw_parquet/")
HDFS_CURATED_RESULTS_PATH = os.getenv(
    "HDFS_CURATED_RESULTS_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/curated_results/"
)
HDFS_MANIFEST_PATH = os.getenv("HDFS_MANIFEST_PATH", "data/hdfs_manifest.json")

# Capa de ingesta desde sensores termicos en la cadena de frio de transporte.
# Un camion no tiene ruta de red sostenida hacia el NameNode de HDFS, por lo
# que el sensor nunca escribe directo al Data Lake: acumula en un buffer local
# (disco del gateway de borde) y solo sube por lotes cuando recupera señal.
# BUFFER_LOCAL_PATH y STAGING_PATH simulan en disco lo que en produccion
# serian el almacenamiento del gateway y una cola MQTT/Kafka + staging S3.
BUFFER_LOCAL_PATH = os.getenv("BUFFER_LOCAL_PATH", "dataset_mangos/ingesta/buffer_local")
STAGING_PATH = os.getenv("STAGING_PATH", "dataset_mangos/ingesta/staging")
HDFS_RAW_PATH_LOCAL = os.getenv("HDFS_RAW_PATH_LOCAL", "dataset_mangos/ingesta/hdfs_simulado/raw")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "resnet18-mango-qc")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mango-thermal-qc")

CLASS_NAMES = ["defectuoso", "sano"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42
TEST_SIZE = 0.2
