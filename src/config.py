"""Configuracion centralizada de URIs HDFS, MLflow y variables de entorno del sistema MangoVision."""
import os

HDFS_NAMENODE = os.getenv("HDFS_NAMENODE", "hdfs://nodo-maestro:9000")
HDFS_RAW_PATH = os.getenv("HDFS_RAW_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/raw/")
HDFS_RAW_PARQUET_PATH = os.getenv("HDFS_RAW_PARQUET_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/raw_parquet/")
HDFS_CURATED_RESULTS_PATH = os.getenv(
    "HDFS_CURATED_RESULTS_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/curated_results/"
)
HDFS_MANIFEST_PATH = os.getenv("HDFS_MANIFEST_PATH", "data/hdfs_manifest.json")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "resnet18-mango-qc")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mango-thermal-qc")

CLASS_NAMES = ["sano", "necrosis_celular", "desorden_metabolico"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42
TEST_SIZE = 0.2
