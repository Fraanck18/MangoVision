"""Configuracion centralizada de URIs HDFS, MLflow y variables de entorno del sistema MangoVision."""
import os

# HDFS real (NameNode + DataNode en docker-compose.hdfs.yml). Dos endpoints
# del mismo cluster:
#   HDFS_NAMENODE       -> RPC nativo (hdfs://), usado por pyarrow/Spark.
#   HDFS_WEBHDFS_URL    -> REST WebHDFS (http://), usado por el cliente
#                          python puro `hdfs` en la capa de ingesta (no
#                          requiere libhdfs/Java nativo en el host).
# El hostname "namenode" debe resolver al host docker (ver README: entrada
# en /etc/hosts o C:\Windows\System32\drivers\etc\hosts -> 127.0.0.1 namenode datanode).
HDFS_NAMENODE = os.getenv("HDFS_NAMENODE", "hdfs://namenode:9000")
HDFS_WEBHDFS_URL = os.getenv("HDFS_WEBHDFS_URL", "http://namenode:9870")
HDFS_WEBHDFS_USER = os.getenv("HDFS_WEBHDFS_USER", "root")

HDFS_RAW_PATH = os.getenv("HDFS_RAW_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/raw/")
HDFS_RAW_PARQUET_PATH = os.getenv("HDFS_RAW_PARQUET_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/raw_parquet/")
HDFS_CURATED_RESULTS_PATH = os.getenv(
    "HDFS_CURATED_RESULTS_PATH", f"{HDFS_NAMENODE}/data-lake/mangos/curated_results/"
)
HDFS_RAW_PATH_WEBHDFS = os.getenv("HDFS_RAW_PATH_WEBHDFS", "/data-lake/mangos/raw")
HDFS_MANIFEST_PATH = os.getenv("HDFS_MANIFEST_PATH", "data/hdfs_manifest.json")

# Capa de ingesta desde sensores termicos en la cadena de frio de transporte.
# Un camion no tiene ruta de red sostenida hacia el NameNode de HDFS, por lo
# que el sensor nunca escribe directo al Data Lake: acumula en un buffer local
# (disco del gateway de borde) y solo sube por lotes cuando recupera señal.
# BUFFER_LOCAL_PATH y STAGING_PATH son el almacenamiento real del gateway de
# borde (esto seguiria siendo disco local incluso en produccion real); lo
# unico que dejo de simularse es el destino final: ya no es una carpeta
# local, es el HDFS real de arriba.
BUFFER_LOCAL_PATH = os.getenv("BUFFER_LOCAL_PATH", "dataset_mangos/ingesta/buffer_local")
STAGING_PATH = os.getenv("STAGING_PATH", "dataset_mangos/ingesta/staging")

# Preferencias de UI (tema, etc.) persistidas en backend para sobrevivir
# entre navegadores/dispositivos, no solo en localStorage del navegador actual.
PREFERENCIAS_USUARIO_PATH = os.getenv("PREFERENCIAS_USUARIO_PATH", "dataset_mangos/preferencias_usuario.json")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "resnet50-mango-qc")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mango-thermal-qc")

CLASS_NAMES = ["alta_calidad", "calidad_promedio", "defectuoso"]
NUM_CLASSES = len(CLASS_NAMES)

# Mapeo de cada clase del clasificador (entrenado sobre dataset_mangos/processed)
# a la lectura de madurez termica mostrada en el dashboard. El modelo no mide
# temperatura directamente: el rango traduce su salida a una magnitud legible.
MATURITY_INFO = {
    "calidad_promedio": {"etiqueta": "Verde / Inmaduro", "rango_temp": (15.0, 21.0), "color": "#2f6fed"},
    "alta_calidad": {"etiqueta": "Maduro / Optimo", "rango_temp": (22.0, 32.0), "color": "#16a34a"},
    "defectuoso": {"etiqueta": "Sobre-Maduro", "rango_temp": (33.0, 42.0), "color": "#dc2626"},
}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Simulacion de llegada de imagenes del camion en tiempo real: cada cuanto
# "llega" un lote nuevo del sensor termico y cuantas imagenes trae cada lote.
# Sin un camion fisico ni un cluster HDFS real disponible, el pool de origen
# es el split de test del dataset real (nunca usado en entrenamiento).
INTERVALO_INGESTA_SEGUNDOS = int(os.getenv("INTERVALO_INGESTA_SEGUNDOS", "300"))
IMAGENES_MIN_POR_LOTE = 2
IMAGENES_MAX_POR_LOTE = 6
POOL_IMAGENES_SENSOR = "dataset_mangos/processed/test"

# Workers para inferencia en paralelo (paralelismo de datos: cada worker usa
# una copia logica del mismo modelo ya entrenado sobre un subconjunto propio
# de imagenes, no hay aproximacion ni perdida de precision).
NUM_WORKERS_INFERENCIA = int(os.getenv("NUM_WORKERS_INFERENCIA", str(min(4, os.cpu_count() or 2))))
LOTES_PROCESADOS_PATH = os.getenv("LOTES_PROCESADOS_PATH", "dataset_mangos/ingesta/lotes_procesados.json")

# --- Streaming Kafka + evaluacion online en ventana deslizante ---
# El broker es un Kafka real (KRaft, sin Zookeeper) corriendo en Docker. El
# scheduler actua de PRODUCTOR (publica un evento por imagen al topic con su
# ground-truth) y un daemon consumidor evalua en continuo: ŷ vs y_true ->
# is_correct -> ventana deslizante -> deteccion de drift. La metrica deja de
# estar congelada y "baila" segun el flujo reciente.
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_STREAM = os.getenv("KAFKA_TOPIC_STREAM", "mango-thermal-stream")
KAFKA_GRUPO_EVALUADOR = os.getenv("KAFKA_GRUPO_EVALUADOR", "evaluador-online")

EVALUACIONES_DB = os.getenv("EVALUACIONES_DB", "dataset_mangos/ingesta/live_evaluations.sqlite")
VENTANA_DESLIZANTE = int(os.getenv("VENTANA_DESLIZANTE", "60"))      # ultimas N muestras para accuracy
VENTANA_CONFIANZA = int(os.getenv("VENTANA_CONFIANZA", "12"))        # ultimas N para confianza media (drift)
DRIFT_UMBRAL_ACCURACY = float(os.getenv("DRIFT_UMBRAL_ACCURACY", "0.88"))
DRIFT_UMBRAL_CONFIANZA = float(os.getenv("DRIFT_UMBRAL_CONFIANZA", "0.65"))

# Fraccion de imagenes que llegan "degradadas" (condensacion en lente FLIR /
# perdida de cadena de frio): blur + ruido + shift de brillo. Provocan errores
# realistas para que la metrica de ventana suba y baje, y se dispare el drift.
FRACCION_DEGRADADAS = float(os.getenv("FRACCION_DEGRADADAS", "0.18"))
