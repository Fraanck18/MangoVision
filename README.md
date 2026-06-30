# MangoVision

Sistema de inspección no destructiva de calidad para mango de exportación, basado en termografía infrarroja y aprendizaje profundo. Detecta necrosis celular y desórdenes metabólicos internos sin recurrir a muestreos destructivos, a escala industrial.

## Contexto del problema

La inspección visual manual de mango solo evalúa la superficie del fruto y es ineficaz para detectar defectos internos. MangoVision aprovecha el principio de **Inercia Térmica** (Ley de Enfriamiento de Newton): el tejido necrosado altera el calor específico y la conductividad de la pulpa, generando gradientes térmicos superficiales anómalos y detectables por cámara infrarroja — ver [`src/domain_physics/thermal_inertia.py`](src/domain_physics/thermal_inertia.py).

## Arquitectura

| Dimensión | Implementación |
| :--- | :--- |
| Almacenamiento | Data Lake distribuido en HDFS (`data-lake/mangos/`) |
| Cómputo distribuido | Apache Spark, inferencia batch vía `mapInPandas` |
| Modelo principal | ResNet50 (Transfer Learning) + cabecera con BatchNorm/Dropout |
| Modelo de referencia | HOG + LinearSVC |
| Tracking / Registry | MLflow |
| Despliegue | FastAPI + Docker |

## Resultados (ResNet50 vs. Baseline)

| Métrica | ResNet50 | Baseline (HOG+SVM) |
| :--- | :--- | :--- |
| Accuracy | 95.2%* | — |
| F1-Score | 95.5%* | — |
| Precision | 96.0%* | 89.5% |
| Recall | 95.0%* | 88.0% |

\* Cifras heredadas de la corrida con ResNet18 (arquitectura anterior); se actualizan automáticamente la próxima vez que se ejecute `scripts/train_resnet50.py`, ya que los pesos viejos no son compatibles con la nueva arquitectura y el modelo debe reentrenarse.

Todas las métricas se calculan dinámicamente sobre tensores reales en `src/models/evaluator.py` — sin valores estáticos ni hardcoding (ver `tests/test_metrics.py`).

## Estructura del repositorio

```text
mango-thermal-qc/
├── data/hdfs_manifest.json        # Manifiesto de linaje (esquema; no contiene binarios)
├── notebooks/                     # EDA y curvas de pérdida
├── src/
│   ├── config.py                  # URIs de HDFS, MLflow y constantes
│   ├── domain_physics/            # Fundamento físico (Ley de Newton)
│   ├── models/                    # ResNet50, baseline SVM, evaluador
│   ├── pipeline/                  # Inferencia batch distribuida en PySpark
│   └── api/                       # Servicio FastAPI de producción
├── tests/                         # Tests de física y anti-hardcoding de métricas
├── Dockerfile
└── requirements.txt
```

## Uso local

```bash
pip install -r requirements.txt
pytest tests/ -v

# Servir la API (requiere MLflow Tracking Server con el modelo registrado)
uvicorn src.api.app:app --reload
```

## HDFS real (Docker, un solo nodo)

```bash
# 1. Levantar NameNode + DataNode reales
docker compose -f docker-compose.hdfs.yml up -d

# 2. Resolver los hostnames del cluster desde el host (una sola vez,
#    requiere admin). En Windows, como Administrador:
#    Add-Content C:\Windows\System32\drivers\etc\hosts "127.0.0.1 namenode datanode"
#    En Linux/Mac: echo "127.0.0.1 namenode datanode" | sudo tee -a /etc/hosts

# 3. Crear la estructura del Data Lake
docker exec mangovision-hdfs-namenode hdfs dfs -mkdir -p /data-lake/mangos/raw /data-lake/mangos/raw_parquet /data-lake/mangos/curated_results

# WebUI del NameNode (estado del cluster, navegador de archivos): http://localhost:9870
```

La capa de ingesta (`src/ingestion/uploader_batch.py`, `src/pipeline/worker_inference.py`) habla con este HDFS real via WebHDFS (`src/ingestion/hdfs_client.py`, librería `hdfs`, sin dependencias nativas). No hay carpeta local que simule HDFS: el destino final de cada lote es el NameNode de arriba.

## Cluster Spark real (Docker, conectado al HDFS de arriba)

`spark_inference.py` y `cargar_imagen_hdfs` (en `resnet50_custom.py`) usan el cliente nativo de HDFS de pyarrow/Spark (`hdfs://...`, libhdfs vía JNI), no WebHDFS. Esto requiere un cliente Hadoop nativo instalado (`HADOOP_HOME`, `JAVA_HOME`, `libhdfs.so`) — por eso `docker-compose.hdfs.yml` incluye un Spark Master + Worker reales (`Dockerfile.spark`: imagen oficial de Spark 3.5.1 + Hadoop 3.3.6 nativo + PyTorch CPU-only + el código de `src/`), en la misma red Docker que el NameNode.

```bash
# Construye la imagen de Spark (Hadoop nativo + PyTorch + mlflow/pyarrow) y
# levanta master + worker junto con el HDFS real.
docker compose -f docker-compose.hdfs.yml up -d --build

# WebUI del Spark Master (estado del cluster, workers, jobs): http://localhost:8080
# WebUI del Spark Worker: http://localhost:8081

# Correr el pipeline batch real dentro del cluster:
docker compose -f docker-compose.hdfs.yml exec spark-master \
  /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
  /opt/mangovision/src/pipeline/spark_inference.py
```

Verificado en este entorno: el Worker se registra `ALIVE` ante el Master, `pyarrow.fs.FileSystem` (vía `libhdfs` nativo) lee imágenes reales del Data Lake desde dentro del contenedor, y el cliente JVM de Spark (`fs.listStatus`) ve los mismos lotes — ambos caminos confirmados contra el mismo HDFS real, no solo "diseñado pero nunca probado".

**Pendiente**: `spark_inference.py` espera que las imágenes ya estén empaquetadas en `raw_parquet/` (anti "small file problem" de HDFS, ver docstring del archivo). Esa etapa de ETL — convertir lo que aterriza en `raw/` a Parquet particionado — todavía no existe en este repo; sin ella, `ejecutar_pipeline_batch()` falla al no encontrar datos en `raw_parquet/`.

## Docker

```bash
docker build -t mangovision-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 mangovision-api
```
