# MangoVision

Sistema de inspección no destructiva de calidad para mango de exportación, basado en termografía infrarroja y aprendizaje profundo. Clasifica la calidad del fruto mediante **detección de anomalías térmicas** —cuánto se desvía su firma térmica de la población normal— sin muestreo destructivo y a escala industrial. **No** diagnostica una patología concreta (necrosis, desorden metabólico): señala la desviación térmica, y esa es la señal de calidad. Ver [`docs/INTEGRIDAD.md`](docs/INTEGRIDAD.md).

## Contexto del problema

La inspección visual manual de mango solo evalúa la superficie del fruto y es ineficaz para detectar defectos internos. MangoVision aprovecha la **inercia térmica** de la pulpa, pero — al disponer solo de un **único frame estático** por fruto — no mide la dinámica de enfriamiento (dT/dt), sino su huella espacial: el **Perfil Espacial de Distribución Térmica Estática (PEDTE)**, el gradiente térmico centro-periferia y las anomalías locales captables en una sola imagen — ver [`src/domain_physics/thermal_inertia.py`](src/domain_physics/thermal_inertia.py). Las intensidades se expresan como **Índice de Intensidad Térmica Relativa (ITR)**, no en °C: la cámara de consumo entrega pseudo-color, no radiometría absoluta.

## Arquitectura

| Dimensión | Implementación |
| :--- | :--- |
| Almacenamiento | Data Lake distribuido en HDFS (`data-lake/mangos/`) |
| Cómputo distribuido | Apache Spark, inferencia batch vía `mapInPandas` |
| Modelo principal | Backbone ligero (ResNet18 por defecto / MobileNetV3, `MANGO_BACKBONE`) + cabecera BatchNorm/Dropout |
| Modelo de referencia | HOG + LinearSVC · y features físicas PEDTE + LogisticRegression |
| Tracking / Registry | MLflow |
| Despliegue | FastAPI + Docker |

## Resultados

> **Sin métricas heredadas.** Las cifras anteriores (~95 %) provenían de otra
> arquitectura (ResNet18 previo) con pesos incompatibles y, sobre todo, se medían
> en su mayoría sobre un **dataset sintético** — no eran exactitud sobre fruta real.
> Se han retirado por integridad.

La **Línea Base Real Independiente** se obtiene ejecutando el pipeline honesto y se
publicará aquí con su valor real (aunque sea modesto):

- **Pre-entrenamiento** sobre el pool sintético estocástico (solo domain adaptation).
- **Fine-tuning** sobre el dato real con separación estricta **Train / Validation /
  Test** (el Test real queda bloqueado y se evalúa una única vez; el modelo se
  elige por *Validation*, nunca por *test*).
- **Evaluación honesta** por **Leave-One-Group-Out** agrupando por mango físico
  (`scripts/honest_grouped_cv.py`).

```bash
python scripts/generate_synthetic_thermal.py   # pool sintético de PRE-ENTRENAMIENTO (aislado)
python scripts/train_resnet50.py               # pre-train + fine-tune + test único → best_model.pth
python scripts/honest_grouped_cv.py            # métrica honesta LOGO restringido a verdad humana
```

> **Correcciones de fallos críticos** (carga de pesos arquitectura-consciente, ETL
> Spark raw→parquet, LOGO restringido a procedencia humana, y gate de auditoría
> relativo y honesto): ver **[`docs/FIXES_CRITICOS.md`](docs/FIXES_CRITICOS.md)**.
> El gate ya no usa un 0.90 quemado inalcanzable: deriva su umbral de la línea base
> honesta documentada (`web/data/cv_honesta.json`) y devuelve un veredicto
> estructurado que realmente bloquea o aprueba, medido solo sobre datos reales.

Todas las métricas se calculan dinámicamente sobre tensores reales en
`src/models/evaluator.py` — sin valores estáticos ni hardcoding (ver
`tests/test_metrics.py`).

## Data Augmentation avanzada

Con muy pocas imágenes térmicas reales, el aumento de datos es central. La
estrategia completa (9 técnicas *térmicamente seguras*, su justificación y dónde
vive cada una) está en **[`docs/DATA_AUGMENTATION.md`](docs/DATA_AUGMENTATION.md)**.
Regla que lo filtra todo: en pseudo-color Ironbow **el matiz es la etiqueta**, así
que nunca se perturba el color; las técnicas o son geométricas, o operan en el
dominio térmico (ITR) y re-codifican con la paleta física, o mezclan en el espacio
de la etiqueta.

```bash
python scripts/materializar_augmentation.py      # #1 augmentation física (offline -> HDFS-ready)
python scripts/pretrain_self_supervised.py       # #2 SimSiam sobre el stream NO etiquetado
python scripts/train_gan_augment.py              # #8 cGAN + DiffAugment (data-efficient)
# #4-7 (MixUp, erasing, elástica, geométrico) van integradas en el train;
# #3 (agregación temporal) y #9 (TTA) actúan en streaming/inferencia.
```

### Evaluación honesta (detección de anomalías)

```bash
python scripts/deteccion_anomalias.py    # detector one-class + baseline trivial (AUC)
python scripts/honest_grouped_cv.py      # LOGO por mango sobre dato real (binario)
```

**Resultado presentable** (fruta real, held-out por mango): un score de **brillo
térmico separa `defectuoso` de `alta_calidad` a AUC 0.87** (etiqueta humana → no
circular); el detector PEDTE (0.65) **no lo supera**, así que se reporta que la señal
es simple. Ledger completo en [`docs/RESULTADOS.md`](docs/RESULTADOS.md).

> [!important] Alcance real e integridad
> MangoVision es **clasificación de calidad por detección de ANOMALÍAS térmicas**
> (cuánto se desvía la firma del fruto de lo normal), **no** un diagnóstico de
> necrosis interna. La augmentation de temperatura y las muestras físicas son
> **datos de entrenamiento / sondas declaradas**, nunca conjunto de evaluación: un
> baseline trivial de brillo iguala al CNN sobre ellas (tautología, +0.9 pp). La
> única métrica honesta es LOGO por mango sobre dato real, con el baseline trivial
> al lado. Contrato en **[`docs/INTEGRIDAD.md`](docs/INTEGRIDAD.md)** · resultados en
> **[`docs/RESULTADOS.md`](docs/RESULTADOS.md)**.

Código en [`src/augmentation/`](src/augmentation/),
[`src/inference/tta.py`](src/inference/tta.py) y
[`src/streaming/agregacion_temporal.py`](src/streaming/agregacion_temporal.py);
tests en [`tests/test_augmentation.py`](tests/test_augmentation.py).

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
