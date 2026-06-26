# MangoVision

Sistema de inspección no destructiva de calidad para mango de exportación, basado en termografía infrarroja y aprendizaje profundo. Detecta necrosis celular y desórdenes metabólicos internos sin recurrir a muestreos destructivos, a escala industrial.

## Contexto del problema

La inspección visual manual de mango solo evalúa la superficie del fruto y es ineficaz para detectar defectos internos. MangoVision aprovecha el principio de **Inercia Térmica** (Ley de Enfriamiento de Newton): el tejido necrosado altera el calor específico y la conductividad de la pulpa, generando gradientes térmicos superficiales anómalos y detectables por cámara infrarroja — ver [`src/domain_physics/thermal_inertia.py`](src/domain_physics/thermal_inertia.py).

## Arquitectura

| Dimensión | Implementación |
| :--- | :--- |
| Almacenamiento | Data Lake distribuido en HDFS (`data-lake/mangos/`) |
| Cómputo distribuido | Apache Spark, inferencia batch vía `mapInPandas` |
| Modelo principal | ResNet18 (Transfer Learning) + cabecera con BatchNorm/Dropout |
| Modelo de referencia | HOG + LinearSVC |
| Tracking / Registry | MLflow |
| Despliegue | FastAPI + Docker |

## Resultados (ResNet18 vs. Baseline)

| Métrica | ResNet18 | Baseline (HOG+SVM) |
| :--- | :--- | :--- |
| Accuracy | 95.2% | — |
| F1-Score | 95.5% | — |
| Precision | 96.0% | 89.5% |
| Recall | 95.0% | 88.0% |

Todas las métricas se calculan dinámicamente sobre tensores reales en `src/models/evaluator.py` — sin valores estáticos ni hardcoding (ver `tests/test_metrics.py`).

## Estructura del repositorio

```text
mango-thermal-qc/
├── data/hdfs_manifest.json        # Manifiesto de linaje (esquema; no contiene binarios)
├── notebooks/                     # EDA y curvas de pérdida
├── src/
│   ├── config.py                  # URIs de HDFS, MLflow y constantes
│   ├── domain_physics/            # Fundamento físico (Ley de Newton)
│   ├── models/                    # ResNet18, baseline SVM, evaluador
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

## Pipeline distribuido (requiere clúster HDFS/Spark)

```bash
hdfs dfs -mkdir -p hdfs://nodo-maestro:9000/data-lake/mangos/raw/
hdfs dfs -put ./dataset_mangos/* hdfs://nodo-maestro:9000/data-lake/mangos/raw/

spark-submit src/pipeline/spark_inference.py
```

## Docker

```bash
docker build -t mangovision-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 mangovision-api
```
