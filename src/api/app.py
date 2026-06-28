"""
Microservicio REST de diagnostico en tiempo real para integracion en la faja
transportadora industrial. Intenta cargar el modelo ResNet18 de produccion
desde el Model Registry de MLflow al arrancar; si no hay servidor MLflow
disponible (entorno local/demo) cae a los pesos locales entrenados por
`scripts/train_resnet18.py`. Expone inferencia sincrona sobre imagenes
termicas individuales y el estado del entrenamiento por epocas en curso.
"""
import io
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.config import HDFS_RAW_PATH_LOCAL, INTERVALO_INGESTA_SEGUNDOS, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE, NUM_CLASSES
from src.inference.diagnostico import diagnosticar_imagen_pil
from src.ingestion.scheduler import activar_fallo_sensor, iniciar_scheduler_ingesta
from src.models.resnet18_custom import (
    ResNet18MangoQC,
    cargar_modelo_mlflow_registry,
    obtener_transformaciones_validacion,
)
from src.pipeline.worker_inference import analizar_lotes_pendientes, listar_lotes_pendientes
from src.streaming.evaluador_online import calcular_metricas_vivo, iniciar_evaluador_online

logger = logging.getLogger("mangovision.api")

RAIZ = Path(__file__).resolve().parent.parent.parent
PESOS_LOCALES = RAIZ / "dataset_mangos" / "resnet18_mango_qc_multiclase.pt"
ESTADO_ENTRENAMIENTO = RAIZ / "dataset_mangos" / "estado_entrenamiento.json"
SCRIPT_ENTRENAMIENTO = RAIZ / "scripts" / "train_resnet18.py"

app = FastAPI(
    title="MangoVision QC API",
    description="Diagnostico no destructivo de calidad/madurez de mango via termografia + ResNet18",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

estado_modelo = {"modelo": None, "device": None, "preprocess": None, "origen": None}
entrenamiento_lock = threading.Lock()
entrenamiento_en_curso = {"activo": False}


class DiagnosticoResponse(BaseModel):
    etiqueta: str
    clase_id: int
    confianza: float
    categoria: str
    temperatura_estimada: float
    color: str


def _cargar_pesos_locales(device):
    if not PESOS_LOCALES.exists():
        raise FileNotFoundError(f"No existen pesos locales en {PESOS_LOCALES}")
    modelo = ResNet18MangoQC(num_classes=NUM_CLASSES)
    modelo.load_state_dict(torch.load(PESOS_LOCALES, map_location=device))
    return modelo.to(device)


@app.on_event("startup")
def cargar_modelo_produccion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estado_modelo["device"] = device
    estado_modelo["preprocess"] = obtener_transformaciones_validacion()

    try:
        modelo = cargar_modelo_mlflow_registry(MLFLOW_MODEL_NAME, stage=MLFLOW_MODEL_STAGE).to(device)
        estado_modelo["origen"] = "mlflow_registry"
        logger.info("Modelo %s (%s) cargado desde MLflow en %s", MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE, device)
    except Exception:
        logger.warning("MLflow Registry no disponible, intentando pesos locales.")
        try:
            modelo = _cargar_pesos_locales(device)
            estado_modelo["origen"] = "pesos_locales"
            logger.info("Modelo cargado desde pesos locales: %s", PESOS_LOCALES)
        except Exception:
            logger.exception("No fue posible cargar ningun modelo de produccion.")
            return

    modelo.eval()
    estado_modelo["modelo"] = modelo
    iniciar_scheduler_ingesta()
    iniciar_evaluador_online(estado_modelo)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "modelo_cargado": estado_modelo["modelo"] is not None,
        "origen_modelo": estado_modelo["origen"],
    }


@app.post("/diagnostico", response_model=DiagnosticoResponse)
async def diagnosticar_mango(archivo: UploadFile = File(...)):
    if estado_modelo["modelo"] is None:
        raise HTTPException(status_code=503, detail="El modelo de produccion aun no esta disponible.")

    contenido = await archivo.read()
    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo de imagen invalido.")

    resultado = diagnosticar_imagen_pil(
        imagen, estado_modelo["modelo"], estado_modelo["device"], estado_modelo["preprocess"]
    )
    return DiagnosticoResponse(**resultado)


def _ejecutar_entrenamiento():
    try:
        subprocess.run([sys.executable, str(SCRIPT_ENTRENAMIENTO)], cwd=str(RAIZ), check=True)
    except Exception:
        logger.exception("El proceso de entrenamiento por epocas fallo.")
    finally:
        entrenamiento_en_curso["activo"] = False


@app.post("/entrenamiento/iniciar")
def iniciar_entrenamiento(background_tasks: BackgroundTasks):
    with entrenamiento_lock:
        if entrenamiento_en_curso["activo"]:
            raise HTTPException(status_code=409, detail="Ya hay un entrenamiento por epocas en curso.")
        entrenamiento_en_curso["activo"] = True

    background_tasks.add_task(_ejecutar_entrenamiento)
    return {"estado": "iniciado"}


@app.get("/entrenamiento/estado")
def estado_entrenamiento():
    if not ESTADO_ENTRENAMIENTO.exists():
        return {"estado": "sin_iniciar"}
    return json.loads(ESTADO_ENTRENAMIENTO.read_text(encoding="utf-8"))


@app.get("/hdfs/estado")
def estado_hdfs():
    """Lotes que el sensor del camion ya aterrizo en el Data Lake y que el boton Analizar aun no proceso."""
    pendientes = listar_lotes_pendientes()
    return {
        "intervalo_ingesta_s": INTERVALO_INGESTA_SEGUNDOS,
        "lotes_pendientes": pendientes,
        "total_lotes_pendientes": len(pendientes),
        "ruta_data_lake": str(HDFS_RAW_PATH_LOCAL),
    }


@app.post("/hdfs/analizar")
def analizar_lotes():
    """
    Botón 'Analizar': procesa en paralelo (paralelismo de datos entre
    workers, mismo modelo, sin perdida de precision) todos los lotes
    aterrizados en el Data Lake desde la ultima vez que se llamo este
    endpoint.
    """
    if estado_modelo["modelo"] is None:
        raise HTTPException(status_code=503, detail="El modelo de produccion aun no esta disponible.")

    return analizar_lotes_pendientes(
        estado_modelo["modelo"], estado_modelo["device"], estado_modelo["preprocess"]
    )


@app.get("/metricas/vivo")
def metricas_vivo():
    """
    Metrica de evaluacion online en ventana deslizante (calculada por el daemon
    evaluador desde el stream Kafka). El frontend hace fetch periodico de esto;
    NO calcula nada en el navegador.
    """
    return calcular_metricas_vivo()


@app.post("/simular/fallo-sensor")
def simular_fallo_sensor(lotes: int = 3):
    """
    Demo: simula condensacion en la lente FLIR / perdida de cadena de frio
    durante los proximos N lotes (degradacion alta). La metrica de ventana caera
    y el detector de drift se disparara; luego el sistema se auto-recupera.
    """
    activar_fallo_sensor(lotes)
    return {"estado": "fallo_sensor_activado", "lotes_afectados": lotes}
