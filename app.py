"""
Entrypoint de DESPLIEGUE (Hugging Face Spaces) para MangoVision.

Version "delgada": expone la INFERENCIA real (/diagnostico) + el estado del
modelo (/health) y sirve el frontend estatico de web/. Los endpoints de
entrenamiento, HDFS, Kafka y metricas en vivo se responden como STUBS en
"modo demo" para que el dashboard cargue limpio (sin errores en consola),
pero SIN arrancar ningun servicio pesado: en la nube no existen HDFS,
Kafka, Spark ni MLflow — esos son solo del entorno local de entrenamiento.

Correr local:  uvicorn app:app --host 0.0.0.0 --port 7860
En HF Spaces:  el Dockerfile ya ejecuta ese comando.
"""
import io
import logging
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.config import CLASS_NAMES, MATURITY_INFO, NUM_CLASSES
from src.inference.diagnostico import diagnosticar_imagen_pil
from src.models.resnet50_custom import (
    cargar_pesos_arquitectura_consciente,
    obtener_transformaciones_validacion,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mangovision.deploy")

RAIZ = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(RAIZ / "best_model.pth")))
WEB_DIR = RAIZ / "web"

app = FastAPI(
    title="MangoVision QC API (deploy)",
    description="Diagnostico no destructivo de calidad/madurez de mango via termografia + ResNet50",
    version="1.1.0-deploy",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

estado_modelo = {"modelo": None, "device": None, "preprocess": None, "origen": None}
# Preferencias en memoria (el disco de HF es efimero; para persistencia real se
# conectaria MongoDB Atlas mas adelante). Sirve para que el selector de tema no falle.
_preferencias = {}


class DiagnosticoResponse(BaseModel):
    etiqueta: str
    clase_id: int
    confianza: float
    categoria: str
    indice_itr: float
    unidad: str
    color: str


@app.on_event("startup")
def cargar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estado_modelo["device"] = device
    estado_modelo["preprocess"] = obtener_transformaciones_validacion()

    if not MODEL_PATH.exists():
        logger.error("No se encontro el modelo en %s", MODEL_PATH)
        return
    try:
        modelo = cargar_pesos_arquitectura_consciente(MODEL_PATH, device, num_classes=NUM_CLASSES)
        modelo.eval()
        estado_modelo["modelo"] = modelo
        estado_modelo["origen"] = "pesos_locales"
        logger.info("Modelo cargado desde %s en %s", MODEL_PATH, device)
    except Exception:
        logger.exception("No fue posible cargar el modelo.")


# ---------------------------------------------------------------------------
# NUCLEO REAL: inferencia sincrona sobre una imagen individual.
# ---------------------------------------------------------------------------
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
        raise HTTPException(status_code=503, detail="El modelo aun no esta disponible.")
    contenido = await archivo.read()
    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo de imagen invalido.")
    resultado = diagnosticar_imagen_pil(
        imagen, estado_modelo["modelo"], estado_modelo["device"], estado_modelo["preprocess"]
    )
    return DiagnosticoResponse(**resultado)


@app.get("/config/escala")
def obtener_escala():
    """Fuente unica de la escala ITR para el frontend (identica al backend real)."""
    return {
        "unidad": "ITR (relativo 0-100, no °C)",
        "rango": [0, 100],
        "clases": {
            clase: {
                "etiqueta": MATURITY_INFO[clase]["etiqueta"],
                "banda_itr": list(MATURITY_INFO[clase]["banda_itr"]),
                "color": MATURITY_INFO[clase]["color"],
            }
            for clase in CLASS_NAMES
        },
    }


@app.get("/config/preferencias")
def obtener_preferencias():
    return _preferencias


@app.post("/config/preferencias")
def guardar_preferencias(preferencias: dict):
    _preferencias.update(preferencias)
    return {"estado": "ok", "preferencias": _preferencias}


# ---------------------------------------------------------------------------
# STUBS "MODO DEMO": mantienen el dashboard limpio sin infra pesada (HDFS/Kafka/
# entrenamiento). Devuelven respuestas neutras/vacias, nunca arrancan servicios.
# ---------------------------------------------------------------------------
@app.get("/entrenamiento/estado")
def estado_entrenamiento_demo():
    return {"estado": "sin_iniciar"}


@app.get("/entrenamiento/historial")
def historial_entrenamiento_demo():
    return {"corridas": []}


@app.get("/hdfs/estado")
def estado_hdfs_demo():
    return {"total_lotes_pendientes": 0, "intervalo_ingesta_s": 300}


@app.post("/hdfs/analizar")
def analizar_hdfs_demo():
    return {
        "lotes_procesados": [],
        "resultados": [],
        "tiempo_total_s": 0,
        "tiempo_secuencial_estimado_s": 0,
        "num_workers": 1,
    }


@app.get("/metricas/vivo")
def metricas_vivo_demo():
    # El frontend interpreta null como "sin ventana viva" y usa placeholders.
    return None


@app.get("/imagenes/lotes")
def lotes_imagenes_demo():
    return {"lotes": []}


_MENSAJE_DEMO = "Funcion no disponible en la demo desplegada (requiere la infraestructura local: GPU/Spark/HDFS/Kafka)."


@app.post("/entrenamiento/iniciar")
def iniciar_entrenamiento_demo():
    raise HTTPException(status_code=503, detail=_MENSAJE_DEMO)


@app.post("/entrenamiento/detener")
def detener_entrenamiento_demo():
    raise HTTPException(status_code=503, detail=_MENSAJE_DEMO)


@app.post("/simular/fallo-sensor")
def simular_fallo_demo(lotes: int = 3):
    raise HTTPException(status_code=503, detail=_MENSAJE_DEMO)


# ---------------------------------------------------------------------------
# Frontend estatico. Se monta al FINAL para que las rutas de API tengan
# prioridad sobre el catch-all de archivos.
# ---------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    logger.warning("No existe la carpeta web/ en %s; el frontend no se servira.", WEB_DIR)
