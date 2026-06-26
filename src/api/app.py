"""
Microservicio REST de diagnostico en tiempo real para integracion en la faja
transportadora industrial. Descarga el modelo ResNet18 de produccion desde el
Model Registry de MLflow al arrancar el contenedor (`startup event`) y expone
un endpoint de inferencia sincrona sobre imagenes termicas individuales.
"""
import io
import logging

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.config import CLASS_NAMES, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE
from src.models.resnet18_custom import cargar_modelo_mlflow_registry, obtener_transformaciones_validacion

logger = logging.getLogger("mangovision.api")

app = FastAPI(
    title="MangoVision QC API",
    description="Diagnostico no destructivo de defectos internos en mango via termografia + ResNet18",
    version="1.0.0",
)

estado_modelo = {"modelo": None, "device": None, "preprocess": None}


class DiagnosticoResponse(BaseModel):
    etiqueta: str
    clase_id: int
    confianza: float


@app.on_event("startup")
def cargar_modelo_produccion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        modelo = cargar_modelo_mlflow_registry(MLFLOW_MODEL_NAME, stage=MLFLOW_MODEL_STAGE).to(device)
        modelo.eval()
        estado_modelo["modelo"] = modelo
        estado_modelo["device"] = device
        estado_modelo["preprocess"] = obtener_transformaciones_validacion()
        logger.info("Modelo %s (%s) cargado en %s", MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE, device)
    except Exception:
        logger.exception("No fue posible cargar el modelo de produccion desde MLflow Registry.")


@app.get("/health")
def health_check():
    return {"status": "ok", "modelo_cargado": estado_modelo["modelo"] is not None}


@app.post("/diagnostico", response_model=DiagnosticoResponse)
async def diagnosticar_mango(archivo: UploadFile = File(...)):
    if estado_modelo["modelo"] is None:
        raise HTTPException(status_code=503, detail="El modelo de produccion aun no esta disponible.")

    contenido = await archivo.read()
    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo de imagen invalido.")

    tensor = estado_modelo["preprocess"](imagen).unsqueeze(0).to(estado_modelo["device"])

    with torch.no_grad():
        logits = estado_modelo["modelo"](tensor)
        probabilidades = torch.softmax(logits, dim=1)
        confianza, clase_id = torch.max(probabilidades, dim=1)

    return DiagnosticoResponse(
        etiqueta=CLASS_NAMES[clase_id.item()],
        clase_id=clase_id.item(),
        confianza=round(confianza.item(), 4),
    )
