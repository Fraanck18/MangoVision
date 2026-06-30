"""
Microservicio REST de diagnostico en tiempo real para integracion en la faja
transportadora industrial. Intenta cargar el modelo ResNet50 de produccion
desde el Model Registry de MLflow al arrancar; si no hay servidor MLflow
disponible (entorno local/demo) cae a los pesos locales entrenados por
`scripts/train_resnet50.py`. Expone inferencia sincrona sobre imagenes
termicas individuales y el estado del entrenamiento por epocas en curso.
"""
import csv
import io
import json
import logging
import subprocess
import sys
import threading
import time
from io import StringIO
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from src.config import (
    HDFS_NAMENODE,
    INTERVALO_INGESTA_SEGUNDOS,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_STAGE,
    NUM_CLASSES,
    PREFERENCIAS_USUARIO_PATH,
)
from src.inference.diagnostico import diagnosticar_imagen_pil
from src.ingestion.scheduler import activar_fallo_sensor, iniciar_scheduler_ingesta
from src.models.resnet50_custom import (
    ResNet50MangoQC,
    cargar_modelo_mlflow_registry,
    obtener_transformaciones_validacion,
)
from src.pipeline.historial_entrenamiento import agregar_a_historial, escribir_estado, leer_estado, leer_historial
from src.pipeline.worker_inference import analizar_lotes_pendientes, listar_lotes_pendientes
from src.streaming.evaluador_online import calcular_metricas_vivo, iniciar_evaluador_online

logger = logging.getLogger("mangovision.api")

RAIZ = Path(__file__).resolve().parent.parent.parent
PESOS_LOCALES = RAIZ / "dataset_mangos" / "resnet50_mango_qc_multiclase.pt"
SCRIPT_ENTRENAMIENTO = RAIZ / "scripts" / "train_resnet50.py"
PREFERENCIAS_USUARIO = Path(PREFERENCIAS_USUARIO_PATH)

app = FastAPI(
    title="MangoVision QC API",
    description="Diagnostico no destructivo de calidad/madurez de mango via termografia + ResNet50",
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
entrenamiento_en_curso = {"activo": False, "proceso": None, "detenido_manualmente": False}


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
    modelo = ResNet50MangoQC(num_classes=NUM_CLASSES)
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
    """
    Lanza el script de entrenamiento como subproceso real (no `subprocess.run`
    bloqueante) para poder conservar una referencia (`Popen`) y terminarlo
    desde `/entrenamiento/detener` si el usuario lo pide a mitad de corrida.
    """
    proceso = subprocess.Popen([sys.executable, str(SCRIPT_ENTRENAMIENTO)], cwd=str(RAIZ))
    entrenamiento_en_curso["proceso"] = proceso
    try:
        codigo_salida = proceso.wait()
        if codigo_salida != 0 and not entrenamiento_en_curso["detenido_manualmente"]:
            logger.error("El proceso de entrenamiento termino con codigo %s.", codigo_salida)
    finally:
        entrenamiento_en_curso["activo"] = False
        entrenamiento_en_curso["proceso"] = None
        entrenamiento_en_curso["detenido_manualmente"] = False


@app.post("/entrenamiento/iniciar")
def iniciar_entrenamiento(background_tasks: BackgroundTasks):
    with entrenamiento_lock:
        if entrenamiento_en_curso["activo"]:
            raise HTTPException(status_code=409, detail="Ya hay un entrenamiento por epocas en curso.")
        entrenamiento_en_curso["activo"] = True

    # Estado transitorio escrito de inmediato (antes de que el subproceso
    # siquiera termine de arrancar e importar torch). Sin esto hay una
    # ventana de varios segundos en la que /entrenamiento/estado seguiria
    # mostrando "completado" de la corrida ANTERIOR, y si el usuario pulsa
    # Detener en ese instante, /entrenamiento/detener registraria esa corrida
    # vieja en el historial mal etiquetada como "detenido".
    escribir_estado({"estado": "iniciando", "inicio": time.time()})

    background_tasks.add_task(_ejecutar_entrenamiento)
    return {"estado": "iniciado"}


@app.post("/entrenamiento/detener")
def detener_entrenamiento():
    """
    Termina el subproceso de entrenamiento en curso (SIGTERM). El estado
    queda registrado como "detenido" tanto en estado_entrenamiento.json como
    en el historial append-only, para que quede claro que esa corrida no
    completo sus 3 candidatos por intervencion manual, no por error.
    """
    with entrenamiento_lock:
        proceso = entrenamiento_en_curso["proceso"]
        if not entrenamiento_en_curso["activo"] or proceso is None:
            raise HTTPException(status_code=409, detail="No hay un entrenamiento en curso para detener.")
        entrenamiento_en_curso["detenido_manualmente"] = True
        proceso.terminate()

    estado_previo = leer_estado()
    resumen_detenido = {**estado_previo, "estado": "detenido", "fin": time.time()}
    agregar_a_historial(resumen_detenido)
    return {"estado": "detenido"}


@app.get("/entrenamiento/estado")
def estado_entrenamiento():
    return leer_estado()


@app.get("/entrenamiento/historial")
def historial_entrenamiento():
    """Todas las corridas pasadas (completadas o detenidas), no solo la mas reciente."""
    return {"corridas": leer_historial()}


@app.get("/entrenamiento/reporte.csv")
def exportar_reporte_csv():
    """Reporte de clasificacion (precision/recall/f1/support por clase) de la corrida actual, como CSV descargable."""
    estado = leer_estado()
    reporte = estado.get("reporte_test")
    if not reporte:
        raise HTTPException(status_code=404, detail="No hay un reporte de clasificacion disponible todavia.")

    filas = _filas_reporte(reporte)
    columnas = sorted({columna for fila in filas for columna in fila if columna != "clase"})

    buffer = StringIO()
    escritor = csv.writer(buffer)
    escritor.writerow(["clase", *columnas])
    for fila in filas:
        escritor.writerow([fila["clase"], *[fila.get(columna, "") for columna in columnas]])
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=reporte_clasificacion.csv"},
    )


_METRICAS_REPORTE = {"precision", "recall", "f1-score", "support"}


def _filas_reporte(reporte):
    """
    `reporte_test` puede venir en dos orientaciones segun que rama de
    `generar_reporte` (en train_resnet50.py) lo produjo:
      - {metrica: {clase: valor}}  (pandas DataFrame.to_dict() por defecto)
      - {clase: {metrica: valor}}  (sklearn classification_report output_dict)
    Se normalizan ambas a una lista de filas {clase, **metricas}.
    """
    if set(reporte.keys()) <= _METRICAS_REPORTE:
        clases = sorted({clase for metrica in reporte.values() for clase in metrica})
        return [
            {"clase": clase, **{metrica: reporte[metrica].get(clase, "") for metrica in reporte}}
            for clase in clases
        ]
    return [{"clase": clase, **metricas} for clase, metricas in reporte.items()]


@app.get("/config/preferencias")
def obtener_preferencias():
    """Preferencias de UI persistidas en backend (tema, etc.), no solo en localStorage del navegador actual."""
    if not PREFERENCIAS_USUARIO.exists():
        return {}
    return json.loads(PREFERENCIAS_USUARIO.read_text(encoding="utf-8"))


@app.post("/config/preferencias")
def guardar_preferencias(preferencias: dict):
    PREFERENCIAS_USUARIO.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCIAS_USUARIO.write_text(json.dumps(preferencias, ensure_ascii=False, indent=2), encoding="utf-8")
    return preferencias


@app.get("/hdfs/estado")
def estado_hdfs():
    """Lotes que el sensor del camion ya aterrizo en el Data Lake y que el boton Analizar aun no proceso."""
    pendientes = listar_lotes_pendientes()
    return {
        "intervalo_ingesta_s": INTERVALO_INGESTA_SEGUNDOS,
        "lotes_pendientes": pendientes,
        "total_lotes_pendientes": len(pendientes),
        "ruta_data_lake": HDFS_NAMENODE,
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
