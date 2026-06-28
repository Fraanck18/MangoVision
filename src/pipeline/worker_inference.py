"""
Inferencia en paralelo sobre los lotes pendientes en el Data Lake (HDFS
simulado en disco, ver src/ingestion/scheduler.py).

Paralelismo de datos, no de modelo: todos los workers comparten la MISMA
instancia ya cargada del modelo entrenado (PyTorch libera el GIL durante las
operaciones C++ del forward pass, por lo que varios hilos sí corren en
paralelo real sobre CPU) y cada uno procesa un subconjunto distinto de
imagenes. No hay promedio de gradientes ni aproximacion: cada imagen pasa por
exactamente el mismo softmax que en `/diagnostico`, solo que repartidas entre
N workers para reducir el tiempo de pared total. Esto es deliberadamente el
mismo enfoque que `src/pipeline/spark_inference.py` (data parallelism via
mapInPandas), pero ejecutable sin un cluster Spark/HDFS real.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.config import HDFS_RAW_PATH_LOCAL, LOTES_PROCESADOS_PATH, NUM_WORKERS_INFERENCIA
from src.inference.diagnostico import diagnosticar_imagen_pil

logger = logging.getLogger("mangovision.workers")


def _lotes_procesados():
    ruta = Path(LOTES_PROCESADOS_PATH)
    if not ruta.exists():
        return set()
    return set(json.loads(ruta.read_text(encoding="utf-8")))


def _marcar_procesados(lotes_nuevos):
    ruta = Path(LOTES_PROCESADOS_PATH)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    actuales = _lotes_procesados() | set(lotes_nuevos)
    ruta.write_text(json.dumps(sorted(actuales), ensure_ascii=False, indent=2), encoding="utf-8")


def listar_lotes_pendientes():
    """Carpetas lote_ingesta=* aterrizadas en el Data Lake que aun no pasaron por inferencia."""
    raiz = Path(HDFS_RAW_PATH_LOCAL)
    if not raiz.exists():
        return []
    procesados = _lotes_procesados()
    carpetas = sorted(p.name for p in raiz.iterdir() if p.is_dir() and p.name.startswith("lote_ingesta="))
    return [c for c in carpetas if c not in procesados]


def _imagenes_de_lote(nombre_lote):
    carpeta = Path(HDFS_RAW_PATH_LOCAL) / nombre_lote
    items = []
    for ruta_metadata in sorted(carpeta.glob("*.json")):
        ruta_imagen = ruta_metadata.with_suffix(".jpg")
        if ruta_imagen.exists():
            metadata = json.loads(ruta_metadata.read_text(encoding="utf-8"))
            items.append({"ruta_imagen": ruta_imagen, "metadata": metadata, "lote_ingesta": nombre_lote})
    return items


def _procesar_chunk(chunk, modelo, device, preprocess):
    resultados = []
    for item in chunk:
        inicio = time.perf_counter()
        try:
            imagen = Image.open(item["ruta_imagen"]).convert("RGB")
        except Exception:
            logger.warning("Imagen ilegible en %s, se omite.", item["ruta_imagen"])
            continue
        diagnostico = diagnosticar_imagen_pil(imagen, modelo, device, preprocess)
        duracion = time.perf_counter() - inicio

        diagnostico.update({
            "lote_ingesta": item["lote_ingesta"],
            "sensor_id": item["metadata"].get("sensor_id"),
            "clase_real_origen": item["metadata"].get("clase_real_origen"),
            "aumentado": item["metadata"].get("aumentado", "original"),
            "tiempo_inferencia_s": round(duracion, 4),
        })
        resultados.append(diagnostico)
    return resultados


def analizar_lotes_pendientes(modelo, device, preprocess, num_workers=None):
    """
    Punto de entrada del boton "Analizar": procesa TODOS los lotes que hayan
    aterrizado en HDFS desde la ultima vez que se llamo esta funcion,
    repartidos entre `num_workers` hilos en paralelo.
    """
    num_workers = num_workers or NUM_WORKERS_INFERENCIA
    lotes_pendientes = listar_lotes_pendientes()

    if not lotes_pendientes:
        return {
            "lotes_procesados": [],
            "resultados": [],
            "num_workers": num_workers,
            "tiempo_total_s": 0.0,
            "tiempo_secuencial_estimado_s": 0.0,
        }

    items = [item for lote in lotes_pendientes for item in _imagenes_de_lote(lote)]
    chunks = [items[i::num_workers] for i in range(num_workers) if items[i::num_workers]]

    inicio_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futuros = [executor.submit(_procesar_chunk, chunk, modelo, device, preprocess) for chunk in chunks]
        resultados = [r for futuro in futuros for r in futuro.result()]
    tiempo_total = time.perf_counter() - inicio_total

    tiempo_secuencial_estimado = sum(r["tiempo_inferencia_s"] for r in resultados)

    _marcar_procesados(lotes_pendientes)
    logger.info(
        "Procesados %d lotes (%d imagenes) con %d workers en %.2fs (secuencial estimado: %.2fs).",
        len(lotes_pendientes), len(resultados), len(chunks), tiempo_total, tiempo_secuencial_estimado,
    )

    return {
        "lotes_procesados": lotes_pendientes,
        "resultados": resultados,
        "num_workers": len(chunks),
        "tiempo_total_s": round(tiempo_total, 4),
        "tiempo_secuencial_estimado_s": round(tiempo_secuencial_estimado, 4),
    }
