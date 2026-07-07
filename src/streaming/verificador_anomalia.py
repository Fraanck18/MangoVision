"""
Verificacion de ANOMALIA en el stream, PASO PREVIO a la clasificacion.

Flujo streaming (tiempo real, imagen a imagen): llega -> VERIFICA -> clasifica.
Este modulo es el "verifica": puntua cuanto se desvia la firma termica del fruto
respecto de lo NORMAL, antes de que ResNet50 clasifique. Es la capa LIGERA de
verificacion (features PEDTE + una regla), no reemplaza al clasificador.

Honestidad (docs/INTEGRIDAD.md):
  * "Normal" se define por la etiqueta HUMANA de calidad (`alta_calidad`), criterio
    externo a la imagen -> no circular.
  * Cada verificacion trae SIEMPRE el score del BASELINE TRIVIAL de brillo al lado
    (guardarrail): en el dato real el trivial separa mejor (AUC 0.87) que el
    detector PEDTE (0.65), y se reporta tal cual.
"""
import logging
from pathlib import Path

from PIL import Image

from src.anomaly import vector_pedte, itr_medio_imagen, DetectorAnomalias, ScoreTrivialBrillo

logger = logging.getLogger("mangovision.verificador")

CLASE_NORMAL = "alta_calidad"
CARPETA_NORMAL = Path("dataset_mangos/processed/train") / CLASE_NORMAL


def entrenar_verificador(carpeta_normal=CARPETA_NORMAL, metodo="mahalanobis"):
    """
    Ajusta el modelo de NORMALIDAD (detector one-class + baseline trivial) con las
    firmas PEDTE de los mangos buenos REALES. Devuelve un dict verificador o None si
    no hay suficientes imagenes normales.
    """
    carpeta = Path(carpeta_normal)
    rutas = [r for r in carpeta.glob("*") if r.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(rutas) < 8:
        logger.warning("Verificador de anomalia: solo %d imagenes normales, insuficiente.", len(rutas))
        return None

    X, itr = [], []
    for r in rutas:
        img = Image.open(r).convert("RGB")
        X.append(vector_pedte(img))
        itr.append(itr_medio_imagen(img))

    detector = DetectorAnomalias(metodo=metodo).fit(X)
    trivial = ScoreTrivialBrillo().fit(itr)
    logger.info("Verificador de anomalia entrenado con %d mangos normales reales.", len(rutas))
    return {"detector": detector, "trivial": trivial, "n_normales": len(rutas)}


def verificar(verificador, imagen_pil):
    """
    Verifica una imagen: devuelve score de anomalia del detector PEDTE Y del baseline
    trivial de brillo (siempre ambos, guardarrail). None-safe: si no hay verificador
    entrenado, devuelve un dict con banderas False.
    """
    if verificador is None:
        return {"verificado": False, "es_anomalia": False, "es_anomalia_trivial": False,
                "score_anomalia": None, "score_trivial": None}

    x = vector_pedte(imagen_pil)
    itr = itr_medio_imagen(imagen_pil)
    det, triv = verificador["detector"], verificador["trivial"]
    return {
        "verificado": True,
        "score_anomalia": float(det.score([x])[0]),
        "es_anomalia": bool(det.predict_anomalia([x])[0]),
        "score_trivial": float(triv.score([itr])[0]),
        "es_anomalia_trivial": bool(triv.predict_anomalia([itr])[0]),
    }
