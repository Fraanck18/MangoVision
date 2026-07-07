"""
Pseudo-etiquetado semi-supervisado de las 309 imagenes de
`Thermal Mango Image Dataset - FLIR One` (sin clase asignada).

Usa el ResNet18 ya entrenado sobre el dataset etiquetado real para predecir
una clase en cada imagen de FLIR, y solo acepta las predicciones cuya
confianza (probabilidad softmax maxima) supere `UMBRAL_CONFIANZA`. Esto evita
que el ruido de un modelo todavia imperfecto contamine el pool de
entrenamiento: es preferible descartar una imagen ambigua que pseudo-
etiquetarla mal y reforzar un error.

Uso:
    python scripts/pseudo_label_flir.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from PIL import Image

from src.config import CLASS_NAMES
from src.models.resnet50_custom import (
    cargar_pesos_arquitectura_consciente,
    obtener_transformaciones_validacion,
)

CARPETA_FLIR = RAIZ / "dataset_mangos" / "extracted" / "flir_one" / "Thermal Mango Image Dataset - FLIR One" / "Thermal"
PESOS_MODELO = RAIZ / "dataset_mangos" / "best_model.pth"   # ResNet50 (modelo unico)
SALIDA_PSEUDO_ETIQUETAS = RAIZ / "dataset_mangos" / "flir_pseudo_labels.json"

# [Plan de Contingencia F5] Filtro de confianza DRASTICO. Antes 0.90 dejaba pasar
# demasiado ruido y el pseudo-etiquetado se auto-reforzaba, acumulando casi todo en
# `alta_calidad`. Se sube a 0.98: solo se acepta una pseudo-etiqueta si la red esta
# practicamente segura. ADEMAS es OBLIGATORIA una auditoria manual de las imagenes
# aceptadas como `alta_calidad` (ver aviso al final). Si el desbalance persiste, se
# corrige aguas abajo con undersampling de la clase mayoritaria o WeightedCrossEntropy
# (peso inverso a la frecuencia), no aceptando mas ruido aqui.
UMBRAL_CONFIANZA = 0.98


def cargar_modelo_entrenado():
    # Detecta el backbone desde el checkpoint (no lo asume) y carga defensivamente.
    return cargar_pesos_arquitectura_consciente(PESOS_MODELO, torch.device("cpu"))


def pseudo_etiquetar():
    modelo = cargar_modelo_entrenado()
    preprocess = obtener_transformaciones_validacion()

    rutas_imagenes = sorted(CARPETA_FLIR.glob("*"))
    print(f"Imagenes FLIR encontradas: {len(rutas_imagenes)}")

    aceptadas, rechazadas = [], []
    confianzas_por_clase = Counter()

    with torch.no_grad():
        for ruta in rutas_imagenes:
            imagen = Image.open(ruta).convert("RGB")

            tensor = preprocess(imagen).unsqueeze(0)
            logits = modelo(tensor)
            probabilidades = torch.softmax(logits, dim=1)
            confianza, clase_id = torch.max(probabilidades, dim=1)
            confianza, clase_id = confianza.item(), clase_id.item()

            registro = {
                "ruta_origen": str(ruta.relative_to(RAIZ)).replace("\\", "/"),
                "clase_predicha": CLASS_NAMES[clase_id],
                "confianza": round(confianza, 4),
            }

            if confianza >= UMBRAL_CONFIANZA:
                aceptadas.append(registro)
                confianzas_por_clase[CLASS_NAMES[clase_id]] += 1
            else:
                rechazadas.append(registro)

    print(f"\nUmbral de confianza: {UMBRAL_CONFIANZA}")
    print(f"Aceptadas (>= umbral): {len(aceptadas)} / {len(rutas_imagenes)}")
    print(f"Rechazadas (< umbral, descartadas): {len(rechazadas)}")
    print("Distribucion de clases aceptadas:", dict(confianzas_por_clase))

    # [F5] Aviso de auditoria manual obligatoria + deteccion de desbalance.
    total_aceptadas = sum(confianzas_por_clase.values())
    if total_aceptadas:
        clase_mayor, n_mayor = confianzas_por_clase.most_common(1)[0]
        if n_mayor / total_aceptadas > 0.6:
            print(
                f"\n[F5][ADVERTENCIA] Desbalance: '{clase_mayor}' concentra "
                f"{n_mayor}/{total_aceptadas} ({n_mayor/total_aceptadas:.0%}) de las pseudo-etiquetas. "
                f"AUDITAR MANUALMENTE esas imagenes antes de reentrenar; si el balance no se "
                f"puede corregir, aplicar undersampling o WeightedCrossEntropy en el entrenamiento."
            )

    with open(SALIDA_PSEUDO_ETIQUETAS, "w") as f:
        json.dump(aceptadas, f, indent=2, ensure_ascii=False)
    print(f"\nPseudo-etiquetas aceptadas guardadas en: {SALIDA_PSEUDO_ETIQUETAS}")

    return aceptadas, rechazadas


if __name__ == "__main__":
    pseudo_etiquetar()
