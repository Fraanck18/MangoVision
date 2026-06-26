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
from src.models.resnet18_custom import ResNet18MangoQC, obtener_transformaciones_validacion

CARPETA_FLIR = RAIZ / "dataset_mangos" / "extracted" / "flir_one" / "Thermal Mango Image Dataset - FLIR One" / "Thermal"
PESOS_MODELO = RAIZ / "dataset_mangos" / "resnet18_mango_qc.pt"
SALIDA_PSEUDO_ETIQUETAS = RAIZ / "dataset_mangos" / "flir_pseudo_labels.json"

UMBRAL_CONFIANZA = 0.90


def cargar_modelo_entrenado():
    modelo = ResNet18MangoQC()
    modelo.load_state_dict(torch.load(PESOS_MODELO, map_location="cpu"))
    modelo.eval()
    return modelo


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

    with open(SALIDA_PSEUDO_ETIQUETAS, "w") as f:
        json.dump(aceptadas, f, indent=2, ensure_ascii=False)
    print(f"\nPseudo-etiquetas aceptadas guardadas en: {SALIDA_PSEUDO_ETIQUETAS}")

    return aceptadas, rechazadas


if __name__ == "__main__":
    pseudo_etiquetar()
