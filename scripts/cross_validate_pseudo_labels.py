"""
Pseudo-etiquetado de FLIR por acuerdo entre dos modelos independientes
(ResNet18 + HOG-LinearSVC), en vez de confiar en el softmax de un solo
modelo (que demostro estar mal calibrado para `calidad_promedio`).

Solo se acepta una pseudo-etiqueta cuando AMBOS modelos coinciden en la
misma clase. Como tienen representaciones matematicas distintas (jerarquias
aprendidas vs. gradientes de borde), su coincidencia es evidencia mucho mas
fuerte que la confianza interna de uno solo.

Uso:
    python scripts/cross_validate_pseudo_labels.py
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, IMAGE_SIZE
from src.models.resnet18_custom import ResNet18MangoQC, obtener_transformaciones_validacion
from src.models.baseline_svm import extraer_descriptor_hog

CARPETA_FLIR = RAIZ / "dataset_mangos" / "extracted" / "flir_one" / "Thermal Mango Image Dataset - FLIR One" / "Thermal"
PESOS_RESNET = RAIZ / "dataset_mangos" / "resnet18_mango_qc.pt"
PESOS_SVM = RAIZ / "dataset_mangos" / "baseline_svm.pkl"
SALIDA = RAIZ / "dataset_mangos" / "flir_pseudo_labels.json"


def predecir_resnet(rutas):
    modelo = ResNet18MangoQC()
    modelo.load_state_dict(torch.load(PESOS_RESNET, map_location="cpu"))
    modelo.eval()
    preprocess = obtener_transformaciones_validacion()

    predicciones = []
    with torch.no_grad():
        for ruta in rutas:
            imagen = Image.open(ruta).convert("RGB")
            tensor = preprocess(imagen).unsqueeze(0)
            logits = modelo(tensor)
            predicciones.append(torch.argmax(logits, dim=1).item())
    return predicciones


def predecir_svm(rutas):
    with open(PESOS_SVM, "rb") as f:
        pipeline = pickle.load(f)

    descriptores = []
    for ruta in rutas:
        imagen = Image.open(ruta).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        descriptores.append(extraer_descriptor_hog(imagen))
    X = np.stack(descriptores)
    return pipeline.predict(X).tolist()


def main():
    rutas = sorted(CARPETA_FLIR.glob("*"))
    print(f"Imagenes FLIR a evaluar: {len(rutas)}")

    print("Prediciendo con ResNet18...")
    pred_resnet = predecir_resnet(rutas)
    print("Prediciendo con HOG-SVM...")
    pred_svm = predecir_svm(rutas)

    aceptadas = []
    acuerdos_por_clase = Counter()
    for ruta, clase_resnet, clase_svm in zip(rutas, pred_resnet, pred_svm):
        if clase_resnet == clase_svm:
            nombre_clase = CLASS_NAMES[clase_resnet]
            acuerdos_por_clase[nombre_clase] += 1
            aceptadas.append({
                "ruta_origen": str(ruta.relative_to(RAIZ)).replace("\\", "/"),
                "clase_predicha": nombre_clase,
                "metodo": "acuerdo_resnet18_hogsvm",
            })

    print(f"\nAcuerdo entre ambos modelos: {len(aceptadas)} / {len(rutas)}")
    print("Distribucion de clases aceptadas (por acuerdo):", dict(acuerdos_por_clase))
    print(f"Desacuerdo (descartadas): {len(rutas) - len(aceptadas)}")

    with open(SALIDA, "w") as f:
        json.dump(aceptadas, f, indent=2, ensure_ascii=False)
    print(f"\nPseudo-etiquetas por acuerdo guardadas en: {SALIDA}")
    return aceptadas


if __name__ == "__main__":
    main()
