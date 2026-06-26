"""
Entrenamiento del baseline HOG + LinearSVC sobre el mismo dataset real y
split sin fuga usado para ResNet18 (`dataset_mangos/processed/`).

Sirve dos propositos: (1) referencia clasica de comparacion contra la red
profunda, y (2) segundo verificador independiente para el pseudo-etiquetado
de FLIR (ver scripts/cross_validate_pseudo_labels.py) — al tener una
representacion matematica distinta (gradientes de borde, no jerarquias
aprendidas), su acuerdo con ResNet18 es evidencia mas fuerte que la
confianza de un solo modelo.

Uso:
    python scripts/train_baseline_svm.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, IMAGE_SIZE
from src.models.baseline_svm import extraer_descriptor_hog, construir_pipeline_baseline

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
PESOS_SALIDA = RAIZ / "dataset_mangos" / "baseline_svm.pkl"


def cargar_split(nombre_split):
    """Recorre dataset_mangos/processed/<split>/<clase>/*.jpg y arma X (HOG) e y (clase)."""
    rutas, etiquetas = [], []
    for clase in CLASS_NAMES:
        carpeta_clase = PROCESADOS / nombre_split / clase
        for ruta in sorted(carpeta_clase.glob("*")):
            rutas.append(ruta)
            etiquetas.append(CLASS_NAMES.index(clase))

    descriptores = []
    for ruta in rutas:
        imagen = Image.open(ruta).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        descriptores.append(extraer_descriptor_hog(imagen))

    return np.stack(descriptores), np.array(etiquetas)


def main():
    print("Extrayendo descriptores HOG de train...")
    X_train, y_train = cargar_split("train")
    print(f"Extrayendo descriptores HOG de test... (train: {X_train.shape[0]} muestras)")
    X_test, y_test = cargar_split("test")
    print(f"Test: {X_test.shape[0]} muestras")

    pipeline = construir_pipeline_baseline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"\nAccuracy baseline HOG+SVM en test real: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    with open(PESOS_SALIDA, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline guardado en: {PESOS_SALIDA}")


if __name__ == "__main__":
    main()
