"""
Aplica `extraer_anomalia_termica_imagen` (Ley de Newton + decodificacion de
paleta Ironbow) sobre el dataset binario real, y evalua si esas features
fisicas por si solas separan `sano` de `defectuoso` -- sin usar ResNet18 ni
HOG, para medir el aporte de la fisica de forma aislada.

Uso:
    python scripts/extract_thermal_features.py
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.domain_physics.thermal_inertia import extraer_anomalia_termica_imagen

PROCESADOS = RAIZ / "dataset_mangos" / "processed_binary"
SALIDA_CSV = RAIZ / "dataset_mangos" / "features_termicas.csv"


def recolectar_features():
    filas = []
    for particion in ("train", "test"):
        for clase in ("sano", "defectuoso"):
            carpeta = PROCESADOS / particion / clase
            if not carpeta.exists():
                continue
            for ruta in sorted(carpeta.glob("*")):
                imagen = Image.open(ruta).convert("RGB")
                features = extraer_anomalia_termica_imagen(imagen)
                features.update({"particion": particion, "clase": clase, "archivo": ruta.name})
                filas.append(features)
    return pd.DataFrame(filas)


def main():
    df = recolectar_features()
    df.to_csv(SALIDA_CSV, index=False)
    print(f"Features extraidas: {len(df)} imagenes -> {SALIDA_CSV}")

    print(f"\nFruto detectado en: {df['fruto_detectado'].sum()} / {len(df)} imagenes")

    print("\nEstadisticas de fraccion_area_anomala_baja por clase (particion=train):")
    print(df[df.particion == "train"].groupby("clase")["fraccion_area_anomala_baja"].describe())

    print("\nEstadisticas de desviacion_maxima_baja por clase (particion=train):")
    print(df[df.particion == "train"].groupby("clase")["desviacion_maxima_baja"].describe())


if __name__ == "__main__":
    main()
