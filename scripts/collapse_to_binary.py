"""
Colapsa el dataset de 3 clases (`alta_calidad`, `calidad_promedio`,
`defectuoso`) a un problema binario (`sano` vs `defectuoso`).

Justificacion: tras dos rondas de entrenamiento (solo datos reales, y datos
reales + pseudo-etiquetas validadas por acuerdo entre ResNet18 y HOG-SVM),
`calidad_promedio` se mantuvo en 0% de recall en ambos casos y con ambos
modelos. La frontera `alta_calidad` vs `calidad_promedio` no es separable
con las features/datos disponibles; en cambio, `defectuoso` se detecta de
forma consistente y casi perfecta. Fusionar `alta_calidad` + `calidad_promedio`
en `sano` produce un problema binario que SI esta respaldado por evidencia
real, en vez de forzar una distincion de 3 clases que el dato no sostiene.

No se modifica `data/hdfs_manifest.json` (que documenta el linaje real de 3
clases); el colapso a binario es una transformacion de entrenamiento, no de
los datos de origen.

Uso:
    python scripts/collapse_to_binary.py
"""
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

PROCESADOS_3CLASES = RAIZ / "dataset_mangos" / "processed"
PROCESADOS_BINARIO = RAIZ / "dataset_mangos" / "processed_binary"

MAPA_A_BINARIO = {
    "alta_calidad": "sano",
    "calidad_promedio": "sano",
    "defectuoso": "defectuoso",
}


def main():
    if PROCESADOS_BINARIO.exists():
        shutil.rmtree(PROCESADOS_BINARIO)

    contador = {"train": {"sano": 0, "defectuoso": 0}, "test": {"sano": 0, "defectuoso": 0}}

    for particion in ("train", "test"):
        for clase_origen, clase_binaria in MAPA_A_BINARIO.items():
            carpeta_origen = PROCESADOS_3CLASES / particion / clase_origen
            if not carpeta_origen.exists():
                continue
            carpeta_destino = PROCESADOS_BINARIO / particion / clase_binaria
            carpeta_destino.mkdir(parents=True, exist_ok=True)

            for ruta in carpeta_origen.glob("*"):
                shutil.copyfile(ruta, carpeta_destino / ruta.name)
                contador[particion][clase_binaria] += 1

    for particion, clases in contador.items():
        print(f"{particion}: {clases}")
    print(f"\nDataset binario materializado en: {PROCESADOS_BINARIO}")


if __name__ == "__main__":
    main()
