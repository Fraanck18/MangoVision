"""
Integra las pseudo-etiquetas de FLIR (aceptadas por acuerdo entre ResNet18 y
HOG-SVM, ver scripts/cross_validate_pseudo_labels.py) al pool de
entrenamiento.

Decision metodologica: las pseudo-etiquetas SOLO se agregan a `train`, nunca
a `test`. El test set debe seguir compuesto exclusivamente por etiquetas
reales verificadas por humanos (Roboflow), para que las metricas reportadas
sigan siendo honestas y no se midan contra datos que el propio sistema se
auto-etiqueto.

Uso:
    python scripts/merge_pseudo_labels.py
"""
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
PSEUDO_ETIQUETAS = RAIZ / "dataset_mangos" / "flir_pseudo_labels.json"
MANIFIESTO = RAIZ / "data" / "hdfs_manifest.json"


def main():
    with open(PSEUDO_ETIQUETAS) as f:
        pseudo_registros = json.load(f)
    print(f"Pseudo-etiquetas a integrar (solo a train): {len(pseudo_registros)}")

    with open(MANIFIESTO) as f:
        manifiesto = json.load(f)
    siguiente_id = len(manifiesto["registros"])

    for registro in pseudo_registros:
        clase = registro["clase_predicha"]
        ruta_origen = RAIZ / registro["ruta_origen"]

        carpeta_destino = PROCESADOS / "train" / clase
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        nombre_destino = f"flir_pseudo_{ruta_origen.name}"
        ruta_destino = carpeta_destino / nombre_destino
        shutil.copyfile(ruta_origen, ruta_destino)

        manifiesto["registros"].append({
            "id_muestra": f"mango_{siguiente_id:05d}",
            "foto_base_id": f"flir_{ruta_origen.stem}",
            "fuente_original": "flir_one_pseudo_etiquetado",
            "uri_hdfs": f"hdfs://nodo-maestro:9000/data-lake/mangos/raw/train/{clase}/{nombre_destino}",
            "ruta_local": str(ruta_destino.relative_to(RAIZ)).replace("\\", "/"),
            "etiqueta": CLASS_NAMES.index(clase),
            "particion": "train",
            "pseudo_etiquetado": True,
            "metodo_etiquetado": registro["metodo"],
        })
        siguiente_id += 1

    manifiesto["total_imagenes"] = len(manifiesto["registros"])
    manifiesto["descripcion"] += (
        " Incluye ademas pseudo-etiquetas de FLIR One aceptadas por acuerdo "
        "entre ResNet18 y HOG-SVM (ver pseudo_etiquetado=true en cada registro), "
        "agregadas unicamente a la particion train."
    )

    with open(MANIFIESTO, "w") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    print(f"Manifiesto actualizado: {manifiesto['total_imagenes']} imagenes totales")
    print(f"Imagenes pseudo-etiquetadas copiadas a: {PROCESADOS / 'train'}")


if __name__ == "__main__":
    main()
