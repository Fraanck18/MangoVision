"""
Materializacion OFFLINE de la augmentation fisica (#1) sobre el dato real.

Por que offline y no online: la augmentation fisica (decode Ironbow -> perturbar
ITR -> re-encode) es relativamente cara y, aplicada por época en el DataLoader,
frenaria el entrenamiento. Ademas, materializarla UNA vez la hace reproducible y
—alineado con la arquitectura del proyecto— empaquetable en HDFS/Parquet para el
pipeline batch (evitando el small-file problem). Este script la genera a disco.

Salida: `dataset_mangos/physics_augment/train/<clase>/`, que `train_head_fast.py`
suma automaticamente al TRAIN (esta en DIRS_AUMENTO). NUNCA a val/test.

[ANTI-FUGA / F4] Cada imagen aumentada CONSERVA en su nombre el token de grupo del
mango de origen (Mango_10, v3i_Mango_2, flir_pseudo_xxx). Asi el split agrupado
Train/Val de `train_head_fast.py` puede EXCLUIR del train las variantes aumentadas
de un mango que cayo en validacion -> no hay fuga de identidad de fruto.

Uso:
    python scripts/materializar_augmentation.py
    MANGO_PHYS_POR_IMG=3 python scripts/materializar_augmentation.py
"""
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.augmentation.termica_fisica import AugmentacionTermicaFisica

PROCESADOS = RAIZ / "dataset_mangos" / "processed" / "train"
SALIDA = RAIZ / "dataset_mangos" / "physics_augment" / "train"

# Variantes fisicas por imagen real.
POR_IMAGEN = int(os.getenv("MANGO_PHYS_POR_IMG", "3"))
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

# Mismo patron de grupo que train_head_fast.py / honest_grouped_cv.py.
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def _stem_con_grupo(nombre):
    """
    Devuelve un fragmento del nombre original que CONTIENE el token de grupo, para
    que PATRON_GRUPO lo siga detectando en el archivo aumentado (anti-fuga).
    """
    m = PATRON_GRUPO.search(nombre)
    return m.group(1) if m else Path(nombre).stem[:40]


def main():
    if not PROCESADOS.exists():
        print(f"No existe {PROCESADOS}. Prepara el dataset real primero "
              f"(scripts/prepare_real_dataset.py).")
        return

    if SALIDA.parent.exists():
        shutil.rmtree(SALIDA.parent)

    rng_seed = RANDOM_SEED
    total = 0
    for clase in CLASS_NAMES:
        carpeta_in = PROCESADOS / clase
        if not carpeta_in.exists():
            continue
        carpeta_out = SALIDA / clase
        carpeta_out.mkdir(parents=True, exist_ok=True)

        imagenes = [p for p in carpeta_in.iterdir() if p.suffix.lower() in EXTENSIONES]
        for ruta in imagenes:
            grupo = _stem_con_grupo(ruta.name)
            img = Image.open(ruta).convert("RGB")
            for k in range(POR_IMAGEN):
                # Semilla determinista por (imagen, variante) -> reproducible.
                aug = AugmentacionTermicaFisica(p=1.0, seed=rng_seed)
                rng_seed += 1
                salida_img = aug(img)
                # El nombre conserva el token de grupo -> anti-fuga en el split.
                salida_img.save(carpeta_out / f"phys_{grupo}_{ruta.stem[:24]}_a{k}.jpg", quality=92)
                total += 1
        print(f"  {clase}: {len(imagenes)} reales -> {len(imagenes) * POR_IMAGEN} fisicas")

    print(f"\nMaterializadas {total} imagenes de augmentation fisica en: {SALIDA}")
    print("Se suman a TRAIN automaticamente (DIRS_AUMENTO en train_head_fast.py).")


if __name__ == "__main__":
    main()
