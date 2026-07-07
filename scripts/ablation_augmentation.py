"""
[#1-A] Ablation HONESTA del curriculo de augmentation.

Principio (del plan): cada tecnica se acepta SOLO si mueve la metrica sobre dato
REAL no visto, no la de train. El 0.72 no es un destino fijo; es lo que da un
dataset diminuto sin explotar el toolkit. Este script entrena la cabeza rapida
(backbone congelado, CPU) bajo distintas configuraciones de augmentation y compara
su accuracy sobre el MISMO split de validacion agrupado por mango fisico (sin fuga).

No sustituye a `honest_grouped_cv.py` (LOGO PEDTE, la metrica publicable): es una
sonda rapida para DECIDIR que tecnicas conservar antes de pagar el LOGO completo.
La regla se mantiene: se compara siempre contra dato real agrupado, jamas contra
train.

Configuraciones (env que consume resnet50_custom.obtener_transformaciones_entrenamiento):
  * basico              -> MANGO_AUG_BASICO=1 (flip+rot, referencia)
  * +geometrico/elastica/fisica/erasing -> flags MANGO_AUG_* = 0/1
  * +mixup              -> MANGO_MIXUP_ALPHA
  * +ssl                -> MANGO_SSL_BACKBONE (si hay checkpoint)

Uso:
    python scripts/ablation_augmentation.py
    python scripts/ablation_augmentation.py --epocas 40
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED

PROCESADOS = RAIZ / "dataset_mangos" / "processed"

# Cada config: nombre -> dict de variables de entorno a fijar durante el run.
CONFIGS = {
    "basico (flip+rot)":       {"MANGO_AUG_BASICO": "1", "MANGO_MIXUP_ALPHA": "0"},
    "+geometrico":             {"MANGO_AUG_BASICO": "0", "MANGO_AUG_GEOMETRICO": "1",
                                "MANGO_AUG_ELASTICA": "0", "MANGO_AUG_FISICA": "0",
                                "MANGO_AUG_ERASING": "0", "MANGO_MIXUP_ALPHA": "0"},
    "+elastica":               {"MANGO_AUG_BASICO": "0", "MANGO_AUG_GEOMETRICO": "1",
                                "MANGO_AUG_ELASTICA": "1", "MANGO_AUG_FISICA": "0",
                                "MANGO_AUG_ERASING": "0", "MANGO_MIXUP_ALPHA": "0"},
    "+fisica":                 {"MANGO_AUG_BASICO": "0", "MANGO_AUG_GEOMETRICO": "1",
                                "MANGO_AUG_ELASTICA": "1", "MANGO_AUG_FISICA": "1",
                                "MANGO_AUG_ERASING": "0", "MANGO_MIXUP_ALPHA": "0"},
    "+erasing":                {"MANGO_AUG_BASICO": "0", "MANGO_AUG_GEOMETRICO": "1",
                                "MANGO_AUG_ELASTICA": "1", "MANGO_AUG_FISICA": "1",
                                "MANGO_AUG_ERASING": "1", "MANGO_MIXUP_ALPHA": "0"},
    "+mixup (todo)":           {"MANGO_AUG_BASICO": "0", "MANGO_AUG_GEOMETRICO": "1",
                                "MANGO_AUG_ELASTICA": "1", "MANGO_AUG_FISICA": "1",
                                "MANGO_AUG_ERASING": "1", "MANGO_MIXUP_ALPHA": "0.2"},
}


def _importar_head_fast():
    """Importa el modulo del entrenador rapido para reutilizar su logica de split/features."""
    import importlib
    return importlib.import_module("scripts.train_head_fast")


def evaluar_config(nombre, envs, epocas):
    """
    Aplica las variables de entorno de la config, reimporta el pipeline (para que
    las transformaciones se reconstruyan con los flags) y entrena la cabeza,
    devolviendo la accuracy de VALIDATION agrupada (dato real no visto).
    """
    for k, v in envs.items():
        os.environ[k] = v

    from torchvision.datasets import ImageFolder
    from src.models.resnet50_custom import (
        ResNet50MangoQC, obtener_transformaciones_entrenamiento, obtener_transformaciones_validacion,
        cargar_backbone_ssl_si_existe,
    )
    thf = _importar_head_fast()

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")

    # Split agrupado por mango fisico (mismo criterio anti-fuga que el entrenador).
    ds_val = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_validacion())
    idx_train, idx_val = thf.separar_train_val(ds_val)

    # Train con la augmentation de la config; features NO cacheadas (varian por época).
    ds_train = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_entrenamiento())
    modelo = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    cargar_backbone_ssl_si_existe(modelo, verbose=False)

    loader_train = DataLoader(Subset(ds_train, idx_train), batch_size=32, shuffle=True)
    f_val, y_val = thf.precomputar_features(modelo, ds_val, idx_val, device)

    optim = torch.optim.Adam(modelo.head.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    mejor = 0.0
    for _ in range(epocas):
        modelo.train()
        for imgs, ys in loader_train:
            optim.zero_grad()
            crit(modelo(imgs.to(device)), ys.to(device)).backward()
            optim.step()
        modelo.eval()
        with torch.no_grad():
            pred = torch.argmax(modelo.head(f_val.to(device)), dim=1).cpu()
        acc = float((pred == y_val).float().mean())
        mejor = max(mejor, acc)
    return mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=25)
    args = ap.parse_args()

    print(f"Ablation honesta (val agrupada por mango, {args.epocas} epocas/config)\n")
    resultados = {}
    base = None
    for nombre, envs in CONFIGS.items():
        t0 = time.time()
        acc = evaluar_config(nombre, envs, args.epocas)
        resultados[nombre] = acc
        if base is None:
            base = acc
        delta = acc - base
        print(f"  {nombre:22s} val_acc={acc:.3f}  (Δ vs basico {delta:+.3f})  [{time.time()-t0:.0f}s]")

    print("\nRegla: conservar una tecnica SOLO si su Δ sobre dato real es positivo y "
          "estable. La metrica publicable final es el LOGO de honest_grouped_cv.py.")


if __name__ == "__main__":
    main()
