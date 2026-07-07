"""
Entrenamiento RAPIDO en CPU (feature-extractor + cabeza).

[Optimizacion CPU] El fine-tuning de la CNN completa es lento en CPU porque cada
epoca hace backprop por millones de parametros. Aqui el backbone va TOTALMENTE
congelado, asi que sus features son deterministas: se PRECOMPUTAN una sola vez
(un forward por imagen) y luego se entrena UNICAMENTE la cabeza (un MLP pequeno)
sobre esos vectores cacheados. Resultado: horas -> segundos, sin GPU.

Se conserva la disciplina honesta del plan:
  * [F4] Split Train/Val/Test independiente; el modelo se elige por VALIDATION y
    el Test real se evalua UNA sola vez.
  * [F7] Backbone ligero (ResNet18 por defecto, MANGO_BACKBONE).
El artefacto guardado (best_model.pth) es el state_dict completo del modelo
(backbone congelado + cabeza entrenada), 100% compatible con src/api/app.py.

Uso:
    python scripts/train_head_fast.py
"""
import copy
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.models.resnet50_custom import (
    ResNet50MangoQC,
    obtener_transformaciones_validacion,
    cargar_backbone_ssl_si_existe,
)
from src.augmentation.regularizadores import mixup_lote, perdida_mixup
from src.models.evaluator import evaluar_convergencia_real
from src.pipeline.historial_entrenamiento import agregar_a_historial, escribir_estado

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
PESOS_SALIDA = RAIZ / "dataset_mangos" / "best_model.pth"

# Directorios de DATA AUGMENTATION que se suman SOLO al train (nunca val/test):
# imagenes GAN condicionales, el pool sintetico fisico y la augmentation fisica
# MATERIALIZADA offline (scripts/materializar_augmentation.py). Rellenan los
# campos flojos (frio/promedio/caliente) y balancean las clases del dataset escaso.
DIRS_AUMENTO = [
    RAIZ / "dataset_mangos" / "gan_augment" / "train",
    RAIZ / "dataset_mangos" / "synthetic_pretrain" / "train",
    RAIZ / "dataset_mangos" / "physics_augment" / "train",
]

# [#4] MixUp en el espacio de FEATURES (backbone congelado -> mezclamos vectores).
MIXUP_ALPHA = float(os.getenv("MANGO_MIXUP_ALPHA", "0.2"))

VAL_FRACCION = 0.2
EPOCAS_CABEZA = 80
TAMANO_LOTE = 64
LR_CABEZA = 1e-3

PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def _grupo_de(ruta):
    match = PATRON_GRUPO.search(Path(ruta).name)
    return match.group(1) if match else Path(ruta).name


def separar_train_val(dataset):
    """[F4] Split Train/Val agrupado por mango fisico (fallback aleatorio)."""
    grupos = [_grupo_de(r) for r, _ in dataset.samples]
    indices = list(range(len(dataset)))
    if len(set(grupos)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACCION, random_state=RANDOM_SEED)
        idx_train, idx_val = next(gss.split(indices, groups=grupos))
        if len(idx_val) and len(idx_train):
            return list(idx_train), list(idx_val)
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    perm = torch.randperm(len(indices), generator=gen).tolist()
    corte = max(1, int(len(perm) * VAL_FRACCION))
    return perm[corte:], perm[:corte]


@torch.no_grad()
def precomputar_features(modelo, dataset, indices, device):
    """Ejecuta el backbone congelado UNA vez sobre los indices dados -> (features, labels)."""
    modelo.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=TAMANO_LOTE, shuffle=False)
    feats, labels = [], []
    for imagenes, etiquetas in loader:
        feats.append(modelo.backbone(imagenes.to(device)).cpu())
        labels.append(etiquetas)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def features_de_aumento(modelo, tf, device, grupos_val=frozenset()):
    """
    Carga las carpetas de DATA AUGMENTATION (GAN + sintetico fisico + fisica
    materializada) y devuelve sus features/labels para SUMARLAS al train de la
    cabeza. Verifica el orden de clases. Devuelve (None, None) si no hay.

    [ANTI-FUGA / F4] Se EXCLUYE toda imagen aumentada cuyo token de grupo pertenezca
    a `grupos_val`: si un mango cayo en validacion, sus variantes fisicas derivadas
    (que conservan el token de grupo en el nombre) no pueden entrar al train. Las
    imagenes GAN/sinteticas no matchean ningun grupo real, asi que nunca se excluyen.
    """
    feats, labels = [], []
    for carpeta in DIRS_AUMENTO:
        if not carpeta.exists():
            continue
        ds = ImageFolder(carpeta, transform=tf)
        if ds.classes != CLASS_NAMES:
            print(f"  [aviso] {carpeta.name}: orden de clases {ds.classes} != {CLASS_NAMES}, se omite.")
            continue
        indices = [i for i, (ruta, _) in enumerate(ds.samples)
                   if _grupo_de(ruta) not in grupos_val]
        excluidas = len(ds) - len(indices)
        if not indices:
            print(f"  [aviso] {carpeta.parent.name}: todas sus imgs son de grupos en val, se omite.")
            continue
        f, y = precomputar_features(modelo, ds, indices, device)
        feats.append(f); labels.append(y)
        nota = f" ({excluidas} excluidas por estar en val)" if excluidas else ""
        print(f"  + augmentation '{carpeta.parent.name}': {len(indices)} imgs {dict(Counter(y.tolist()))}{nota}")
    if not feats:
        return None, None
    return torch.cat(feats), torch.cat(labels)


def pesos_clase(labels, device):
    conteo = Counter(labels.tolist())
    total = len(labels)
    k = len(CLASS_NAMES)
    return torch.tensor([total / (k * conteo.get(i, 1)) for i in range(k)],
                        dtype=torch.float, device=device)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).float().mean().item())


def generar_reporte(y_true, y_pred):
    # El gate relativo (evaluator.py) ya no lanza: imprime veredicto + devuelve el
    # DataFrame. El Test real es 100% humano -> metrica del gate ya es "solo real".
    df = evaluar_convergencia_real(y_true, y_pred)
    print(df)
    return df.to_dict()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    inicio = time.time()

    # Transforms de validacion (sin augmentation) -> features deterministas cacheables.
    tf = obtener_transformaciones_validacion()
    ds_train_full = ImageFolder(PROCESADOS / "train", transform=tf)
    ds_test = ImageFolder(PROCESADOS / "test", transform=tf)
    assert ds_train_full.classes == CLASS_NAMES, f"{ds_train_full.classes} != {CLASS_NAMES}"

    idx_train, idx_val = separar_train_val(ds_train_full)
    print(f"Backbone congelado: {ResNet50MangoQC().arquitectura} | "
          f"train={len(idx_train)} val={len(idx_val)} test={len(ds_test)}")

    # Backbone TOTALMENTE congelado (capas_descongeladas=0) -> solo la cabeza entrena.
    modelo = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    # [#2 SSL] Features de un backbone auto-supervisado (stream no etiquetado) en vez
    # de ImageNet puro, si hay checkpoint (MANGO_SSL_BACKBONE / ssl_backbone.pth).
    cargar_backbone_ssl_si_existe(modelo)

    print("Precomputando features (un forward por imagen, una sola vez)...")
    f_train, y_train = precomputar_features(modelo, ds_train_full, idx_train, device)
    f_val, y_val = precomputar_features(modelo, ds_train_full, idx_val, device)
    f_test, y_test = precomputar_features(modelo, ds_test, list(range(len(ds_test))), device)

    # DATA AUGMENTATION: se suma SOLO al train (val/test intactos, reales).
    # Grupos que cayeron en VAL -> se excluyen sus variantes aumentadas (anti-fuga).
    grupos_val = {_grupo_de(ds_train_full.samples[i][0]) for i in idx_val}
    n_real = len(y_train)
    f_aug, y_aug = features_de_aumento(modelo, tf, device, grupos_val=grupos_val)
    if f_aug is not None:
        f_train = torch.cat([f_train, f_aug])
        y_train = torch.cat([y_train, y_aug])
        print(f"Train aumentado: {n_real} reales + {len(y_aug)} augment = {len(y_train)} "
              f"| balance {dict(Counter(y_train.tolist()))}")
    else:
        print("Sin carpetas de augmentation (ejecuta train_gan_augment.py / generate_synthetic_thermal.py).")
    print(f"Features listas en {time.time()-inicio:.1f}s (dim={f_train.shape[1]}). Entrenando cabeza...")

    # Entrenamiento de la cabeza sobre features cacheadas (rapidisimo en CPU).
    sampler = WeightedRandomSampler(
        [1.0 / Counter(y_train.tolist())[t.item()] for t in y_train],
        num_samples=len(y_train), replacement=True,
    )
    loader = DataLoader(TensorDataset(f_train, y_train), batch_size=TAMANO_LOTE, sampler=sampler)
    optimizador = torch.optim.Adam(modelo.head.parameters(), lr=LR_CABEZA)
    criterio = torch.nn.CrossEntropyLoss(weight=pesos_clase(y_train, device))

    mejor_val = -1.0
    mejor_head = copy.deepcopy(modelo.head.state_dict())
    historial = []
    for epoca in range(1, EPOCAS_CABEZA + 1):
        modelo.head.train()
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            optimizador.zero_grad()
            if MIXUP_ALPHA > 0:
                # MixUp en el espacio de features (equivalente lineal, backbone congelado).
                fb, y_a, y_b, lam = mixup_lote(fb, yb, alpha=MIXUP_ALPHA)
                perdida = perdida_mixup(criterio, modelo.head(fb), y_a, y_b, lam)
            else:
                perdida = criterio(modelo.head(fb), yb)
            perdida.backward()
            optimizador.step()

        modelo.head.eval()
        with torch.no_grad():
            val_pred = torch.argmax(modelo.head(f_val.to(device)), dim=1).cpu()
        val_acc = accuracy(y_val, val_pred)
        historial.append({"epoca": epoca, "val_acc": round(val_acc, 4)})
        if val_acc > mejor_val:                       # [F4] checkpoint por VAL
            mejor_val = val_acc
            mejor_head = copy.deepcopy(modelo.head.state_dict())
        if epoca % 10 == 0 or epoca == 1:
            print(f"  epoca {epoca}/{EPOCAS_CABEZA} - val_acc: {val_acc:.4f}")

    modelo.head.load_state_dict(mejor_head)
    print(f"\nMejor val_acc: {mejor_val:.4f}")

    # ===== EVALUACION UNICA FINAL SOBRE EL TEST REAL BLOQUEADO (F4) =====
    print("\n=== Evaluacion UNICA final sobre el Test REAL ===")
    modelo.head.eval()
    with torch.no_grad():
        p_test = torch.argmax(modelo.head(f_test.to(device)), dim=1).cpu()
    reporte = generar_reporte(y_test, p_test)
    acc_test = accuracy(y_test, p_test)
    print(f"\nLinea Base Real Independiente (features {modelo.arquitectura} + cabeza) -> "
          f"accuracy_test: {acc_test:.4f}")

    torch.save(modelo.state_dict(), PESOS_SALIDA)
    print(f"Pesos guardados en: {PESOS_SALIDA}")

    resumen = {
        "estado": "completado",
        "metodo": "feature-extractor CPU-fast (backbone congelado + cabeza)",
        "backbone": modelo.arquitectura,
        "val_acc": round(mejor_val, 4),
        "accuracy_test": round(acc_test, 4),
        "reporte_test": reporte,
        "historial": historial,
        "epocas_totales": EPOCAS_CABEZA,
        "epoca_actual": EPOCAS_CABEZA,
        "seleccion": "por VALIDATION (test evaluado una sola vez)",
        "inicio": inicio,
        "fin": time.time(),
        "pesos": str(PESOS_SALIDA),
    }
    escribir_estado(resumen)
    agregar_a_historial(resumen)
    print(f"\nListo en {time.time()-inicio:.1f}s total (CPU).")


if __name__ == "__main__":
    main()
