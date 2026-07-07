"""
Evaluacion HONESTA (LOGO por mango fisico) del CLASIFICADOR CNN, para MEDIR el
impacto real de la data augmentation en la generalizacion a mangos NUEVOS.

A diferencia de honest_grouped_cv.py (que evalua el baseline PEDTE+LogReg binario),
aqui se evalua el modelo de produccion: backbone congelado + cabeza. Como el
backbone esta congelado, sus features son DETERMINISTAS y se precomputan UNA sola
vez; luego cada fold LOGO solo reentrena la cabeza (segundos) -> se pueden comparar
varias configuraciones de augmentation de forma barata y justa.

Disciplina anti-fuga (identica al resto del proyecto):
  * LOGO sobre el mango fisico: en cada fold se deja fuera UN mango humano y se
    evalua SOLO sobre sus imagenes reales.
  * Las variantes aumentadas del mango held-out (que comparten su token de grupo)
    quedan FUERA de su train. Las muestras sinteticas/GAN (grupo propio) solo
    expanden el train, nunca se evaluan.
  * La metrica se reporta SOLO sobre imagenes humanas reales.

Configuraciones comparadas (backbone x expansion), cada una con su propio
precomputo de features:
  A. baseline        : backbone ImageNet, train = solo real humano.
  B. +augmentation   : backbone ImageNet, train = real + fisica + sintetico (+GAN).
  C. +SSL            : backbone auto-supervisado (SimSiam), train = real + expansion.

Uso:
    python scripts/honest_cv_cnn.py
    python scripts/honest_cv_cnn.py --epocas 60
"""
import argparse
import copy
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.models.resnet50_custom import ResNet50MangoQC, obtener_transformaciones_validacion, cargar_backbone_ssl_si_existe
from scripts.honest_grouped_cv import procedencia_de

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
SSL_CKPT = RAIZ / "dataset_mangos" / "ssl_backbone.pth"
DIRS_EXPANSION = [
    RAIZ / "dataset_mangos" / "physics_augment" / "train",
    RAIZ / "dataset_mangos" / "synthetic_pretrain" / "train",
    RAIZ / "dataset_mangos" / "gan_augment" / "train",
]
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def _grupo(nombre):
    m = PATRON_GRUPO.search(nombre)
    return m.group(1) if m else Path(nombre).stem


def _recolectar_rutas():
    """Reune (ruta, clase_idx, grupo, procedencia) del dato real (train+test) + expansion."""
    items = []
    for particion in ("train", "test"):
        for c, clase in enumerate(CLASS_NAMES):
            carpeta = PROCESADOS / particion / clase
            if carpeta.exists():
                for r in carpeta.glob("*"):
                    if r.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        items.append((r, c, _grupo(r.name), "humana"))
    for d in DIRS_EXPANSION:
        for c, clase in enumerate(CLASS_NAMES):
            carpeta = d / clase
            if carpeta.exists():
                for r in carpeta.glob("*"):
                    if r.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        items.append((r, c, _grupo(r.name), procedencia_de(r.name)))
    return items


@torch.no_grad()
def _precomputar_features(items, usar_ssl):
    """Backbone congelado (ImageNet o SSL) -> features deterministas de TODAS las imagenes."""
    device = torch.device("cpu")
    modelo = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    if usar_ssl:
        if not cargar_backbone_ssl_si_existe(modelo, ruta=str(SSL_CKPT), verbose=False):
            raise FileNotFoundError(f"No hay checkpoint SSL en {SSL_CKPT} (corre pretrain_self_supervised.py).")
    modelo.eval()

    tf = obtener_transformaciones_validacion()
    from PIL import Image

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i):
            r = items[i][0]
            return tf(Image.open(r).convert("RGB")), i

    feats = torch.zeros(len(items), modelo.head[0].in_features)
    for xb, idx in DataLoader(_DS(), batch_size=64):
        feats[idx] = modelo.backbone(xb).cpu()
    y = torch.tensor([it[1] for it in items])
    grupos = np.array([it[2] for it in items])
    prov = np.array([it[3] for it in items])
    return feats, y, grupos, prov


def _entrenar_cabeza(F_tr, y_tr, epocas, mixup_alpha, device, seed=0):
    """Entrena una cabeza nueva sobre features cacheadas, con class-weight (y MixUp opcional)."""
    from src.augmentation.regularizadores import mixup_lote, perdida_mixup

    # Semilla por (fold, seed) -> reproducible; permite promediar sobre seeds para
    # domar la varianza del head-training en un LOGO de solo 11 grupos.
    torch.manual_seed(seed)
    modelo = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    conteo = Counter(y_tr.tolist())
    pesos = torch.tensor([len(y_tr) / (len(CLASS_NAMES) * conteo.get(i, 1)) for i in range(len(CLASS_NAMES))],
                         dtype=torch.float, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=pesos)
    opt = torch.optim.Adam(modelo.head.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(F_tr, y_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    modelo.head.train()
    for _ in range(epocas):
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            opt.zero_grad()
            if mixup_alpha > 0:
                fb, ya, yb2, lam = mixup_lote(fb, yb, alpha=mixup_alpha)
                perdida = perdida_mixup(crit, modelo.head(fb), ya, yb2, lam)
            else:
                perdida = crit(modelo.head(fb), yb)
            perdida.backward()
            opt.step()
    return modelo


def logo_cnn(feats, y, grupos, prov, incluir_prov, epocas=50, mixup_alpha=0.0, seeds=(0, 1, 2)):
    """
    LOGO restringido sobre features cacheadas, PROMEDIADO sobre varias seeds de
    head-training para domar la varianza (11 grupos son pocos). `incluir_prov` =
    procedencias admitidas en TRAIN (siempre incluye 'humana'). Devuelve
    (acc_media, acc_std_entre_mangos, recall_por_clase, n_folds).
    """
    device = torch.device("cpu")
    idx = np.arange(len(y))
    grupos_humanos = sorted(set(grupos[prov == "humana"]))
    admitidas = set(incluir_prov) | {"humana"}

    acc_por_seed = []
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)  # confusion agregada (todas las seeds)
    acc_por_mango = defaultdict(list)
    for seed in seeds:
        accs = []
        for g in grupos_humanos:
            test = idx[(grupos == g) & (prov == "humana")]
            train = idx[(grupos != g) & np.isin(prov, list(admitidas))]
            if not len(test) or not len(train):
                continue
            modelo = _entrenar_cabeza(feats[train], y[train], epocas, mixup_alpha, device, seed=seed)
            modelo.head.eval()
            with torch.no_grad():
                pred = torch.argmax(modelo.head(feats[test].to(device)), dim=1).cpu().numpy()
            yt = y[test].numpy()
            a = float((pred == yt).mean())
            accs.append(a); acc_por_mango[g].append(a)
            for t, p in zip(yt, pred):
                conf[t, p] += 1
        acc_por_seed.append(float(np.mean(accs)))

    # Media = promedio sobre seeds; std = dispersion ENTRE MANGOS (media por mango).
    media_por_mango = np.array([np.mean(v) for v in acc_por_mango.values()])
    recall = np.array([conf[c, c] / max(1, conf[c].sum()) for c in range(len(CLASS_NAMES))])
    return float(np.mean(acc_por_seed)), float(media_por_mango.std()), recall, len(media_por_mango)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--mixup", type=float, default=0.2)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    items = _recolectar_rutas()
    n_hum = sum(1 for it in items if it[3] == "humana")
    print(f"Imagenes: {len(items)} (humanas: {n_hum}, expansion: {len(items) - n_hum})")
    print(f"Procedencias: {dict(Counter(it[3] for it in items))}\n")

    configs = [
        ("A. baseline (solo real)",                False, set(),                 0.0),
        ("+fisica (K=3)",                          False, {"phys_aug"},          0.0),
        ("+sintetico",                             False, {"synthetic_physics"}, 0.0),
        ("+sintetico+MixUp",                       False, {"synthetic_physics"}, args.mixup),
    ]
    if SSL_CKPT.exists():
        configs.append(
            ("+SSL (SimSiam) + todo", True, {"phys_aug", "synthetic_physics", "gan"}, args.mixup))
    else:
        print(f"(sin checkpoint SSL en {SSL_CKPT}: se omite la config SSL)\n")

    # Precompute de features por backbone (ImageNet una vez; SSL una vez si aplica).
    cache = {}
    print("Precomputando features (backbone congelado, una vez por backbone)...")
    cache[False] = _precomputar_features(items, usar_ssl=False)
    if any(c[1] for c in configs):
        cache[True] = _precomputar_features(items, usar_ssl=True)

    print(f"\n{'Config':40s} {'acc LOGO':>14s}  recall por clase [alta, prom, defect]")
    print("-" * 92)
    base_acc = None
    for nombre, usar_ssl, incluir, mixup in configs:
        feats, y, grupos, prov = cache[usar_ssl]
        acc, std, recall, nf = logo_cnn(feats, y, grupos, prov, incluir, args.epocas, mixup)
        if base_acc is None:
            base_acc = acc
        delta = acc - base_acc
        rec = "[" + ", ".join(f"{r:.2f}" for r in recall) + "]"
        print(f"{nombre:40s} {acc*100:5.1f}% ±{std*100:4.1f}  Δ{delta*100:+5.1f}  {rec}")

    print(f"\n({nf} folds LOGO = mangos humanos; metrica solo sobre imagenes reales)")
    print("recall clases: [alta_calidad, calidad_promedio, defectuoso]  <- vigilar calidad_promedio")


if __name__ == "__main__":
    main()
