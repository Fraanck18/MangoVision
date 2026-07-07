"""
Entrenamiento + evaluacion sobre una POBLACION TERMICAMENTE DIVERSA generada por
data augmentation de temperatura.

Motivacion (del usuario): el dataset no son "solo 11 mangos". Cada imagen es una
matriz termica que puede llevarse a OTRAS temperaturas; esas variantes son muestras
termicas NUEVAS y validas porque el proyecto clasifica por REGIMEN TERMICO. Este
script:

  1. Genera, de cada imagen real, K variantes a distintas temperaturas
     (`variantes_temperatura`), REETIQUETADAS por la regla fisica determinista
     (no por el modelo -> sin circularidad).
  2. Entrena la cabeza (backbone congelado) sobre  real_train + variantes(train).
  3. Evalua en DOS conjuntos, sin fuga de identidad de mango (train/test se separan
     por mango fisico):
       (a) REAL test  -> etiquetas humanas: mide clasificacion de mangos reales.
       (b) TEMP test  -> variantes de temperatura de los mangos de test, etiquetadas
           por fisica: mide clasificacion del REGIMEN TERMICO ("evaluar entre ellas").

Reporta accuracy + recall por clase en ambos, con honestidad sobre que mide cada uno.

Uso:
    python scripts/entrenar_temperatura.py --k 6 --epocas 60
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.models.resnet50_custom import ResNet50MangoQC, obtener_transformaciones_validacion
from src.augmentation.temperatura import variantes_temperatura

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def _grupo(nombre):
    m = PATRON_GRUPO.search(nombre)
    return m.group(1) if m else Path(nombre).stem


def _recolectar_reales():
    items = []
    for particion in ("train", "test"):
        for c, clase in enumerate(CLASS_NAMES):
            carpeta = PROCESADOS / particion / clase
            if carpeta.exists():
                for r in carpeta.glob("*"):
                    if r.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        items.append((r, c, _grupo(r.name)))
    return items


@torch.no_grad()
def _construir_features(items, k, tf, backbone, device, rng):
    """
    Para cada imagen real produce: su feature (prov='real', etiqueta humana) + K
    variantes de temperatura (prov='temp', etiqueta fisica). Devuelve tensores
    F, Y, y arrays grupo, prov.
    """
    feats, ys, grupos, prov = [], [], [], []
    lote_imgs, lote_meta = [], []

    def _flush():
        if not lote_imgs:
            return
        x = torch.stack(lote_imgs).to(device)
        f = backbone(x).cpu()
        for i, (yy, gg, pp) in enumerate(lote_meta):
            feats.append(f[i]); ys.append(yy); grupos.append(gg); prov.append(pp)
        lote_imgs.clear(); lote_meta.clear()

    for ruta, clase_humana, g in items:
        img = Image.open(ruta).convert("RGB")
        lote_imgs.append(tf(img)); lote_meta.append((clase_humana, g, "real"))
        for var_img, etiqueta in variantes_temperatura(img, rng, n=k):
            lote_imgs.append(tf(var_img)); lote_meta.append((CLASS_NAMES.index(etiqueta), g, "temp"))
        if len(lote_imgs) >= 128:
            _flush()
    _flush()
    return torch.stack(feats), torch.tensor(ys), np.array(grupos), np.array(prov)


def _entrenar_cabeza(F_tr, y_tr, epocas, device, seed=0):
    torch.manual_seed(seed)
    modelo = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    conteo = Counter(y_tr.tolist())
    pesos = torch.tensor([len(y_tr) / (len(CLASS_NAMES) * conteo.get(i, 1)) for i in range(len(CLASS_NAMES))],
                         dtype=torch.float, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=pesos)
    opt = torch.optim.Adam(modelo.head.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(F_tr, y_tr), batch_size=64, shuffle=True)
    modelo.head.train()
    for _ in range(epocas):
        for fb, yb in loader:
            opt.zero_grad()
            crit(modelo.head(fb.to(device)), yb.to(device)).backward()
            opt.step()
    modelo.head.eval()
    return modelo


@torch.no_grad()
def _evaluar(modelo, F, y, device):
    if len(y) == 0:
        return None, None
    pred = torch.argmax(modelo.head(F.to(device)), dim=1).cpu().numpy()
    yt = y.numpy()
    acc = float((pred == yt).mean())
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for t, p in zip(yt, pred):
        conf[t, p] += 1
    recall = np.array([conf[c, c] / max(1, conf[c].sum()) for c in range(len(CLASS_NAMES))])
    return acc, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="variantes de temperatura por imagen real")
    ap.add_argument("--epocas", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    tf = obtener_transformaciones_validacion()

    items = _recolectar_reales()
    grupos_unicos = sorted({g for _, _, g in items})
    print(f"Imagenes reales: {len(items)} | mangos fisicos: {len(grupos_unicos)}")

    # Split por MANGO FISICO (sin fuga): train/test de mangos.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    idx = np.arange(len(items))
    g_all = np.array([g for _, _, g in items])
    itr_train, itr_test = next(gss.split(idx, groups=g_all))
    grupos_test = set(g_all[itr_test])

    print(f"Generando features: real + {args.k} variantes de temperatura por imagen "
          f"(backbone congelado)...")
    backbone = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    backbone.eval()
    F, Y, G, P = _construir_features(items, args.k, tf, backbone.backbone, device, rng)

    es_test_mango = np.isin(G, list(grupos_test))
    # Conjuntos:
    m_train = ~es_test_mango                      # todo (real+temp) de mangos de train
    m_eval_real = es_test_mango & (P == "real")   # (a) real test, etiqueta humana
    m_eval_temp = es_test_mango & (P == "temp")   # (b) temp test, etiqueta fisica

    print(f"Train: {m_train.sum()} (real+temp de mangos train) | "
          f"balance clases {dict(Counter(Y[m_train].tolist()))}")
    print(f"Eval REAL (humano): {m_eval_real.sum()} | Eval TEMP (fisico): {m_eval_temp.sum()}\n")

    modelo = _entrenar_cabeza(F[m_train], Y[m_train], args.epocas, device)

    acc_r, rec_r = _evaluar(modelo, F[m_eval_real], Y[m_eval_real], device)
    acc_t, rec_t = _evaluar(modelo, F[m_eval_temp], Y[m_eval_temp], device)

    def _fmt(rec):
        return "[" + ", ".join(f"{r:.2f}" for r in rec) + "]"

    print("=== RESULTADOS (mangos de test nunca vistos en train) ===")
    print(f"(a) REAL test  (etiqueta humana, clasif. de mango real):   acc={acc_r*100:5.1f}%  recall {_fmt(rec_r)}")
    print(f"(b) TEMP test  (etiqueta fisica, clasif. de REGIMEN):      acc={acc_t*100:5.1f}%  recall {_fmt(rec_t)}")
    print("\nrecall por clase: [alta_calidad, calidad_promedio, defectuoso]")
    print("\nLectura honesta:")
    print("  (b) mide si el modelo, entrenado con diversidad de temperatura, reconoce")
    print("      el REGIMEN TERMICO de variantes no vistas -> es la tarea que la")
    print("      augmentation de temperatura optimiza directamente.")
    print("  (a) mide clasificacion de mango real con etiqueta humana; puede diferir")
    print("      de (b) si la etiqueta humana de madurez no coincide con el regimen")
    print("      termico instantaneo (honestidad: son dos definiciones de clase).")


if __name__ == "__main__":
    main()
