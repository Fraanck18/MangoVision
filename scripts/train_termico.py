"""
Entrenamiento de PRODUCCION del clasificador de ESTADO TERMICO del fruto.

Objetivo confirmado: el sistema debe leer el REGIMEN TERMICO que ve el sensor
infrarrojo en la faja (calidad_promedio=frio/verde, alta_calidad=medio/maduro,
defectuoso=caliente/sobre-maduro), no la etiqueta humana de madurez. La clase se
define por la firma termica, y la data augmentation de temperatura expande la
diversidad termica del dato real.

Flujo HONESTO (sin fuga de identidad de mango):
  1. Split por MANGO FISICO en Train / Val / Test (grupos disjuntos).
  2. De cada imagen real se generan K variantes a distintas temperaturas
     (`variantes_temperatura`), REETIQUETADAS por la regla fisica determinista.
  3. Se entrena la cabeza (backbone congelado -> best_model.pth 100% compatible con
     la API). El modelo se elige por VALIDATION (regimen termico), el Test se evalua
     UNA sola vez.
  4. Gate honesto (evaluator.evaluar_gate) sobre el Test. Se guarda best_model.pth y
     el resultado en web/data/thermal_cv.json.

Uso:
    python scripts/train_termico.py --k 6 --epocas 80
"""
import argparse
import copy
import json
import re
import sys
import time
from collections import Counter
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
from src.models.evaluator import evaluar_gate
from src.augmentation.temperatura import variantes_temperatura

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
PESOS_SALIDA = RAIZ / "dataset_mangos" / "best_model.pth"
RESULTADO_JSON = RAIZ / "web" / "data" / "thermal_cv.json"
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def _grupo(nombre):
    m = PATRON_GRUPO.search(nombre)
    return m.group(1) if m else Path(nombre).stem


def _recolectar():
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
def _features(items_idx, items, k, incluir_real, tf, backbone, device, rng):
    """
    Features de un subconjunto (por indice) = para cada imagen: opcionalmente su
    feature real (etiqueta humana) + K variantes de temperatura (etiqueta fisica).
    Para Val/Test se usa incluir_real=False -> se evalua el REGIMEN termico puro.
    """
    feats, ys = [], []
    lote, meta = [], []

    def _flush():
        if not lote:
            return
        f = backbone(torch.stack(lote).to(device)).cpu()
        for i, yy in enumerate(meta):
            feats.append(f[i]); ys.append(yy)
        lote.clear(); meta.clear()

    for i in items_idx:
        ruta, clase_humana, _ = items[i]
        img = Image.open(ruta).convert("RGB")
        if incluir_real:
            lote.append(tf(img)); meta.append(clase_humana)
        for var_img, etiqueta in variantes_temperatura(img, rng, n=k):
            lote.append(tf(var_img)); meta.append(CLASS_NAMES.index(etiqueta))
        if len(lote) >= 128:
            _flush()
    _flush()
    return torch.stack(feats), torch.tensor(ys)


def _split_grupos(items):
    """Split por mango: Test 25%, Val 20% de lo restante; grupos disjuntos."""
    g = np.array([it[2] for it in items])
    idx = np.arange(len(items))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    resto, test = next(gss1.split(idx, groups=g))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=RANDOM_SEED)
    tr_rel, val_rel = next(gss2.split(resto, groups=g[resto]))
    return resto[tr_rel], resto[val_rel], test


def _acc_recall(y, p):
    y, p = np.asarray(y), np.asarray(p)
    acc = float((y == p).mean())
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for t, q in zip(y, p):
        conf[t, q] += 1
    recall = [conf[c, c] / max(1, conf[c].sum()) for c in range(len(CLASS_NAMES))]
    return acc, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epocas", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    tf = obtener_transformaciones_validacion()
    t0 = time.time()

    items = _recolectar()
    idx_tr, idx_val, idx_test = _split_grupos(items)
    g = np.array([it[2] for it in items])
    print(f"Mangos -> train:{len(set(g[idx_tr]))} val:{len(set(g[idx_val]))} test:{len(set(g[idx_test]))} "
          f"(grupos disjuntos, sin fuga)")

    backbone = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    backbone.eval()
    print(f"Generando features termicas (K={args.k} variantes/imagen)...")
    F_tr, y_tr = _features(idx_tr, items, args.k, True, tf, backbone.backbone, device, rng)
    F_val, y_val = _features(idx_val, items, args.k, False, tf, backbone.backbone, device, rng)
    F_test, y_test = _features(idx_test, items, args.k, False, tf, backbone.backbone, device, rng)
    print(f"Train:{len(y_tr)} {dict(Counter(y_tr.tolist()))} | Val:{len(y_val)} | Test:{len(y_test)}")

    # Entrena la cabeza; selecciona por VAL (regimen termico).
    modelo = ResNet50MangoQC(congelar_backbone=True, capas_descongeladas=0).to(device)
    pesos = torch.tensor([len(y_tr) / (len(CLASS_NAMES) * Counter(y_tr.tolist()).get(i, 1))
                          for i in range(len(CLASS_NAMES))], dtype=torch.float, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=pesos)
    opt = torch.optim.Adam(modelo.head.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(F_tr, y_tr), batch_size=64, shuffle=True)

    mejor_val, mejor_head = -1.0, copy.deepcopy(modelo.head.state_dict())
    for ep in range(1, args.epocas + 1):
        modelo.head.train()
        for fb, yb in loader:
            opt.zero_grad(); crit(modelo.head(fb.to(device)), yb.to(device)).backward(); opt.step()
        modelo.head.eval()
        with torch.no_grad():
            pv = torch.argmax(modelo.head(F_val.to(device)), dim=1).cpu().numpy()
        va = float((pv == y_val.numpy()).mean())
        if va > mejor_val:
            mejor_val, mejor_head = va, copy.deepcopy(modelo.head.state_dict())
        if ep % 20 == 0:
            print(f"  epoca {ep}/{args.epocas} - val_regimen: {va:.4f}")
    modelo.head.load_state_dict(mejor_head)

    # ===== Evaluacion UNICA final sobre el Test (mangos no vistos) =====
    modelo.head.eval()
    with torch.no_grad():
        p_test = torch.argmax(modelo.head(F_test.to(device)), dim=1).cpu().numpy()
    acc_test, recall_test = _acc_recall(y_test.numpy(), p_test)
    veredicto = evaluar_gate(torch.tensor(y_test.numpy()), torch.tensor(p_test))

    print(f"\n=== ESTADO TERMICO — Test (mangos no vistos, una sola vez) ===")
    print(f"Val regimen (seleccion): {mejor_val*100:.1f}%")
    print(f"Test regimen accuracy:   {acc_test*100:.1f}%")
    print(f"recall [alta, promedio, defecto]: [{', '.join(f'{r:.2f}' for r in recall_test)}]")
    print(f"[GATE {'APROBADO' if veredicto['aprobado'] else 'BLOQUEADO'}] {veredicto['motivo']}")

    torch.save(modelo.state_dict(), PESOS_SALIDA)
    print(f"\nModelo de produccion guardado (API-compatible): {PESOS_SALIDA}")

    RESULTADO_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO_JSON.write_text(json.dumps({
        "tarea": "clasificacion de REGIMEN TERMICO (frio/medio/caliente)",
        "metodo": "augmentation de temperatura + reetiquetado fisico determinista; "
                  "split por mango fisico (Train/Val/Test disjuntos); seleccion por Val",
        "k_variantes": args.k,
        "val_regimen_accuracy": round(mejor_val, 4),
        "test_regimen_accuracy": round(acc_test, 4),
        "recall_por_clase": {c: round(float(r), 4) for c, r in zip(CLASS_NAMES, recall_test)},
        "gate": veredicto,
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado escrito en: {RESULTADO_JSON}")
    print(f"Listo en {time.time()-t0:.1f}s (CPU).")


if __name__ == "__main__":
    main()
