"""
Evaluacion HONESTA por validacion cruzada Leave-One-Group-Out (LOGO) RESTRINGIDO A
PROCEDENCIA HUMANA.

[Plan de Contingencia F6] Antes se usaba GroupKFold con muy pocos mangos
fisicos, lo que generaba folds de test minusculos y metricas de altisima
varianza. LOGO corrige esto: con N mangos fisicos, el modelo se entrena N veces
y en cada iteracion se evalua sobre las fotos del UNICO mango dejado fuera.

[Fallo #2 — sin fuga por auto-etiquetas]  El pseudo-etiquetado (FLIR), la
augmentation fisica y las muestras sinteticas son tecnicas de EXPANSION DEL TRAIN
(Noisy Student / weak supervision). La regla de oro es: los datos no-humanos entran
SOLO al lado de entrenamiento; la evaluacion se hace SIEMPRE contra verdad humana.
Antes, `recolectar()` mezclaba todo en un pool LOGO plano y un mango FLIR pseudo-
etiquetado podia caer como held-out -> su `y_true` era la etiqueta que el propio
modelo adivino -> concordancia circular. Ahora:

  * Se etiqueta la PROCEDENCIA de cada imagen (humana | pseudo | phys_aug | gan |
    synthetic_physics), derivada del nombre de archivo.
  * LOGO RESTRINGIDO: el grupo held-out (test) es SIEMPRE de procedencia humana y
    su test contiene SOLO imagenes humanas. Los datos pseudo/aumentados/sinteticos
    pueden estar en el TRAIN de cualquier fold, nunca ser el grupo evaluado.
  * Se reporta `n_grupos_humanos_evaluables` aparte de `n_grupos`, para dejar
    explicito contra que se midio.

Se evalua con el Perfil Espacial de Distribucion Termica Estatica (PEDTE):
features de ITR relativo (Min-Max local + Gaussiano) sobre un solo frame, sin
dinamica temporal (ver F1/F3 en src/domain_physics/thermal_inertia.py).

Uso:
    python scripts/honest_grouped_cv.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.domain_physics.thermal_inertia import extraer_perfil_espacial_termico

PROCESADOS = RAIZ / "dataset_mangos" / "processed_binary"
SALIDA_JSON = RAIZ / "web" / "data" / "cv_honesta.json"

PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def extraer_grupo(nombre_archivo):
    """Identifica el mango fisico de origen a partir del nombre del archivo."""
    match = PATRON_GRUPO.search(nombre_archivo)
    return match.group(1) if match else nombre_archivo


def procedencia_de(nombre_archivo):
    """
    Clasifica la PROCEDENCIA de una imagen por su nombre. Solo 'humana' es verdad
    verificada por personas (Roboflow); el resto son tecnicas de expansion del
    train que NUNCA deben usarse como ground-truth de evaluacion.
    """
    n = nombre_archivo.lower()
    if n.startswith("flir_pseudo"):
        return "pseudo"
    if n.startswith("phys_"):
        return "phys_aug"
    if n.startswith("gan_"):
        return "gan"
    if n.startswith(("sint_", "synthphys_", "synthetic")):
        return "synthetic_physics"
    return "humana"


def recolectar():
    rutas, etiquetas, grupos, procedencias = [], [], [], []
    for particion in ("train", "test"):
        for clase_idx, clase in enumerate(("defectuoso", "sano")):
            carpeta = PROCESADOS / particion / clase
            if not carpeta.exists():
                continue
            for ruta in sorted(carpeta.glob("*")):
                rutas.append(ruta)
                etiquetas.append(clase_idx)
                grupos.append(extraer_grupo(ruta.name))
                procedencias.append(procedencia_de(ruta.name))
    return rutas, np.array(etiquetas), np.array(grupos), np.array(procedencias)


def folds_logo_restringido(grupos, procedencias):
    """
    Genera (idx_train, idx_test, grupo_held_out) para un LOGO restringido: el grupo
    evaluado es SIEMPRE humano y su test contiene SOLO imagenes humanas; el train es
    todo lo demas (cualquier procedencia) salvo el grupo held-out completo (incluidas
    sus variantes aumentadas, que comparten token de grupo -> anti-fuga).
    """
    indices = np.arange(len(grupos))
    grupos_humanos = sorted(set(grupos[procedencias == "humana"]))
    for g in grupos_humanos:
        idx_test = indices[(grupos == g) & (procedencias == "humana")]
        idx_train = indices[grupos != g]
        if len(idx_test) and len(idx_train):
            yield idx_train, idx_test, g


def main():
    rutas, y, grupos, procedencias = recolectar()
    n_humanas = int((procedencias == "humana").sum())
    print(f"Total imagenes: {len(rutas)} (humanas: {n_humanas}, "
          f"expansion train: {len(rutas) - n_humanas})  |  "
          f"Grupos totales: {len(set(grupos))}")

    print("Extrayendo features PEDTE (ITR local + Gaussiano + gradiente centro-periferia)...")
    X = []
    for ruta in rutas:
        f = extraer_perfil_espacial_termico(Image.open(ruta).convert("RGB"))
        X.append([
            f["itr_medio_fruto"], f["itr_std_fruto"],
            f["fraccion_area_anomala_baja"], f["desviacion_maxima_baja"],
            f["gradiente_centro_periferia"],
        ])
    X = np.nan_to_num(np.array(X))

    grupos_humanos = sorted(set(grupos[procedencias == "humana"]))
    accs, precs, recs, f1s = [], [], [], []
    folds = list(folds_logo_restringido(grupos, procedencias))
    for i, (idx_train, idx_test, g) in enumerate(folds, 1):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[idx_train], y[idx_train])
        y_pred = clf.predict(X[idx_test])

        acc = accuracy_score(y[idx_test], y_pred)
        p, r, f, _ = precision_recall_fscore_support(
            y[idx_test], y_pred, average="binary", pos_label=0, zero_division=0
        )
        accs.append(acc); precs.append(p); recs.append(r); f1s.append(f)
        n_exp = len(idx_train) - int((procedencias[idx_train] == "humana").sum())
        print(f"  Fold {i}/{len(folds)}: acc={acc:.3f}  (mango HUMANO en test: {g}, "
              f"train={len(idx_train)} de los cuales {n_exp} son expansion)")

    resumen = {
        "metodo": ("LeaveOneGroupOut RESTRINGIDO: solo grupos de procedencia humana "
                   "se evaluan; pseudo/aumentado/sintetico solo expanden el train"),
        "features": "PEDTE (ITR Min-Max local + Gaussiano 3x3 + gradiente centro-periferia)",
        "n_splits": len(folds),
        "n_grupos": int(len(set(grupos))),
        "n_grupos_humanos_evaluables": len(grupos_humanos),
        "n_imagenes_humanas": n_humanas,
        "n_imagenes_expansion_train": int(len(rutas) - n_humanas),
        "accuracy_media": round(float(np.mean(accs)), 4) if accs else None,
        "accuracy_std": round(float(np.std(accs)), 4) if accs else None,
        "precision_defectuoso_media": round(float(np.mean(precs)), 4) if precs else None,
        "recall_defectuoso_media": round(float(np.mean(recs)), 4) if recs else None,
        "f1_defectuoso_media": round(float(np.mean(f1s)), 4) if f1s else None,
        "accuracy_por_fold": [round(float(a), 4) for a in accs],
    }

    print("\n=== RESULTADO HONESTO (generalizacion a mangos nuevos, SOLO verdad humana) ===")
    print(f"Grupos humanos evaluados: {resumen['n_grupos_humanos_evaluables']}")
    if accs:
        print(f"Accuracy: {resumen['accuracy_media']*100:.1f}% +/- {resumen['accuracy_std']*100:.1f}%")
        print(f"Recall defectuoso: {resumen['recall_defectuoso_media']*100:.1f}%")

    SALIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_JSON, "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    print(f"\nGuardado en: {SALIDA_JSON}")


if __name__ == "__main__":
    main()
