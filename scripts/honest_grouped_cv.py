"""
Evaluacion HONESTA por validacion cruzada agrupada (GroupKFold).

El problema del 100% reportado antes: el test set provenia de solo 5 mangos
fisicos. Aqui se agrupa por foto base (todas las variantes aumentadas de un
mismo mango forman un grupo) y se hace CV de modo que cada mango fisico cae
en el fold de prueba exactamente una vez, y nunca se evalua sobre un mango
que el modelo vio entrenando. Esto estima la generalizacion real a mangos
NUEVOS, que es lo que importa en produccion.

Se evalua con las features fisicas finales (Newton + filtro de mediana +
ecualizacion), que son las tecnicas definitivas implementadas.

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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.domain_physics.thermal_inertia import extraer_anomalia_termica_imagen

PROCESADOS = RAIZ / "dataset_mangos" / "processed_binary"
SALIDA_JSON = RAIZ / "web" / "data" / "cv_honesta.json"

PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def extraer_grupo(nombre_archivo):
    """Identifica el mango fisico de origen a partir del nombre del archivo."""
    match = PATRON_GRUPO.search(nombre_archivo)
    return match.group(1) if match else nombre_archivo


def recolectar():
    rutas, etiquetas, grupos = [], [], []
    for particion in ("train", "test"):
        for clase_idx, clase in enumerate(("defectuoso", "sano")):
            carpeta = PROCESADOS / particion / clase
            if not carpeta.exists():
                continue
            for ruta in sorted(carpeta.glob("*")):
                rutas.append(ruta)
                etiquetas.append(clase_idx)
                grupos.append(extraer_grupo(ruta.name))
    return rutas, np.array(etiquetas), np.array(grupos)


def main():
    rutas, y, grupos = recolectar()
    print(f"Total imagenes: {len(rutas)}  |  Grupos (mangos fisicos): {len(set(grupos))}")

    print("Extrayendo features fisicas (Newton + mediana + ecualizacion)...")
    X = []
    for ruta in rutas:
        f = extraer_anomalia_termica_imagen(Image.open(ruta).convert("RGB"))
        X.append([
            f["temp_media_fruto"], f["temp_std_fruto"],
            f["fraccion_area_anomala_fria"], f["desviacion_maxima_fria"],
        ])
    X = np.nan_to_num(np.array(X))

    n_grupos = len(set(grupos))
    n_splits = min(5, n_grupos)
    gkf = GroupKFold(n_splits=n_splits)

    accs, precs, recs, f1s = [], [], [], []
    for i, (idx_train, idx_test) in enumerate(gkf.split(X, y, grupos), 1):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[idx_train], y[idx_train])
        y_pred = clf.predict(X[idx_test])

        acc = accuracy_score(y[idx_test], y_pred)
        p, r, f, _ = precision_recall_fscore_support(
            y[idx_test], y_pred, average="binary", pos_label=0, zero_division=0
        )
        accs.append(acc); precs.append(p); recs.append(r); f1s.append(f)
        grupos_test = sorted(set(grupos[idx_test]))
        print(f"  Fold {i}: acc={acc:.3f}  (mangos en test: {grupos_test})")

    resumen = {
        "metodo": "GroupKFold por foto base (cada mango fisico en test exactamente una vez)",
        "features": "fisicas (Ley de Newton + filtro mediana + ecualizacion de histograma)",
        "n_splits": n_splits,
        "n_grupos": n_grupos,
        "accuracy_media": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "precision_defectuoso_media": round(float(np.mean(precs)), 4),
        "recall_defectuoso_media": round(float(np.mean(recs)), 4),
        "f1_defectuoso_media": round(float(np.mean(f1s)), 4),
        "accuracy_por_fold": [round(float(a), 4) for a in accs],
    }

    print("\n=== RESULTADO HONESTO (generalizacion a mangos nuevos) ===")
    print(f"Accuracy: {resumen['accuracy_media']*100:.1f}% +/- {resumen['accuracy_std']*100:.1f}%")
    print(f"Recall defectuoso: {resumen['recall_defectuoso_media']*100:.1f}%")

    SALIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_JSON, "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    print(f"\nGuardado en: {SALIDA_JSON}")


if __name__ == "__main__":
    main()
