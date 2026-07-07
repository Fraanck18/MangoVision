"""
Evaluación HONESTA del detector de anomalías térmicas para control de calidad.

Enfoque (ver docs/INTEGRIDAD.md y la nota "Deteccion de Anomalias"):
  * NORMAL = mangos `alta_calidad` (etiqueta HUMANA de calidad = criterio EXTERNO a
    la imagen térmica -> no circular). El detector se entrena SOLO con su firma PEDTE.
  * ANOMALÍA real = mangos `defectuoso` (y `calidad_promedio` como desviación media).
  * Métricas, todas honestas y reproducibles:
      - Tasa de FALSAS ALARMAS sobre normal real HELD-OUT por mango (LOGO): de N
        mangos buenos no vistos, ¿qué fracción se marca por error?
      - AUC de separación normal-vs-defectuoso real (sin fabricar nada).
      - Todo ello CON el BASELINE TRIVIAL de brillo al lado (guardarraíl).
  * SONDAS FÍSICAS declaradas: anomalías sintéticas (frío/calor) para test de
    sensibilidad -> NO cuentan como verdad de evaluación, solo estrés declarado.

Uso:
    python scripts/deteccion_anomalias.py
    python scripts/deteccion_anomalias.py --metodo isolation
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.anomaly import vector_pedte, itr_medio_imagen, DetectorAnomalias, ScoreTrivialBrillo
from src.domain_physics.etiquetado_fisico import base_canonica_fruto, simular_estres_termico, simular_enfriamiento
from src.augmentation.paleta_ironbow import mapa_a_ironbow

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
RESULTADO_JSON = RAIZ / "web" / "data" / "anomalias.json"
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")

CLASE_NORMAL = "alta_calidad"
CLASE_ANOMALIA = "defectuoso"
CLASE_INTERMEDIA = "calidad_promedio"


def _grupo(nombre):
    m = PATRON_GRUPO.search(nombre)
    return m.group(1) if m else Path(nombre).stem


def _recolectar():
    """Devuelve dict clase -> lista de (ruta, grupo) del dato REAL (train+test)."""
    items = defaultdict(list)
    for particion in ("train", "test"):
        for clase in (CLASE_NORMAL, CLASE_ANOMALIA, CLASE_INTERMEDIA):
            carpeta = PROCESADOS / particion / clase
            if carpeta.exists():
                for r in carpeta.glob("*"):
                    if r.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        items[clase].append((r, _grupo(r.name)))
    return items


def _features(rutas_grupos):
    """Extrae (X_pedte, x_itr_trivial, grupos) para una lista de (ruta, grupo)."""
    X, itr, grupos = [], [], []
    for ruta, g in rutas_grupos:
        img = Image.open(ruta).convert("RGB")
        X.append(vector_pedte(img))
        itr.append(itr_medio_imagen(img))
        grupos.append(g)
    return np.array(X), np.array(itr), np.array(grupos)


def _sondas_fisicas(n=60, seed=42):
    """
    SONDAS declaradas: anomalías térmicas SINTÉTICAS (estrés=calor, enfriamiento=frío)
    sobre substrato canónico. Sirven para medir SENSIBILIDAD del detector, NO como
    verdad de evaluación. Devuelve (X_pedte, x_itr).
    """
    rng = np.random.default_rng(seed)
    X, itr = [], []
    for _ in range(n):
        base = base_canonica_fruto(224, rng)
        gen = simular_estres_termico if rng.random() < 0.5 else simular_enfriamiento
        itr_map, _ = (gen(base, rng) if gen is simular_estres_termico else gen(base))
        img = Image.fromarray(mapa_a_ironbow(itr_map), mode="RGB")
        X.append(vector_pedte(img))
        itr.append(itr_medio_imagen(img))
    return np.array(X), np.array(itr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", choices=["mahalanobis", "isolation"], default="mahalanobis")
    args = ap.parse_args()

    items = _recolectar()
    Xn, itrn, gn = _features(items[CLASE_NORMAL])       # NORMAL (real)
    Xa, itra, _ = _features(items[CLASE_ANOMALIA])       # ANOMALÍA real
    grupos_normales = sorted(set(gn))
    print(f"NORMAL (alta_calidad): {len(Xn)} imgs, {len(grupos_normales)} mangos")
    print(f"ANOMALÍA (defectuoso): {len(Xa)} imgs, {len(set(items[CLASE_ANOMALIA][i][1] for i in range(len(items[CLASE_ANOMALIA]))))} mangos\n")

    # ---- LOGO por mango NORMAL: falsas alarmas + AUC (detector y trivial) ----
    fa_det, fa_triv, auc_det, auc_triv = [], [], [], []
    for g in grupos_normales:
        tr = gn != g            # entrena con los OTROS mangos normales
        te = gn == g            # normal held-out
        if tr.sum() < 3 or te.sum() == 0:
            continue

        det = DetectorAnomalias(metodo=args.metodo).fit(Xn[tr])
        triv = ScoreTrivialBrillo().fit(itrn[tr])

        # Falsas alarmas sobre normal held-out.
        fa_det.append(float(det.predict_anomalia(Xn[te]).mean()))
        fa_triv.append(float(triv.predict_anomalia(itrn[te]).mean()))

        # AUC: normal held-out (0) vs TODA la anomalía real (1). La anomalía nunca se
        # entrena, así que puede ir entera en cada fold sin fuga.
        y = np.r_[np.zeros(te.sum()), np.ones(len(Xa))]
        s_det = np.r_[det.score(Xn[te]), det.score(Xa)]
        s_triv = np.r_[triv.score(itrn[te]), triv.score(itra)]
        if len(np.unique(y)) == 2:
            auc_det.append(roc_auc_score(y, s_det))
            auc_triv.append(roc_auc_score(y, s_triv))

    # ---- Sondas físicas declaradas (sensibilidad, NO verdad de evaluación) ----
    det_full = DetectorAnomalias(metodo=args.metodo).fit(Xn)
    Xs, itrs = _sondas_fisicas()
    sens_det = float(det_full.predict_anomalia(Xs).mean())

    def _ms(v):
        return (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) if v else (None, None)

    fa_d, fa_d_s = _ms(fa_det); fa_t, _ = _ms(fa_triv)
    auc_d, auc_d_s = _ms(auc_det); auc_t, _ = _ms(auc_triv)

    print("=== RESULTADO HONESTO (LOGO por mango normal) ===")
    print(f"Falsas alarmas sobre normal real held-out: detector {fa_d*100:.1f}% ± {fa_d_s*100:.1f}"
          f"  |  trivial {fa_t*100:.1f}%")
    if auc_d is not None:
        print(f"AUC separación normal-vs-defectuoso REAL:   detector {auc_d:.3f} ± {auc_d_s:.3f}"
              f"  |  trivial {auc_t:.3f}")
        veredicto = ("SUPERA" if auc_d > auc_t + 0.02 else
                     "NO SUPERA" if auc_d < auc_t - 0.02 else "EMPATA con")
        print(f"\n>> El detector PEDTE {veredicto} el baseline trivial de brillo.")
    print(f"\nSensibilidad a SONDAS físicas declaradas (sintéticas, no evaluación): "
          f"{sens_det*100:.1f}% flageadas")

    print("\nLectura honesta:")
    print("  - Falsas alarmas y AUC se miden SOLO sobre fruta REAL, held-out por mango.")
    print("  - Si el detector no supera claramente al trivial, NO aporta (guardarraíl).")
    print("  - Las sondas físicas son estrés declarado; NO son verdad de evaluación.")
    print("  - N de mangos es pequeño: los números tienen alta varianza (se reporta).")

    RESULTADO_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTADO_JSON.write_text(json.dumps({
        "enfoque": "deteccion de anomalias termicas; normal=alta_calidad (etiqueta humana externa)",
        "metodo": args.metodo,
        "n_mangos_normales": len(grupos_normales),
        "falsas_alarmas_detector": fa_d, "falsas_alarmas_trivial": fa_t,
        "auc_detector": auc_d, "auc_trivial": auc_t,
        "detector_supera_trivial": (auc_d is not None and auc_t is not None and auc_d > auc_t + 0.02),
        "sensibilidad_sondas_fisicas_declaradas": round(sens_det, 4),
        "nota": "sondas fisicas = sinteticas declaradas, NO verdad de evaluacion",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en: {RESULTADO_JSON}")


if __name__ == "__main__":
    main()
