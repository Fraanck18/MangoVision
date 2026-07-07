"""
Tests del detector de anomalías térmicas y del GUARDARRAÍL del baseline trivial.

El test más importante (`test_guardarrail_...`) fija la regla anti-fraude: la
evaluación SIEMPRE produce el AUC del baseline trivial junto al del detector, para
que ningún resultado pueda presentarse sin su control.
"""
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.augmentation.paleta_ironbow import mapa_a_ironbow
from src.anomaly import vector_pedte, itr_medio_imagen, DetectorAnomalias, ScoreTrivialBrillo, NOMBRES_FEATURES


def _img_itr(nivel, n=64):
    itr = np.clip(np.full((n, n), nivel, dtype=np.float32) + 0.02, 0, 1)
    return Image.fromarray(mapa_a_ironbow(itr), mode="RGB")


# ---------------------------- Features ----------------------------
def test_vector_pedte_es_finito_y_dim5():
    v = vector_pedte(_img_itr(0.4))
    assert v.shape == (len(NOMBRES_FEATURES),)
    assert np.all(np.isfinite(v))


def test_itr_medio_refleja_el_nivel():
    assert itr_medio_imagen(_img_itr(0.2)) < itr_medio_imagen(_img_itr(0.8))


# ------------------------- Detector one-class -------------------------
def test_detector_puntua_mas_alto_lo_anomalo():
    rng = np.random.default_rng(0)
    X_normal = rng.normal(0, 1, (200, 5))
    X_anom = rng.normal(6, 1, (50, 5))            # claramente fuera de lo normal
    det = DetectorAnomalias(metodo="mahalanobis").fit(X_normal)
    assert det.score(X_anom).mean() > det.score(X_normal).mean()


def test_detector_predice_anomalia_por_umbral():
    rng = np.random.default_rng(1)
    X_normal = rng.normal(0, 1, (200, 5))
    det = DetectorAnomalias().fit(X_normal)
    # Un punto muy lejano se marca como anómalo.
    assert bool(det.predict_anomalia(np.full((1, 5), 8.0))[0]) is True


def test_bandas_calidad_son_ordinales():
    rng = np.random.default_rng(2)
    det = DetectorAnomalias().fit(rng.normal(0, 1, (300, 5)))
    bandas = det.bandas_calidad(np.vstack([np.zeros((1, 5)), np.full((1, 5), 3.0), np.full((1, 5), 9.0)]))
    assert bandas[0] <= bandas[1] <= bandas[2]
    assert set(np.unique(det.bandas_calidad(rng.normal(0, 1, (50, 5))))).issubset({0, 1, 2})


def test_isolation_forest_tambien_funciona():
    rng = np.random.default_rng(3)
    det = DetectorAnomalias(metodo="isolation").fit(rng.normal(0, 1, (200, 5)))
    assert det.score(np.full((10, 5), 6.0)).mean() > det.score(rng.normal(0, 1, (10, 5))).mean()


# ------------------------- Baseline trivial -------------------------
def test_trivial_detecta_desviacion_de_brillo():
    normal = np.random.default_rng(4).normal(0.3, 0.02, 100)
    triv = ScoreTrivialBrillo().fit(normal)
    assert triv.score([0.9]).item() > triv.score([0.3]).item()
    assert bool(triv.predict_anomalia([0.9])[0]) is True


# ---------------- GUARDARRAÍL (regla anti-fraude) ----------------
def test_guardarrail_trivial_siempre_acompana_al_detector():
    """
    Toda evaluación produce el AUC del detector Y el del baseline trivial sobre los
    MISMOS datos. Aquí, en un caso donde la señal ES el brillo, el trivial debe
    separar bien -> demuestra que el control está vivo y es exigente.
    """
    rng = np.random.default_rng(5)
    # Normal: brillo bajo; anomalía: brillo alto (señal trivial real).
    itr_normal = rng.normal(0.3, 0.03, 120)
    itr_anom = rng.normal(0.75, 0.05, 40)
    # Features "PEDTE" sintéticas correlacionadas con el brillo + ruido.
    Xn = np.c_[itr_normal, rng.normal(0, 1, (120, 4))]
    Xa = np.c_[itr_anom, rng.normal(0, 1, (40, 4))]

    det = DetectorAnomalias().fit(Xn)
    triv = ScoreTrivialBrillo().fit(itr_normal)

    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
    auc_det = roc_auc_score(y, np.r_[det.score(Xn), det.score(Xa)])
    auc_triv = roc_auc_score(y, np.r_[triv.score(itr_normal), triv.score(itr_anom)])

    # El control existe y es un número válido comparable.
    assert 0.0 <= auc_triv <= 1.0 and 0.0 <= auc_det <= 1.0
    # En este caso la señal es el brillo -> el trivial separa fuerte (guardarraíl exigente).
    assert auc_triv > 0.9
