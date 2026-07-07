"""
Baseline TRIVIAL de anomalía (guardarraíl anti-fraude).

Un "detector" que usa una sola feature —el ITR medio (brillo térmico medio)— y
puntúa la anomalía como la desviación absoluta respecto del brillo medio de lo
NORMAL. Sin PEDTE, sin covarianza, sin textura.

Su única razón de existir: **si el detector real no lo supera de forma clara sobre
dato real, no aporta nada** y así debe reportarse. Ver
`src/anomaly/detector.py` y `docs/INTEGRIDAD.md` §3.
"""
import numpy as np


class ScoreTrivialBrillo:
    def __init__(self, cuantil_umbral=0.95):
        self.cuantil_umbral = float(cuantil_umbral)

    def fit(self, itr_medios_normales):
        v = np.asarray(itr_medios_normales, dtype=np.float64).ravel()
        self.centro_ = float(np.median(v))
        scores = np.abs(v - self.centro_)
        self.umbral_ = float(np.quantile(scores, self.cuantil_umbral))
        return self

    def score(self, itr_medios):
        v = np.asarray(itr_medios, dtype=np.float64).ravel()
        return np.abs(v - self.centro_)

    def predict_anomalia(self, itr_medios):
        return self.score(itr_medios) > self.umbral_
