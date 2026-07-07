"""
Detector de anomalías one-class: aprende el modelo de NORMALIDAD y puntúa la
desviación de cada fruto.

Dos métodos (ambos entrenan SOLO con normal):
  * 'mahalanobis' (por defecto): distancia de Mahalanobis a la distribución normal,
    con covarianza Ledoit-Wolf (shrinkage) -> estable con pocas features/muestras.
  * 'isolation': IsolationForest sobre las features normales.

El score es SIEMPRE "mayor = más anómalo". El umbral se calibra como un cuantil alto
de los scores del propio conjunto normal de entrenamiento (p.ej. 0.95 -> ~5% de
falsas alarmas esperadas por construcción; la falsa alarma REAL se mide sobre normal
held-out, ver scripts/deteccion_anomalias.py).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest


class DetectorAnomalias:
    def __init__(self, metodo="mahalanobis", cuantil_umbral=0.95, random_state=42):
        if metodo not in ("mahalanobis", "isolation"):
            raise ValueError(f"metodo desconocido: {metodo!r}")
        self.metodo = metodo
        self.cuantil_umbral = float(cuantil_umbral)
        self.random_state = random_state

    def fit(self, X_normal):
        """Ajusta el modelo de normalidad SOLO con features de mangos buenos reales."""
        X_normal = np.asarray(X_normal, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X_normal)
        Xs = self.scaler_.transform(X_normal)

        if self.metodo == "mahalanobis":
            self.modelo_ = LedoitWolf().fit(Xs)
        else:
            self.modelo_ = IsolationForest(
                n_estimators=200, contamination="auto", random_state=self.random_state
            ).fit(Xs)

        scores_train = self._score_estandarizado(Xs)
        self.umbral_ = float(np.quantile(scores_train, self.cuantil_umbral))
        # Cuantiles del normal para mapear score -> bandas de calidad (ordinal).
        self.q_bandas_ = np.quantile(scores_train, [0.60, 0.90])
        return self

    def _score_estandarizado(self, Xs):
        if self.metodo == "mahalanobis":
            return self.modelo_.mahalanobis(Xs)          # distancia^2, mayor = más anómalo
        return -self.modelo_.score_samples(Xs)           # invertido: mayor = más anómalo

    def score(self, X):
        """Score de anomalía (mayor = más anómalo) para cada fila de X."""
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self._score_estandarizado(Xs)

    def predict_anomalia(self, X):
        """True si el score supera el umbral calibrado (marcado como anómalo)."""
        return self.score(X) > self.umbral_

    def bandas_calidad(self, X):
        """
        Mapea el score a una banda ORDINAL de desviación (no un diagnóstico):
          score bajo  -> 0 = alta_calidad  (dentro de lo normal)
          score medio -> 1 = calidad_promedio
          score alto  -> 2 = defectuoso    (firma anómala)
        Los cortes son cuantiles del score del conjunto normal (calibrados en fit).
        """
        s = self.score(X)
        return np.digitize(s, self.q_bandas_)
