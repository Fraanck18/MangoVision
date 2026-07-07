"""
Detección de ANOMALÍAS térmicas para control de calidad (enfoque honesto).

El proyecto NO diagnostica una patología: mide cuánto se desvía la firma térmica de
un fruto respecto de la población NORMAL, y esa desviación es el indicador de calidad
(ver `docs/INTEGRIDAD.md` y la nota de Obsidian "Deteccion de Anomalias").

Regla de oro implementada aquí:
  * El modelo de normalidad se entrena SOLO con mangos buenos REALES, definidos por
    un criterio EXTERNO a la propia imagen térmica (etiqueta humana de calidad
    `alta_calidad`), no por su apariencia térmica -> evita la circularidad.
  * Las imágenes generadas por la física son SONDAS declaradas, nunca verdad de
    evaluación.
  * Toda métrica se reporta junto al BASELINE TRIVIAL de brillo.
"""
from src.anomaly.features import vector_pedte, itr_medio_imagen, NOMBRES_FEATURES
from src.anomaly.detector import DetectorAnomalias
from src.anomaly.baseline_trivial import ScoreTrivialBrillo

__all__ = [
    "vector_pedte",
    "itr_medio_imagen",
    "NOMBRES_FEATURES",
    "DetectorAnomalias",
    "ScoreTrivialBrillo",
]
