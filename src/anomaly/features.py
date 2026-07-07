"""
Features para el detector de anomalías: PEDTE (físicas, en ITR relativo) + la feature
TRIVIAL de brillo para el guardarraíl.

Reutiliza `extraer_perfil_espacial_termico` (src/domain_physics/thermal_inertia.py):
el perfil espacial estático de intensidad térmica relativa que el proyecto ya calcula
sobre un único frame (sin dinámica dT/dt, F1).
"""
import numpy as np

from src.domain_physics.thermal_inertia import extraer_perfil_espacial_termico
from src.augmentation.paleta_ironbow import ironbow_a_itr

# Orden FIJO de las features PEDTE (mismo que usa honest_grouped_cv.py).
NOMBRES_FEATURES = [
    "itr_medio_fruto",
    "itr_std_fruto",
    "fraccion_area_anomala_baja",
    "desviacion_maxima_baja",
    "gradiente_centro_periferia",
]


def vector_pedte(imagen_pil):
    """
    Extrae el vector PEDTE (5 features físicas en ITR relativo) de una imagen térmica.
    Los NaN (fruto no detectado) se sustituyen por 0 -> vector siempre finito.
    """
    perfil = extraer_perfil_espacial_termico(imagen_pil.convert("RGB"))
    vec = np.array([perfil[n] for n in NOMBRES_FEATURES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def itr_medio_imagen(imagen_pil):
    """
    Feature TRIVIAL para el baseline anti-fraude: el ITR medio (brillo térmico medio)
    de toda la imagen. Un solo número, sin PEDTE, sin textura.
    """
    itr = ironbow_a_itr(np.asarray(imagen_pil.convert("RGB")))
    return float(itr.mean())
