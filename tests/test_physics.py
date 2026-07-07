"""
Assertions sobre la coherencia del modelo espacial estatico (PEDTE).

[Plan de Contingencia F1] Se eliminaron los tests de la dinamica temporal
(temperatura_newton / coeficiente_enfriamiento): el dato es un unico frame, no
una secuencia, y el modulo ya no calcula derivadas dT/dt.
"""
import numpy as np
import pytest

from src.domain_physics.thermal_inertia import (
    gradiente_termico_anomalo,
    clasificar_anomalia,
    normalizar_itr_local,
    mapa_itr_local,
    gradiente_centro_periferia,
)


def test_gradiente_termico_anomalo_es_no_negativo():
    referencia = np.array([0.24, 0.242, 0.239])
    sospechoso = np.array([0.261, 0.243, 0.220])
    delta = gradiente_termico_anomalo(referencia, sospechoso)
    assert np.all(delta >= 0)


def test_clasificar_anomalia_detecta_sobre_el_umbral():
    delta = np.array([0.02, 0.03, 0.11])
    assert clasificar_anomalia(delta, umbral_alerta_itr=0.05) is True


def test_clasificar_anomalia_no_detecta_bajo_el_umbral():
    delta = np.array([0.01, 0.02, 0.03])
    assert clasificar_anomalia(delta, umbral_alerta_itr=0.05) is False


def test_itr_local_esta_en_rango_unitario():
    """La normalizacion Min-Max local siempre produce un mapa en [0, 1]."""
    rng = np.random.default_rng(0)
    imagen = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    itr = normalizar_itr_local(imagen)
    assert itr.min() >= 0.0 and itr.max() <= 1.0 + 1e-6
    assert itr.shape == (32, 32)


def test_itr_local_es_invariante_a_shift_global_de_brillo():
    """
    Al auto-referenciarse por imagen, un corrimiento uniforme de brillo (mala
    calibracion global del sensor) no altera el ITR relativo -> robustez F3.
    """
    rng = np.random.default_rng(1)
    base = rng.integers(20, 200, size=(24, 24, 3), dtype=np.uint8)
    itr_a = normalizar_itr_local(base)
    itr_b = normalizar_itr_local(np.clip(base.astype(int) + 20, 0, 255).astype(np.uint8))
    # No exige igualdad exacta (el clip satura extremos) pero sí alta correlacion.
    assert np.corrcoef(itr_a.ravel(), itr_b.ravel())[0, 1] > 0.95


def test_gradiente_centro_periferia_positivo_si_centro_mas_caliente():
    """Un fruto con centro caliente y borde frio debe dar gradiente positivo."""
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / (n / 2)
    mapa_itr = np.clip(1.0 - dist, 0, 1).astype(np.float32)  # 1 en centro, 0 en borde
    mascara = mapa_itr > 0.1
    assert gradiente_centro_periferia(mapa_itr, mascara) > 0
