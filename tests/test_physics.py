"""Assertions sobre la coherencia matematica del modelo de inercia termica."""
import numpy as np
import pytest

from src.domain_physics.thermal_inertia import (
    temperatura_newton,
    coeficiente_enfriamiento,
    gradiente_termico_anomalo,
    clasificar_anomalia,
)


def test_temperatura_converge_a_ambiente_en_el_limite():
    t = np.array([0, 10, 100, 1000])
    temperaturas = temperatura_newton(t, t0=40.0, t_amb=20.0, h=0.05)
    assert temperaturas[0] == pytest.approx(40.0)
    assert temperaturas[-1] == pytest.approx(20.0, abs=1e-2)


def test_temperatura_es_monotona_decreciente_si_t0_mayor_que_ambiente():
    t = np.linspace(0, 50, 20)
    temperaturas = temperatura_newton(t, t0=45.0, t_amb=22.0, h=0.1)
    assert np.all(np.diff(temperaturas) <= 0)


def test_coeficiente_enfriamiento_positivo_con_parametros_validos():
    h = coeficiente_enfriamiento(k_eff=0.5, rho=1050, c_eff=3800, volumen=2e-4, area_superficial=1.5e-2)
    assert h > 0


@pytest.mark.parametrize("rho,c_eff,volumen,area", [(0, 3800, 2e-4, 1.5e-2), (1050, -1, 2e-4, 1.5e-2), (1050, 3800, 0, 1.5e-2)])
def test_coeficiente_enfriamiento_rechaza_propiedades_no_fisicas(rho, c_eff, volumen, area):
    with pytest.raises(ValueError):
        coeficiente_enfriamiento(k_eff=0.5, rho=rho, c_eff=c_eff, volumen=volumen, area_superficial=area)


def test_gradiente_termico_anomalo_es_no_negativo():
    sano = np.array([24.0, 24.2, 23.9])
    sospechoso = np.array([26.1, 24.3, 22.0])
    delta = gradiente_termico_anomalo(sano, sospechoso)
    assert np.all(delta >= 0)


def test_clasificar_anomalia_detecta_sobre_el_umbral():
    delta = np.array([0.2, 0.3, 1.1])
    assert clasificar_anomalia(delta, umbral_alerta_celsius=0.8) is True


def test_clasificar_anomalia_no_detecta_bajo_el_umbral():
    delta = np.array([0.1, 0.2, 0.3])
    assert clasificar_anomalia(delta, umbral_alerta_celsius=0.8) is False
