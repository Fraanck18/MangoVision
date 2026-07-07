"""
Tests del etiquetado programatico por firma termica (weak supervision) y del motor
de simulacion parametrica.

Propiedad clave: las etiquetas las fija una REGLA DETERMINISTA sobre la firma
termica relativa (ITR), no un modelo -> sin concordancia circular. Y cada
transformador parametrico empuja el substrato canonico a su regimen previsto.
"""
import numpy as np

from src.config import CLASS_NAMES
from src.domain_physics.etiquetado_fisico import (
    etiquetar_por_firma_termica,
    descriptores_firma_termica,
    base_canonica_fruto,
    simular_estres_termico,
    simular_enfriamiento,
    firma_madurez,
    GENERADORES_POR_CLASE,
)


def test_labeler_es_determinista():
    """La misma matriz ITR produce SIEMPRE la misma etiqueta (no hay aleatoriedad)."""
    rng = np.random.default_rng(0)
    itr = base_canonica_fruto(64, rng)
    e1 = etiquetar_por_firma_termica(itr)
    e2 = etiquetar_por_firma_termica(itr)
    assert e1 == e2 and e1 in CLASS_NAMES


def test_estres_termico_da_defectuoso():
    for s in range(20):
        rng = np.random.default_rng(s)
        base = base_canonica_fruto(64, rng)
        _, etiqueta = simular_estres_termico(base, rng)
        assert etiqueta == "defectuoso"


def test_enfriamiento_da_calidad_promedio():
    for s in range(20):
        rng = np.random.default_rng(s)
        base = base_canonica_fruto(64, rng)
        _, etiqueta = simular_enfriamiento(base)
        assert etiqueta == "calidad_promedio"


def test_firma_madurez_da_alta_calidad():
    for s in range(20):
        rng = np.random.default_rng(s)
        base = base_canonica_fruto(64, rng)
        _, etiqueta = firma_madurez(base, rng)
        assert etiqueta == "alta_calidad"


def test_generadores_cubren_las_tres_clases():
    assert set(GENERADORES_POR_CLASE) == set(CLASS_NAMES)


def test_transformadores_preservan_rango_itr_unitario():
    rng = np.random.default_rng(3)
    base = base_canonica_fruto(64, rng)
    for gen in GENERADORES_POR_CLASE.values():
        itr, _ = gen(base, rng)
        assert itr.min() >= 0.0 and itr.max() <= 1.0
        assert itr.shape == base.shape


def test_descriptores_son_relativos_no_grados():
    """itr_medio y fraccion_caliente estan en [0,1] (ITR relativo), no en °C."""
    rng = np.random.default_rng(1)
    d = descriptores_firma_termica(base_canonica_fruto(64, rng))
    assert 0.0 <= d["itr_medio"] <= 1.0
    assert 0.0 <= d["fraccion_caliente"] <= 1.0
