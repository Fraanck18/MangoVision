"""
Verificacion anti-hardcoding: garantiza que evaluar_convergencia_real calcule
las metricas dinamicamente a partir de los tensores recibidos, en vez de
devolver valores constantes independientes de la entrada.
"""
import torch
import pytest

from src.models.evaluator import evaluar_convergencia_real, UMBRAL_MINIMO_ACCURACY


def test_metricas_perfectas_cuando_prediccion_es_exacta():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = y_true.clone()
    df_metricas = evaluar_convergencia_real(y_true, y_pred)
    assert df_metricas.loc["accuracy", "precision"] == pytest.approx(1.0)


def test_metricas_cambian_con_la_entrada_real_no_estan_quemadas():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 2)
    y_pred_perfecta = y_true.clone()
    y_pred_con_un_error = y_true.clone()
    y_pred_con_un_error[5] = 0  # una sola muestra mal clasificada (accuracy = 0.95, > umbral)

    df_perfecta = evaluar_convergencia_real(y_true, y_pred_perfecta)
    df_con_error = evaluar_convergencia_real(y_true, y_pred_con_un_error)

    assert df_perfecta.loc["accuracy", "precision"] != df_con_error.loc["accuracy", "precision"]


def test_bloquea_paso_a_produccion_bajo_el_umbral_minimo():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred_deficiente = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    with pytest.raises(AssertionError):
        evaluar_convergencia_real(y_true, y_pred_deficiente)


def test_umbral_minimo_configurado_en_estandar_industrial():
    assert UMBRAL_MINIMO_ACCURACY == pytest.approx(0.90)
