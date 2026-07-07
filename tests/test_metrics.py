"""
Verificacion del gate RELATIVO y de la auditoria dinamica de metricas.

Dos cosas se garantizan:
  1. Anti-hardcoding: las metricas se calculan desde los tensores recibidos, no
     son constantes.
  2. El gate es RELATIVO a la linea base honesta (no un 0.90 quemado inalcanzable):
     BLOQUEA cuando accuracy < baseline+margen y APRUEBA cuando la supera, y mide
     solo sobre muestras reales.
"""
import torch

from src.models.evaluator import evaluar_convergencia_real, evaluar_gate, leer_baseline_honesta


# --------------------------- Anti-hardcoding ---------------------------
def test_metricas_perfectas_cuando_prediccion_es_exacta():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = y_true.clone()
    df = evaluar_convergencia_real(y_true, y_pred)
    assert df.loc["accuracy", "precision"] == 1.0


def test_metricas_cambian_con_la_entrada_no_estan_quemadas():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 2)
    y_pred_perfecta = y_true.clone()
    y_pred_con_error = y_true.clone()
    y_pred_con_error[5] = 0
    df_perfecta = evaluar_convergencia_real(y_true, y_pred_perfecta)
    df_con_error = evaluar_convergencia_real(y_true, y_pred_con_error)
    assert df_perfecta.loc["accuracy", "precision"] != df_con_error.loc["accuracy", "precision"]


# ------------------------- Gate relativo -------------------------
def test_gate_aprueba_cuando_supera_baseline_mas_margen():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = y_true.clone()  # accuracy 1.0
    v = evaluar_gate(y_true, y_pred, baseline=0.70, margen=0.0)
    assert v["aprobado"] is True
    assert v["accuracy"] == 1.0
    assert v["umbral_usado"] == 0.70


def test_gate_bloquea_cuando_cae_bajo_baseline_mas_margen():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    # todo predicho como 2 -> solo aciertan los tres 2 reales -> accuracy 0.3 < 0.60
    y_pred = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    v = evaluar_gate(y_true, y_pred, baseline=0.60, margen=0.0)
    assert v["aprobado"] is False
    assert v["accuracy"] < v["umbral_usado"]


def test_gate_umbral_es_baseline_mas_margen():
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 1])
    v = evaluar_gate(y_true, y_pred, baseline=0.60, margen=0.05)
    assert v["umbral_usado"] == 0.65  # relativo, no un 0.90 quemado


def test_gate_es_estructurado_no_lanza_excepcion():
    """El gate devuelve un veredicto inspeccionable en vez de reventar (no teatro)."""
    y_true = torch.tensor([0, 1, 2, 0])
    y_pred = torch.tensor([2, 2, 2, 2])  # pesimo, pero NO debe lanzar
    v = evaluar_gate(y_true, y_pred, baseline=0.90)
    assert set(v) >= {"aprobado", "accuracy", "umbral_usado", "baseline", "motivo"}
    assert v["aprobado"] is False


def test_gate_mide_solo_muestras_reales():
    """Con mascara es_real, la metrica ignora la expansion (regla de oro)."""
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    # Las reales (indices 0,1,2) son perfectas; las de expansion (3,4,5) son basura.
    y_pred = torch.tensor([0, 1, 2, 2, 2, 2])
    es_real = torch.tensor([1, 1, 1, 0, 0, 0])
    v = evaluar_gate(y_true, y_pred, baseline=0.70, es_real=es_real)
    assert v["n_reales"] == 3 and v["n_muestras"] == 6
    assert v["accuracy"] == 1.0  # solo se midieron las reales
    assert v["aprobado"] is True


def test_baseline_se_lee_de_disco_o_env(monkeypatch):
    """La baseline sale de disco/env, no de una constante 0.90 quemada."""
    monkeypatch.setenv("MANGO_BASELINE_ACCURACY", "0.61")
    baseline, fuente = leer_baseline_honesta()
    assert baseline == 0.61
    assert "env" in fuente
