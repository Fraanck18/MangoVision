"""
Tests del paso de VERIFICACION de anomalia en el stream y del reporte batch por lote.

Flujo streaming: llega -> verifica (anomalia, con baseline trivial al lado) -> clasifica.
Batch: rollup por lote (no re-clasifica). Los tests de DB usan filas fabricadas
(sin imagenes) para ser robustos; el ajuste del verificador se prueba solo si hay
dato normal real en disco.
"""
import time
from pathlib import Path

import pytest

import src.streaming.evaluador_online as ev
import src.streaming.reporte_lotes as rl
from src.streaming.verificador_anomalia import verificar, entrenar_verificador, CARPETA_NORMAL


@pytest.fixture
def db_temp(tmp_path, monkeypatch):
    ruta = str(tmp_path / "live.sqlite")
    monkeypatch.setattr(ev, "EVALUACIONES_DB", ruta)
    monkeypatch.setattr(rl, "EVALUACIONES_DB", ruta)
    ev.inicializar_db()
    return ruta


def _fila(i, lote, es_anom, es_anom_triv, correcto=1):
    return (time.time() + i, f"img{i}", "alta_calidad",
            "alta_calidad" if correcto else "defectuoso", 0.9, correcto, 0, lote,
            5.0, int(es_anom), 3.0, int(es_anom_triv))


# ----------------------- Verificador (None-safe) -----------------------
def test_verificar_none_safe_sin_modelo():
    """Sin verificador entrenado, verificar no falla y marca todo False."""
    v = verificar(None, None)
    assert v["verificado"] is False
    assert v["es_anomalia"] is False and v["es_anomalia_trivial"] is False
    assert v["score_anomalia"] is None


def test_entrenar_verificador_si_hay_normal_real():
    """Si existe dato normal real, el verificador se entrena y verifica una imagen."""
    if not any(Path(CARPETA_NORMAL).glob("*")):
        pytest.skip("Sin imagenes normales reales en disco")
    from PIL import Image
    ver = entrenar_verificador()
    assert ver is not None and "detector" in ver and "trivial" in ver
    ruta = next(Path(CARPETA_NORMAL).glob("*.jpg"))
    v = verificar(ver, Image.open(ruta).convert("RGB"))
    assert v["verificado"] is True
    assert set(v) >= {"score_anomalia", "es_anomalia", "score_trivial", "es_anomalia_trivial"}


# ----------------------- DB: métricas vivas -----------------------
def test_metricas_vivo_incluye_tasas_de_anomalia(db_temp):
    # 10 filas: la mitad marcadas anomalas por el trivial, ninguna por el detector.
    for i in range(10):
        ev.registrar_evaluacion(_fila(i, "L1", es_anom=0, es_anom_triv=(i % 2)))
    m = ev.calcular_metricas_vivo()
    assert m["estado"] == "ok"
    assert "tasa_anomalia_detector" in m and "tasa_anomalia_trivial" in m
    assert m["tasa_anomalia_detector"] == 0.0
    assert m["tasa_anomalia_trivial"] == 50.0  # el trivial SIEMPRE se reporta al lado


def test_migracion_db_antigua_agrega_columnas(tmp_path, monkeypatch):
    """Una DB vieja sin columnas de anomalia se migra sin perder datos."""
    import sqlite3
    ruta = str(tmp_path / "vieja.sqlite")
    con = sqlite3.connect(ruta)
    con.execute("CREATE TABLE live_evaluations (ts REAL, id_imagen TEXT, y_true TEXT, y_pred TEXT, "
                "confianza REAL, is_correct INTEGER, degradada INTEGER, lote_ingesta TEXT)")
    con.execute("INSERT INTO live_evaluations VALUES (1.0,'a','alta_calidad','alta_calidad',0.9,1,0,'L1')")
    con.commit(); con.close()
    monkeypatch.setattr(ev, "EVALUACIONES_DB", ruta)
    ev.inicializar_db()  # debe añadir las 4 columnas sin romper
    cols = {f[1] for f in sqlite3.connect(ruta).execute("PRAGMA table_info(live_evaluations)")}
    assert {"score_anomalia", "es_anomalia", "score_trivial", "es_anomalia_trivial"} <= cols


# ----------------------- Batch: reporte por lote -----------------------
def test_reporte_lotes_agrega_y_muestra_trivial(db_temp):
    for i in range(6):
        ev.registrar_evaluacion(_fila(i, "L1", es_anom=0, es_anom_triv=1))   # lote temprano: todo anomalo trivial
    for i in range(6, 12):
        ev.registrar_evaluacion(_fila(i, "L2", es_anom=0, es_anom_triv=0))   # lote tardio: nada anomalo
    r = rl.calcular_reporte_lotes()
    assert r["estado"] == "ok" and r["n_lotes"] == 2 and r["total_imagenes"] == 12
    l1 = next(l for l in r["lotes"] if l["lote_ingesta"] == "L1")
    assert l1["tasa_anomalia_trivial_pct"] == 100.0
    assert "tasa_anomalia_detector_pct" in l1  # siempre ambos
    assert "nota_honestidad" in r


def test_reporte_lotes_sin_datos(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "EVALUACIONES_DB", str(tmp_path / "noexiste.sqlite"))
    assert rl.calcular_reporte_lotes()["estado"] == "sin_datos"
