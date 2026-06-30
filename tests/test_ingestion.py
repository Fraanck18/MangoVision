"""Verifica que el buffer local nunca dependa de la conexion, y que el uploader solo mueva datos con señal."""
import shutil

import pytest

from src.ingestion.edge_buffer import capturar_lectura, listar_pendientes
from src.ingestion.uploader_batch import drenar_buffer_a_staging, subir_staging_a_hdfs

IMAGEN_FALSA = b"contenido-de-prueba"


@pytest.fixture
def rutas_temporales(tmp_path):
    return {
        "buffer": tmp_path / "buffer_local",
        "staging": tmp_path / "staging",
    }


def test_capturar_lectura_funciona_sin_verificar_conexion(rutas_temporales):
    id_lectura = capturar_lectura(IMAGEN_FALSA, {"sensor_id": "camion-01"}, carpeta_buffer=rutas_temporales["buffer"])
    pendientes = listar_pendientes(rutas_temporales["buffer"])

    assert len(pendientes) == 1
    assert pendientes[0]["metadata"]["id_lectura"] == id_lectura
    assert pendientes[0]["metadata"]["sensor_id"] == "camion-01"


def test_drenar_sin_conexion_no_mueve_nada(rutas_temporales):
    capturar_lectura(IMAGEN_FALSA, {"sensor_id": "camion-01"}, carpeta_buffer=rutas_temporales["buffer"])

    movidas = drenar_buffer_a_staging(
        carpeta_buffer=rutas_temporales["buffer"],
        carpeta_staging=rutas_temporales["staging"],
        verificador_conexion=lambda: False,
    )

    assert movidas == []
    assert len(listar_pendientes(rutas_temporales["buffer"])) == 1


def test_drenar_con_conexion_mueve_todo_a_staging(rutas_temporales):
    capturar_lectura(IMAGEN_FALSA, {"sensor_id": "camion-01"}, carpeta_buffer=rutas_temporales["buffer"])
    capturar_lectura(IMAGEN_FALSA, {"sensor_id": "camion-01"}, carpeta_buffer=rutas_temporales["buffer"])

    movidas = drenar_buffer_a_staging(
        carpeta_buffer=rutas_temporales["buffer"],
        carpeta_staging=rutas_temporales["staging"],
        verificador_conexion=lambda: True,
    )

    assert len(movidas) == 2
    assert listar_pendientes(rutas_temporales["buffer"]) == []
    assert len(list(rutas_temporales["staging"].glob("*.jpg"))) == 2


def test_subir_a_hdfs_particiona_por_lote_ingesta(rutas_temporales, monkeypatch):
    """
    El upload real a HDFS (`hdfs_client.subir_archivo`) requiere un NameNode
    vivo (docker-compose.hdfs.yml); se mockea aqui para probar el
    particionado de rutas sin depender de un cluster levantado.
    """
    rutas_subidas = []
    monkeypatch.setattr(
        "src.ingestion.uploader_batch.subir_archivo",
        lambda ruta_local, ruta_hdfs: rutas_subidas.append(ruta_hdfs),
    )

    capturar_lectura(IMAGEN_FALSA, {"sensor_id": "camion-01"}, carpeta_buffer=rutas_temporales["buffer"])
    drenar_buffer_a_staging(
        carpeta_buffer=rutas_temporales["buffer"],
        carpeta_staging=rutas_temporales["staging"],
        verificador_conexion=lambda: True,
    )

    resultado = subir_staging_a_hdfs(
        lote_ingesta="2026-06-26",
        carpeta_staging=rutas_temporales["staging"],
        carpeta_hdfs_raiz="/data-lake/mangos/raw",
    )

    assert resultado["lote_ingesta"] == "2026-06-26"
    assert len(resultado["subidas"]) == 1
    assert resultado["carpeta_destino"] == "/data-lake/mangos/raw/lote_ingesta=2026-06-26"
    assert any(r.endswith(".jpg") for r in rutas_subidas)
    assert any(r.endswith(".json") for r in rutas_subidas)
