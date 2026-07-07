"""
Tests del nucleo (puro Python) de la etapa ETL raw -> parquet con augmentation.

Garantizan las propiedades que hacen que el Parquet curado sea seguro aguas abajo:
provenance correcta, grupo_fisico como columna (anti-fuga sobrevive al Parquet),
bytes de imagen validos y conteos correctos de expansion.
"""
import io

import numpy as np
from PIL import Image

from src.augmentation.paleta_ironbow import mapa_a_ironbow
from src.pipeline.etl_augmentation import (
    COLUMNAS,
    expandir_registro,
    generar_sinteticos_fisicos,
    expandir_particion,
)


def _imagen_bytes():
    itr = np.linspace(0, 1, 64, dtype=np.float32)[None, :].repeat(64, 0)
    buf = io.BytesIO()
    Image.fromarray(mapa_a_ironbow(itr)).save(buf, format="JPEG")
    return buf.getvalue()


def test_expandir_registro_emite_original_mas_k():
    filas = expandir_registro(_imagen_bytes(), "u://img1", "L1", "Mango_7", "defectuoso", k_aug=3, seed=0)
    assert len(filas) == 4
    provs = [f["provenance"] for f in filas]
    assert provs[0] == "real" and provs[1:] == ["phys_aug"] * 3


def test_expandir_registro_conserva_grupo_en_columna():
    filas = expandir_registro(_imagen_bytes(), "u://img1", "L1", "Mango_7", "alta_calidad", k_aug=2)
    # El grupo_fisico viaja como COLUMNA identica en todas las filas -> anti-fuga
    # sobrevive al empaquetado Parquet (no depende del nombre de archivo).
    assert {f["grupo_fisico"] for f in filas} == {"Mango_7"}
    assert {f["clase"] for f in filas} == {"alta_calidad"}


def test_filas_tienen_todas_las_columnas_y_bytes_validos():
    filas = expandir_registro(_imagen_bytes(), "u://x", "L1", "Mango_1", "defectuoso", k_aug=1)
    for f in filas:
        assert set(f.keys()) == set(COLUMNAS)
        img = Image.open(io.BytesIO(f["imagen_bytes"]))   # decodifica sin error
        assert img.size[0] > 0


def test_sinteticos_llevan_provenance_y_grupo_sintetico():
    filas = generar_sinteticos_fisicos("L1", por_clase=5, seed=1)
    assert all(f["provenance"] == "synthetic_physics" for f in filas)
    # grupo sintetico -> nunca colisiona con un mango real (empieza por synthphys_)
    assert all(f["grupo_fisico"].startswith("synthphys_") for f in filas)
    # auto-etiquetado cubre las 3 clases de forma balanceada
    clases = {f["clase"] for f in filas}
    assert clases == {"defectuoso", "calidad_promedio", "alta_calidad"}


def test_expandir_particion_cuenta_correcta():
    regs = [{"imagen_bytes": _imagen_bytes(), "uri": f"u://{i}", "lote_ingesta": "L1",
             "grupo_fisico": f"Mango_{i}", "clase": "defectuoso"} for i in range(3)]
    filas = expandir_particion(regs, k_aug=2, sinteticos_por_base=1, seed_base=0)
    # 3 bases * (1 original + 2 phys) + 3 * (3 clases sinteticas) = 9 + 9 = 18
    assert len(filas) == 18
    from collections import Counter
    provs = Counter(f["provenance"] for f in filas)
    assert provs["real"] == 3 and provs["phys_aug"] == 6 and provs["synthetic_physics"] == 9
