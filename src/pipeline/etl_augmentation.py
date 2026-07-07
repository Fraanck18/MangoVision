"""
Nucleo (puro Python, testeable sin Spark) de la etapa ETL raw -> parquet con
DATA AUGMENTATION fisica. Lo consume `scripts/etl_raw_a_parquet_spark.py` dentro
de un `mapInPandas`, y los tests lo ejercen directamente.

Cada imagen real produce varias FILAS del Parquet:
  * 1 fila original          -> provenance="real"
  * K filas fisicamente aumentadas (AugmentacionTermicaFisica) -> provenance="phys_aug"
  * (opcional) filas sinteticas auto-etiquetadas por firma termica -> "synthetic_physics"

Columnas de cada fila (esquema del Data Lake curado):
  uri_origen, lote_ingesta, grupo_fisico, clase, provenance, imagen_bytes

`grupo_fisico` viaja como COLUMNA (no solo en el nombre de archivo): asi el anti-
fuga sobrevive al empaquetado Parquet (fallo #2 / disciplina F4). `provenance`
permite excluir por diseño del test todo lo que no sea "real".
"""
import io

import numpy as np
from PIL import Image

from src.augmentation.termica_fisica import AugmentacionTermicaFisica
from src.augmentation.paleta_ironbow import mapa_a_ironbow
from src.domain_physics.etiquetado_fisico import GENERADORES_POR_CLASE, base_canonica_fruto

# Nombres de columnas del Parquet curado (una sola fuente de verdad).
COLUMNAS = ["uri_origen", "lote_ingesta", "grupo_fisico", "clase", "provenance", "imagen_bytes"]


def _a_bytes(imagen_pil, calidad=92):
    buf = io.BytesIO()
    imagen_pil.convert("RGB").save(buf, format="JPEG", quality=calidad)
    return buf.getvalue()


def _fila(uri, lote, grupo, clase, provenance, imagen_pil):
    return {
        "uri_origen": uri,
        "lote_ingesta": lote,
        "grupo_fisico": grupo,
        "clase": clase or "",
        "provenance": provenance,
        "imagen_bytes": _a_bytes(imagen_pil),
    }


def expandir_registro(imagen_bytes, uri, lote_ingesta, grupo_fisico, clase, k_aug=3, seed=None):
    """
    Expande UNA imagen real en su fila original + `k_aug` variantes fisicas
    (label-preserving). Devuelve una lista de dicts con las columnas COLUMNAS.

    La augmentation fisica es monotona en ITR (ganancia/offset, gradiente radial,
    NETD, viñeteo, gamma): NO cambia la clase -> la etiqueta real se conserva en
    todas las variantes. El `grupo_fisico` es el mismo en todas -> anti-fuga.
    """
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    filas = [_fila(uri, lote_ingesta, grupo_fisico, clase, "real", imagen)]
    for k in range(k_aug):
        aug = AugmentacionTermicaFisica(p=1.0, seed=None if seed is None else seed + k)
        filas.append(_fila(f"{uri}#phys{k}", lote_ingesta, grupo_fisico, clase, "phys_aug", aug(imagen)))
    return filas


def generar_sinteticos_fisicos(lote_ingesta, por_clase=1, seed=None, tam=224):
    """
    Motor de simulacion parametrica: genera muestras SINTETICAS auto-etiquetadas por
    firma termica sobre un SUBSTRATO CANONICO (no una imagen real), para que la
    etiqueta dependa de la fisica aplicada y la generacion sea controlable:
    defectuoso via estres termico, calidad_promedio via enfriamiento, alta_calidad
    via firma de madurez.

    Las filas llevan provenance="synthetic_physics" y grupo_fisico sintetico
    (`synthphys_<clase>_<i>`) -> NUNCA colisionan con un mango real y quedan
    EXCLUIDAS del test honesto. La etiqueta la fija el labeler determinista sobre el
    resultado (no un modelo). Devuelve lista de dicts.
    """
    rng = np.random.default_rng(seed)
    filas = []
    contador = 0
    for clase, generador in GENERADORES_POR_CLASE.items():
        for _ in range(por_clase):
            base = base_canonica_fruto(tam, rng)
            itr_nueva, etiqueta = generador(base, rng)
            imagen = Image.fromarray(mapa_a_ironbow(itr_nueva), mode="RGB")
            grupo = f"synthphys_{etiqueta}_{contador}"
            filas.append(_fila(f"synthphys://{lote_ingesta}/{grupo}", lote_ingesta, grupo,
                               etiqueta, "synthetic_physics", imagen))
            contador += 1
    return filas


def expandir_particion(registros, k_aug=3, sinteticos_por_base=0, seed_base=0):
    """
    Aplica `expandir_registro` (y opcionalmente `generar_sinteticos_fisicos`) a un
    iterable de registros {imagen_bytes, uri, lote_ingesta, grupo_fisico, clase}.
    Es el cuerpo que corre dentro de cada particion Spark. Devuelve lista de filas.
    """
    salida = []
    for i, r in enumerate(registros):
        salida.extend(expandir_registro(
            r["imagen_bytes"], r["uri"], r["lote_ingesta"], r["grupo_fisico"],
            r.get("clase", ""), k_aug=k_aug, seed=seed_base + i * (k_aug + 1),
        ))
        if sinteticos_por_base:
            salida.extend(generar_sinteticos_fisicos(
                r["lote_ingesta"], por_clase=sinteticos_por_base, seed=seed_base + 100000 + i,
            ))
    return salida
