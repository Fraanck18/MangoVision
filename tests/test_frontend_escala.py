"""
[F8] Guardarraíl anti-desincronización front/back de la escala térmica.

Raíz del problema que motiva esto: el backend pivotó de °C a ITR (F3) y el
frontend quedó atrás (buckets °C, "precisión" = confianza). Estos tests fallan si
esa clase de divergencia reaparece:

  * El fallback de `web/js/escala.js` debe COINCIDIR con `MATURITY_INFO` (config.py).
  * El histograma no puede volver a usar buckets °C.
  * `precisionModelo = confianza` no puede reaparecer (confianza ≠ precisión).

Es lo que un evaluador anti-fraude vería en una demo en vivo.
"""
import re
from pathlib import Path

from src.config import MATURITY_INFO, CLASS_NAMES

WEB = Path(__file__).resolve().parents[1] / "web" / "js"


def _leer(nombre):
    return (WEB / nombre).read_text(encoding="utf-8")


def test_fallback_escala_coincide_con_config():
    """El objeto ESCALA de escala.js refleja fielmente MATURITY_INFO."""
    js = _leer("escala.js")
    for clase in CLASS_NAMES:
        # banda_itr: [min, max]
        m = re.search(rf"{clase}:\s*{{[^}}]*banda_itr:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", js)
        assert m, f"No se encontró banda_itr de {clase} en escala.js"
        banda_js = (int(m.group(1)), int(m.group(2)))
        banda_cfg = tuple(MATURITY_INFO[clase]["banda_itr"])
        assert banda_js == banda_cfg, f"{clase}: escala.js {banda_js} != config {banda_cfg}"

        # color
        mc = re.search(rf"{clase}:\s*{{[^}}]*color:\s*\"(#[0-9a-fA-F]{{6}})\"", js)
        assert mc, f"No se encontró color de {clase} en escala.js"
        assert mc.group(1).lower() == MATURITY_INFO[clase]["color"].lower(), (
            f"{clase}: color escala.js {mc.group(1)} != config {MATURITY_INFO[clase]['color']}"
        )


def test_histograma_no_usa_buckets_celsius():
    """chartsRenderer.js debe usar la escala ITR única, no buckets °C propios."""
    js = _leer("chartsRenderer.js")
    assert "BUCKETS_TEMP" not in js, "Reaparecieron buckets °C (BUCKETS_TEMP)"
    assert "BUCKETS_ITR" in js, "El histograma debe usar BUCKETS_ITR (escala única)"
    # No debe etiquetar los ejes con grados.
    assert "}°`" not in js and "]}°" not in js, "El histograma etiqueta en grados (°)"


def test_no_se_llama_precision_a_la_confianza():
    """[F9] No debe existir la asignación precisionModelo = confianza."""
    main_js = _leer("main.js")
    assert "precisionModelo" not in main_js, "Reapareció 'precisionModelo' (confianza≠precisión)"
    assert "confianzaUltimaImagen" in main_js, "Debe existir el campo honesto confianzaUltimaImagen"


def test_backend_expone_endpoint_de_escala():
    """El backend debe seguir exponiendo /config/escala como fuente única."""
    app_py = (Path(__file__).resolve().parents[1] / "src" / "api" / "app.py").read_text(encoding="utf-8")
    assert "/config/escala" in app_py, "Falta el endpoint /config/escala (fuente única F8)"
