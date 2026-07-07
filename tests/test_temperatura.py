"""
Tests del filtro de desplazamiento de temperatura (augmentation por régimen térmico).

Propiedades: el desplazamiento ITR respeta [0,1]; el barrido regime-targeted cubre
las tres clases desde bases reales estructuradas; y las etiquetas las pone la REGLA
FÍSICA determinista (no un modelo).
"""
import numpy as np
from PIL import Image

from src.config import CLASS_NAMES
from src.augmentation.paleta_ironbow import mapa_a_ironbow
from src.augmentation.temperatura import (
    desplazar_temperatura_itr,
    variantes_temperatura,
    AugmentacionTemperatura,
)


def _imagen_fruto_real(n=96, rng=None):
    """Ironbow con un fruto ovoide caliente sobre fondo frío (base estructurada)."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.sqrt(((xx - n / 2) / (0.4 * n)) ** 2 + ((yy - n / 2) / (0.34 * n)) ** 2)
    itr = np.clip(0.55 * np.clip(1 - r, 0, 1) + 0.12 + 0.03 * rng.standard_normal((n, n)), 0, 1)
    return Image.fromarray(mapa_a_ironbow(itr), mode="RGB")


def test_desplazamiento_respeta_rango_unitario():
    itr = np.linspace(0, 1, 100, dtype=np.float32)
    out = desplazar_temperatura_itr(itr, delta=0.5, ganancia=1.3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_desplazamiento_enfria_y_calienta():
    itr = np.full((16, 16), 0.5, dtype=np.float32)
    frio = desplazar_temperatura_itr(itr, delta=-0.3)
    calor = desplazar_temperatura_itr(itr, delta=+0.3)
    assert frio.mean() < itr.mean() < calor.mean()


def test_variantes_cubren_las_tres_clases_desde_base_real():
    """El barrido regime-targeted produce las tres clases desde una base estructurada."""
    rng = np.random.default_rng(1)
    etiquetas = set()
    for s in range(30):
        img = _imagen_fruto_real(rng=np.random.default_rng(s))
        for _, lab in variantes_temperatura(img, rng, n=6):
            etiquetas.add(lab)
    assert etiquetas == set(CLASS_NAMES)


def test_variantes_devuelven_etiqueta_valida_y_pil():
    rng = np.random.default_rng(2)
    for img_var, etiqueta in variantes_temperatura(_imagen_fruto_real(), rng, n=6):
        assert isinstance(img_var, Image.Image) and img_var.mode == "RGB"
        assert etiqueta in CLASS_NAMES


def test_transform_online_devuelve_pil_rgb():
    out = AugmentacionTemperatura(seed=3)(_imagen_fruto_real())
    assert isinstance(out, Image.Image) and out.mode == "RGB"
