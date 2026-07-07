"""
Filtro de DESPLAZAMIENTO DE TEMPERATURA (data augmentation por régimen térmico).

Idea (del usuario): el dataset no son "solo 11 mangos" — cada imagen es una MATRIZ
térmica que se puede llevar a OTRAS temperaturas. Una misma imagen base, enfriada o
calentada, es una muestra térmica NUEVA y válida: el proyecto clasifica por RÉGIMEN
TÉRMICO (calidad_promedio=frío/verde, alta_calidad=medio/maduro, defectuoso=
caliente/sobre-maduro, ver MATURITY_INFO). Así que desplazar la temperatura de una
imagen genera diversidad térmica real para entrenar y evaluar.

Cómo (dominio ITR, honesto con F3 — no °C absolutos):
  1. Ironbow RGB  --decode-->  mapa ITR [0,1].
  2. Se desplaza/escala el nivel térmico:  ITR' = clip(ITR * ganancia + delta).
     - delta muy negativo -> versión FRÍA (verde/inmaduro).
     - delta alto (+ focos de calor opcionales) -> versión CALIENTE (sobre-maduro).
  3. Se REETIQUETA con la REGLA FÍSICA determinista (etiquetado_fisico), NO con el
     modelo -> sin concordancia circular. La etiqueta refleja el régimen térmico
     resultante.
  4. Re-encode a Ironbow con la paleta física compartida.

A diferencia de AugmentacionTermicaFisica (perturbación monótona LABEL-PRESERVING),
aquí el desplazamiento es DELIBERADAMENTE grande y puede CAMBIAR la clase -> genera
muestras de OTROS regímenes térmicos, expandiendo la diversidad de temperatura.
"""
import numpy as np
from PIL import Image

from src.augmentation.paleta_ironbow import ironbow_a_itr, mapa_a_ironbow
from src.domain_physics.etiquetado_fisico import etiquetar_por_firma_termica, ITR_CALIENTE


def desplazar_temperatura_itr(itr, delta=0.0, ganancia=1.0):
    """
    Desplaza y escala el nivel térmico de un mapa ITR:  ITR' = clip(ITR*ganancia + delta).
    delta<0 enfría, delta>0 calienta; ganancia comprime/expande la amplitud térmica.
    """
    itr = np.asarray(itr, dtype=np.float32)
    return np.clip(itr * float(ganancia) + float(delta), 0.0, 1.0)


def _inyectar_focos_calor(itr, rng, n_focos=(3, 6), intensidad=(0.25, 0.45)):
    """Añade focos gaussianos de calor (deshidratación/sobre-exposición) al mapa ITR."""
    n = itr.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    for _ in range(int(rng.integers(*n_focos))):
        cx, cy = rng.uniform(0.3, 0.7) * n, rng.uniform(0.3, 0.7) * n
        sigma = rng.uniform(0.06, 0.14) * n
        itr = itr + float(rng.uniform(*intensidad)) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return np.clip(itr, 0.0, 1.0)


def _regime_frio(itr_base, mean, rng):
    """FRÍO/uniforme (verde/inmaduro): homogeneiza FUERTE (aplana el gradiente MUY por
    debajo del umbral) y baja el nivel a una banda claramente fría. Margen amplio
    frente al régimen medio para que `calidad_promedio` sea separable."""
    factor = float(rng.uniform(0.05, 0.11))   # casi plano: gradiente << umbral
    nivel = float(rng.uniform(0.10, 0.17))    # claramente frío
    return np.clip(nivel + (itr_base - mean) * factor, 0.0, 1.0)


def _regime_medio(itr_base, mean, rng):
    """MEDIO (maduro/óptimo): nivel INTERMEDIO con un gradiente de disipación claro,
    SIN crear píxeles calientes (que lo empujarían a defecto). Se separa del frío por
    nivel (~0.44) y por conservar gradiente; se separa del caliente manteniendo el
    máximo por debajo del umbral caliente."""
    factor = float(rng.uniform(0.26, 0.38))
    nivel = float(rng.uniform(0.40, 0.46))
    return np.clip(nivel + (itr_base - mean) * factor, 0.0, 1.0)


def _regime_caliente(itr_base, rng):
    """CALIENTE (sobre-maduro/defecto): sube ganancia+nivel e inyecta focos de calor."""
    itr = np.clip(itr_base * float(rng.uniform(1.05, 1.25)) + float(rng.uniform(0.12, 0.28)), 0.0, 1.0)
    return _inyectar_focos_calor(itr, rng)


def variantes_temperatura(imagen_pil, rng, n=6, **_):
    """
    Genera `n` variantes de la imagen a DISTINTAS temperaturas, barriendo los tres
    regímenes térmicos (frío / medio / caliente) de forma REGIME-TARGETED para que
    la diversidad de temperatura cubra las tres clases incluso desde bases reales
    muy estructuradas. Cada variante se REETIQUETA por la regla física determinista.
    Devuelve lista de (imagen_pil_RGB, etiqueta).
    """
    pil = imagen_pil.convert("RGB") if imagen_pil.mode != "RGB" else imagen_pil
    itr_base = ironbow_a_itr(np.asarray(pil))
    mean = float(itr_base.mean())
    variantes = []
    for k in range(n):
        frac = k / max(1, n - 1)
        if frac < 1 / 3:
            itr = _regime_frio(itr_base, mean, rng)
        elif frac < 2 / 3:
            itr = _regime_medio(itr_base, mean, rng)
        else:
            itr = _regime_caliente(itr_base, rng)
        etiqueta = etiquetar_por_firma_termica(itr)
        variantes.append((Image.fromarray(mapa_a_ironbow(itr), mode="RGB"), etiqueta))
    return variantes


class AugmentacionTemperatura:
    """
    Transform PIL->PIL de UNA variante térmica aleatoria (para uso online). NO es
    label-preserving: cambia el régimen térmico. Devuelve solo la imagen; para
    obtener también la etiqueta reetiquetada usa `variantes_temperatura`.
    """

    def __init__(self, rango_delta=(-0.30, 0.35), rango_ganancia=(0.85, 1.20), seed=None):
        self.rango_delta = rango_delta
        self.rango_ganancia = rango_ganancia
        self._rng = np.random.default_rng(seed)

    def __call__(self, imagen):
        pil = imagen.convert("RGB") if imagen.mode != "RGB" else imagen
        itr = ironbow_a_itr(np.asarray(pil))
        itr = desplazar_temperatura_itr(
            itr, delta=float(self._rng.uniform(*self.rango_delta)),
            ganancia=float(self._rng.uniform(*self.rango_ganancia)))
        return Image.fromarray(mapa_a_ironbow(itr), mode="RGB")
