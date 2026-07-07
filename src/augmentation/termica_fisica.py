"""
Tecnica #1 — Data augmentation en el DOMINIO TERMICO (physics-informed).

Es la augmentation mas potente para MangoVision porque el proyecto YA modela la
fisica (PEDTE, ITR, inercia termica). En vez de perturbar pixeles RGB al azar
—prohibido, porque el matiz Ironbow ES la etiqueta— se hace:

    Ironbow RGB  --decode-->  mapa ITR [0,1]  --perturbar (fisica)-->  ITR'
                 --re-encode (Ironbow)-->  Ironbow RGB'

Las perturbaciones son las que un sensor FLIR real introduce SIN cambiar la clase
del fruto, de modo que la etiqueta se preserva:

  * Ganancia + offset globales de ITR  -> deriva de temperatura ambiente y de la
    calibracion/emisividad del sensor entre capturas.
  * Gradiente radial centro->periferia  -> coherente con el PEDTE
    (`src/domain_physics/thermal_inertia.py`): la inercia termica deja una huella
    espacial; añadir/quitar un poco de gradiente simula frutos y condiciones de
    enfriamiento ligeramente distintos DENTRO de la misma clase.
  * Ruido NETD (gaussiano fino) + viñeteado  -> ruido de detector y caida de
    respuesta hacia los bordes del FPA.
  * Corrimiento gamma monotono  -> respuesta no lineal del mapeo del sensor.

Todo ello es MONOTONO en ITR (no reordena frio<->caliente), por lo que respeta la
señal termica relativa. Se re-codifica con la MISMA paleta que el generador
sintetico (`src/augmentation/paleta_ironbow.py`).

Uso como transform de torchvision (PIL -> PIL), encaja en un Compose:

    from src.augmentation.termica_fisica import AugmentacionTermicaFisica
    tf = transforms.Compose([..., AugmentacionTermicaFisica(p=0.6), ToTensor(), ...])
"""
import numpy as np
from PIL import Image

from src.augmentation.paleta_ironbow import ironbow_a_itr, mapa_a_ironbow


def _gradiente_radial(alto, ancho):
    """Radio normalizado [0,1] desde el centro (0 en el centro, ~1 en la esquina)."""
    yy, xx = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    cy, cx = (alto - 1) / 2.0, (ancho - 1) / 2.0
    dist = np.sqrt(((yy - cy) / (cy or 1.0)) ** 2 + ((xx - cx) / (cx or 1.0)) ** 2)
    return np.clip(dist / (dist.max() or 1.0), 0.0, 1.0)


def perturbar_itr_fisico(
    itr,
    rng,
    ganancia=(0.85, 1.15),
    offset=(-0.06, 0.06),
    gradiente=(-0.05, 0.05),
    gamma=(0.85, 1.20),
    netd=0.012,
    vigneteo=(0.0, 0.05),
):
    """
    Aplica al mapa ITR [0,1] una combinacion de perturbaciones FISICAMENTE validas
    (label-preserving) y devuelve otro mapa ITR [0,1].

    Cada rango es (min, max) de una distribucion uniforme; `netd` es la sigma del
    ruido gaussiano fino del detector. Todas las operaciones son monotonas o
    aditivas de baja amplitud: no reordenan el ranking termico -> la clase se
    conserva.
    """
    itr = np.asarray(itr, dtype=np.float32)
    alto, ancho = itr.shape

    # 1. Gamma monotono (respuesta no lineal del sensor). Preserva el orden.
    g = float(rng.uniform(*gamma))
    itr = np.power(np.clip(itr, 0.0, 1.0), g)

    # 2. Ganancia + offset globales (deriva ambiente / emisividad).
    itr = itr * float(rng.uniform(*ganancia)) + float(rng.uniform(*offset))

    # 3. Gradiente radial centro->periferia (huella PEDTE). Signo aleatorio:
    #    a veces el centro queda algo mas caliente, a veces la periferia.
    radio = _gradiente_radial(alto, ancho)
    itr = itr + float(rng.uniform(*gradiente)) * (radio - 0.5) * 2.0

    # 4. Viñeteado (caida de respuesta hacia los bordes del FPA).
    itr = itr - float(rng.uniform(*vigneteo)) * radio

    # 5. Ruido NETD del detector (gaussiano fino).
    if netd > 0:
        itr = itr + rng.normal(0.0, netd, size=itr.shape).astype(np.float32)

    return np.clip(itr, 0.0, 1.0)


class AugmentacionTermicaFisica:
    """
    Transform PIL->PIL: decodifica Ironbow a ITR, aplica perturbaciones fisicas y
    re-codifica a Ironbow. Con probabilidad `1 - p` devuelve la imagen intacta.

    `seed` es opcional y solo para reproducibilidad en tests; en entrenamiento se
    deja None para que cada llamada varie.
    """

    def __init__(self, p=0.6, seed=None, **kwargs_perturbacion):
        self.p = float(p)
        self.kwargs = kwargs_perturbacion
        self._rng = np.random.default_rng(seed)

    def __call__(self, imagen):
        if self._rng.random() >= self.p:
            return imagen
        pil = imagen.convert("RGB") if imagen.mode != "RGB" else imagen
        itr = ironbow_a_itr(np.asarray(pil))
        itr = perturbar_itr_fisico(itr, self._rng, **self.kwargs)
        return Image.fromarray(mapa_a_ironbow(itr), mode="RGB")

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
