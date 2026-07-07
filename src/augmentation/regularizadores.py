"""
Regularizadores estandar, filtrados para que sean TERMICAMENTE SEGUROS.

Ninguno toca el matiz Ironbow de forma arbitraria:

  * DeformacionElastica  (#6) — warping elastico: el mango es un cuerpo organico
    deformable, asi que estirarlo localmente genera variacion de forma plausible
    con muy pocas imagenes. Es puramente geometrico -> no altera la etiqueta.
  * AutoAugmentGeometrico (#7) — politica auto tipo TrivialAugment pero
    RESTRINGIDA a operaciones geometricas (traslacion, shear, rotacion, escala,
    erasing). Se excluyen deliberadamente TODAS las ops de color de RandAugment/
    AutoAugment (posterize, solarize, color, contrast, equalize) porque en
    pseudo-color desplazarian la temperatura aparente.
  * RandomErasing (#5) — se re-exporta la de torchvision; simula pixeles muertos
    y oclusiones del sensor. Opera sobre el tensor ya normalizado.
  * MixUp (#4) — mezcla a NIVEL DE LOTE dos firmas termicas y sus etiquetas de
    forma lineal. No inventa matices nuevos fuera de la variedad real; suaviza el
    colapso de la clase intermedia. Se prefiere a CutMix, que crearia bordes
    termicos duros no fisicos.

`obtener_transformaciones_entrenamiento_avanzado()` compone el pipeline completo,
termicamente seguro, para reemplazar al basico (solo flip+rot) de
`src/models/resnet50_custom.py`.
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.augmentation.termica_fisica import AugmentacionTermicaFisica


# ---------------------------------------------------------------------------
# #6 Deformacion elastica (geometrica, label-preserving)
# ---------------------------------------------------------------------------
class DeformacionElastica:
    """
    Warping elastico (Simard et al., 2003): desplaza cada pixel segun un campo de
    desplazamiento aleatorio suavizado por un Gaussiano. Simula la variabilidad de
    forma de un fruto organico. Transform PIL->PIL.

    alpha  -> intensidad del desplazamiento (px).
    sigma  -> suavidad del campo (px); mayor = deformaciones mas globales/suaves.
    p      -> probabilidad de aplicarla.
    """

    def __init__(self, alpha=34.0, sigma=4.0, p=0.5, seed=None):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def __call__(self, imagen):
        if self._rng.random() >= self.p:
            return imagen
        from scipy.ndimage import gaussian_filter, map_coordinates

        pil = imagen.convert("RGB") if imagen.mode != "RGB" else imagen
        arreglo = np.asarray(pil, dtype=np.float32)
        alto, ancho = arreglo.shape[:2]

        dx = gaussian_filter((self._rng.random((alto, ancho)) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((self._rng.random((alto, ancho)) * 2 - 1), self.sigma) * self.alpha
        yy, xx = np.meshgrid(np.arange(alto), np.arange(ancho), indexing="ij")
        coords = [np.reshape(yy + dy, (-1,)), np.reshape(xx + dx, (-1,))]

        salida = np.empty_like(arreglo)
        for canal in range(arreglo.shape[2]):
            salida[..., canal] = map_coordinates(
                arreglo[..., canal], coords, order=1, mode="reflect"
            ).reshape(alto, ancho)
        return Image.fromarray(salida.astype(np.uint8), mode="RGB")

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma}, p={self.p})"


# ---------------------------------------------------------------------------
# #7 Auto-augment RESTRINGIDO a geometria (sin ninguna op de color)
# ---------------------------------------------------------------------------
class AutoAugmentGeometrico:
    """
    Politica estilo TrivialAugment: en cada llamada elige UNA operacion geometrica
    al azar con magnitud aleatoria. El catalogo excluye por diseño toda op de color
    (que romperia el pseudo-color termico). Transform PIL->PIL.
    """

    _OPERACIONES = ("identidad", "shear_x", "shear_y", "translate_x",
                    "translate_y", "rotate", "scale")

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def __call__(self, imagen):
        pil = imagen.convert("RGB") if imagen.mode != "RGB" else imagen
        op = self._OPERACIONES[self._rng.integers(0, len(self._OPERACIONES))]
        m = float(self._rng.random())  # magnitud [0,1]

        if op == "identidad":
            return pil
        if op == "shear_x":
            return pil.transform(pil.size, Image.AFFINE,
                                 (1, (m - 0.5) * 0.6, 0, 0, 1, 0), resample=Image.BILINEAR)
        if op == "shear_y":
            return pil.transform(pil.size, Image.AFFINE,
                                 (1, 0, 0, (m - 0.5) * 0.6, 1, 0), resample=Image.BILINEAR)
        if op == "translate_x":
            return pil.transform(pil.size, Image.AFFINE,
                                 (1, 0, (m - 0.5) * 0.3 * pil.size[0], 0, 1, 0), resample=Image.BILINEAR)
        if op == "translate_y":
            return pil.transform(pil.size, Image.AFFINE,
                                 (1, 0, 0, 0, 1, (m - 0.5) * 0.3 * pil.size[1]), resample=Image.BILINEAR)
        if op == "rotate":
            return pil.rotate((m - 0.5) * 50.0, resample=Image.BILINEAR)
        if op == "scale":
            factor = 0.8 + 0.4 * m
            nuevo = (max(1, int(pil.size[0] * factor)), max(1, int(pil.size[1] * factor)))
            return pil.resize(nuevo, Image.BILINEAR).resize(pil.size, Image.BILINEAR)
        return pil

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Pipeline compuesto (reemplaza al basico flip+rot)
# ---------------------------------------------------------------------------
def obtener_transformaciones_entrenamiento_avanzado(
    usar_fisica=True, usar_elastica=True, usar_geometrico=True, usar_erasing=True,
    p_fisica=0.6, p_elastica=0.4,
):
    """
    Pipeline TERMICAMENTE SEGURO para TRAIN. Orden:
      1. Resize
      2. Flip horizontal + rotacion (base historica, segura)
      3. AutoAugmentGeometrico (#7)      [opcional]
      4. DeformacionElastica (#6)        [opcional]
      5. AugmentacionTermicaFisica (#1)  [opcional]  -> unica que toca el ITR, de forma fisica
      6. ToTensor + Normalize
      7. RandomErasing (#5)              [opcional]  -> sobre el tensor normalizado

    Los flags permiten activar/desactivar cada tecnica (util para ablation). MixUp
    NO va aqui: vive a nivel de lote en el loop de entrenamiento (ver mixup_lote).
    """
    pasos = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ]
    if usar_geometrico:
        pasos.append(AutoAugmentGeometrico())
    if usar_elastica:
        pasos.append(DeformacionElastica(p=p_elastica))
    if usar_fisica:
        pasos.append(AugmentacionTermicaFisica(p=p_fisica))
    pasos += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if usar_erasing:
        # value="random" rellena con ruido; scale acotado para no borrar todo el fruto.
        pasos.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), value="random"))
    return transforms.Compose(pasos)


# ---------------------------------------------------------------------------
# #4 MixUp (a nivel de lote, en el loop de entrenamiento)
# ---------------------------------------------------------------------------
def mixup_lote(imagenes, etiquetas, alpha=0.2, generador=None):
    """
    Mezcla lineal de un lote consigo mismo permutado:
        x' = lam * x + (1 - lam) * x[perm]
    Devuelve (x', y_a, y_b, lam), donde lam ~ Beta(alpha, alpha).

    La perdida se combina despues con `perdida_mixup`. Regularizador fuerte para
    datasets diminutos; suaviza fronteras entre clases (ataca el colapso de la
    clase intermedia).
    """
    if alpha <= 0:
        return imagenes, etiquetas, etiquetas, 1.0
    lam = float(np.random.default_rng().beta(alpha, alpha)) if generador is None \
        else float(np.random.default_rng(generador).beta(alpha, alpha))
    indices = torch.randperm(imagenes.size(0), device=imagenes.device)
    mezcladas = lam * imagenes + (1.0 - lam) * imagenes[indices]
    return mezcladas, etiquetas, etiquetas[indices], lam


def perdida_mixup(funcion_perdida, predicciones, y_a, y_b, lam):
    """Combina la perdida sobre las dos etiquetas mezcladas por MixUp."""
    return lam * funcion_perdida(predicciones, y_a) + (1.0 - lam) * funcion_perdida(predicciones, y_b)
