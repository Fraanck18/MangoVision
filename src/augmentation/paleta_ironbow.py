"""
Paleta Ironbow: unica fuente de verdad para CODIFICAR (ITR -> RGB) y
DECODIFICAR (RGB -> ITR) el pseudo-color termico.

El generador sintetico (`scripts/generate_synthetic_thermal.py`) y la
augmentation fisica (`src/augmentation/termica_fisica.py`) comparten esta paleta
para que ambos hablen exactamente el mismo "idioma de color termico".

Ironbow es una rampa MONOTONA frio->caliente: azul -> cian -> verde -> amarillo
-> naranja -> rojo -> blanco. Esa monotonia es lo que permite invertirla de
forma estable a un Indice de Intensidad Termica Relativa (ITR) en [0, 1].
"""
import numpy as np

# (valor_normalizado en [0,1], (R, G, B)) de frio a caliente.
PALETA_IRONBOW = [
    (0.00, (0, 0, 160)),      # azul = frio
    (0.15, (0, 200, 255)),    # cian
    (0.35, (0, 200, 0)),      # verde
    (0.55, (255, 230, 0)),    # amarillo
    (0.75, (255, 130, 0)),    # naranja
    (0.90, (230, 0, 0)),      # rojo
    (1.00, (255, 255, 255)),  # blanco = mas caliente
]

_VALORES = np.array([p[0] for p in PALETA_IRONBOW], dtype=np.float32)
_COLORES = np.array([p[1] for p in PALETA_IRONBOW], dtype=np.float32)


def mapa_a_ironbow(mapa01):
    """
    Interpola linealmente un mapa ITR [0,1] sobre la paleta Ironbow -> RGB uint8.
    Shape (H, W) -> (H, W, 3).
    """
    m = np.clip(np.asarray(mapa01, dtype=np.float32), 0.0, 1.0)
    rgb = np.empty(m.shape + (3,), dtype=np.float32)
    for canal in range(3):
        rgb[..., canal] = np.interp(m, _VALORES, _COLORES[:, canal])
    return rgb.astype(np.uint8)


def ironbow_a_itr(imagen_rgb, muestras=256):
    """
    DECODIFICA una imagen Ironbow RGB de vuelta a un mapa ITR [0,1].

    Estrategia robusta (no una inversa analitica fragil): se discretiza la rampa
    Ironbow en `muestras` puntos y, por cada pixel, se busca el valor ITR cuyo
    color de paleta esta MAS CERCA en RGB (vecino mas cercano en el espacio de
    color de la rampa). Es estable ante el ruido de compresion JPEG porque la
    rampa es 1-D y monotona: pequeñas desviaciones de color caen en el mismo
    tramo. Devuelve un mapa (H, W) en [0, 1].

    Nota: esto NO recupera temperatura absoluta (imposible desde pseudo-color, F3);
    recupera el ORDEN termico relativo, que es justo lo que preserva la etiqueta.
    """
    arreglo = np.asarray(imagen_rgb, dtype=np.float32)
    if arreglo.ndim == 2:  # ya es escala de grises/ITR
        mn, mx = float(arreglo.min()), float(arreglo.max())
        return (arreglo - mn) / ((mx - mn) or 1.0)

    ejes = np.linspace(0.0, 1.0, muestras, dtype=np.float32)
    rampa = mapa_a_ironbow(ejes).astype(np.float32)          # (muestras, 3)

    h, w, _ = arreglo.shape
    planos = arreglo.reshape(-1, 3)                          # (H*W, 3)
    # Distancia euclidea a cada punto de la rampa; argmin -> ITR del vecino.
    # (se hace por bloques para no reventar memoria en imagenes grandes).
    itr = np.empty(planos.shape[0], dtype=np.float32)
    bloque = 4096
    for i in range(0, planos.shape[0], bloque):
        trozo = planos[i:i + bloque][:, None, :]             # (b, 1, 3)
        d2 = ((trozo - rampa[None, :, :]) ** 2).sum(axis=2)  # (b, muestras)
        itr[i:i + bloque] = ejes[np.argmin(d2, axis=1)]
    return itr.reshape(h, w)
