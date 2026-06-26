"""
Baseline clasico de referencia: descriptores HOG (Histogram of Oriented
Gradients) sobre los mapas termicos, clasificados con un LinearSVC.

Se mantiene como punto de comparacion estricto frente a ResNet18: cuantifica
cuanto aporta la jerarquia de caracteristicas aprendidas por la red profunda
frente a un descriptor de gradientes disenado manualmente.
"""
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import IMAGE_SIZE, RANDOM_SEED


def extraer_descriptor_hog(imagen_rgb, orientaciones=9, pixeles_por_celda=(8, 8), celdas_por_bloque=(2, 2)):
    """Convierte una imagen termica RGB en un vector de gradientes orientados HOG."""
    imagen_gris = rgb2gray(np.asarray(imagen_rgb))
    return hog(
        imagen_gris,
        orientations=orientaciones,
        pixels_per_cell=pixeles_por_celda,
        cells_per_block=celdas_por_bloque,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def construir_matriz_features(lista_imagenes_rgb):
    """Aplica extraer_descriptor_hog a un lote de imagenes y apila los vectores resultantes."""
    return np.stack([extraer_descriptor_hog(img) for img in lista_imagenes_rgb])


def construir_pipeline_baseline(C=1.0, max_iter=5000):
    """Pipeline HOG -> StandardScaler -> LinearSVC, equivalente clasico al ResNet18."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(C=C, max_iter=max_iter, random_state=RANDOM_SEED)),
    ])
