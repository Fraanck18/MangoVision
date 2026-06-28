"""
Generador de dataset termico SINTETICO fisicamente coherente para MangoVision.

MOTIVO: el dataset real (v1i) resulto ser termicamente neutro -- todas las
clases muestran el fruto a ~26-30 C en una sola foto (no una curva de
enfriamiento), por lo que NO codifica madurez y ningun modelo puede aprenderla
de ahi (ver analisis: alta_calidad y calidad_promedio a la misma temperatura).
Este generador construye imagenes donde cada clase SI tiene la firma termica
que la fisica predice, para entrenar y evaluar el pipeline completo.

ADVERTENCIA (honestidad): es un dataset SIMULADO. La precision obtenida
demuestra que la CNN aprende el modelo fisico aqui definido y valida el
pipeline (HDFS, workers, epocas, dashboard); NO es una validacion sobre fruta
real. No debe presentarse como exactitud medida en mangos reales.

Modelo fisico por clase (temperatura superficial, Ley de enfriamiento de Newton):
  - Verde/Inmaduro (calidad_promedio): metabolismo bajo -> fruto FRIO (15-21 C)
    y HOMOGENEO (baja varianza, sin puntos calientes).
  - Maduro/Optimo (alta_calidad): metabolismo activo -> 22-32 C con gradiente
    radial moderado (centro algo mas calido que el borde).
  - Sobre-maduro (defectuoso): descomposicion tisular -> 33-42 C con PUNTOS
    CALIENTES locales (anomalias termicas) y alta varianza.

La temperatura se mapea a la paleta Ironbow (la misma que decodifica
src/domain_physics/thermal_inertia.py) sobre un rango fijo 10-45 C, de modo que
el color refleje la temperatura real: verde=cian/verde, maduro=amarillo/naranja,
defectuoso=rojo/blanco con puntos calientes.

Uso:
    python scripts/generate_synthetic_thermal.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED

PROCESADOS = RAIZ / "dataset_mangos" / "processed"

T_MIN_PALETA = 10.0   # extremo frio de la paleta (azul)
T_MAX_PALETA = 45.0   # extremo caliente de la paleta (blanco)

N_TRAIN_POR_CLASE = 150
N_TEST_POR_CLASE = 40

# Paleta Ironbow (valor_normalizado, (R, G, B)) frio -> caliente. Misma
# referencia que src/domain_physics/thermal_inertia.py.
_PALETA = [
    (0.00, (0, 0, 160)),     # azul = frio
    (0.15, (0, 200, 255)),   # cian
    (0.35, (0, 200, 0)),     # verde
    (0.55, (255, 230, 0)),   # amarillo
    (0.75, (255, 130, 0)),   # naranja
    (0.90, (230, 0, 0)),     # rojo
    (1.00, (255, 255, 255)),  # blanco = mas caliente
]


def mapa_a_ironbow(mapa01):
    """Interpola linealmente un mapa [0,1] sobre la paleta Ironbow -> RGB uint8."""
    valores = np.array([p[0] for p in _PALETA])
    colores = np.array([p[1] for p in _PALETA], dtype=float)
    m = np.clip(mapa01, 0.0, 1.0)
    rgb = np.empty(m.shape + (3,), dtype=float)
    for canal in range(3):
        rgb[..., canal] = np.interp(m, valores, colores[:, canal])
    return rgb.astype(np.uint8)


def generar_imagen(clase, rng):
    """Genera una imagen termica sintetica (PIL.Image RGB) de la clase indicada."""
    n = IMAGE_SIZE

    # --- Fondo ambiente frio con gradiente suave (como el fondo azul real) ---
    t_amb = rng.uniform(11.0, 15.0)
    grad = np.linspace(rng.uniform(-2.0, 0.0), rng.uniform(0.0, 3.0), n)
    temp = t_amb + grad[None, :] + rng.normal(0.0, 0.6, (n, n))

    # --- Fruto: elipse aleatoria (posicion, tamano y rotacion variables) ---
    cx, cy = rng.uniform(0.40, 0.60) * n, rng.uniform(0.40, 0.60) * n
    ax, ay = rng.uniform(0.28, 0.40) * n, rng.uniform(0.24, 0.36) * n
    angulo = rng.uniform(-0.4, 0.4)

    yy, xx = np.mgrid[0:n, 0:n]
    xr = (xx - cx) * np.cos(angulo) + (yy - cy) * np.sin(angulo)
    yr = -(xx - cx) * np.sin(angulo) + (yy - cy) * np.cos(angulo)
    r = np.sqrt((xr / ax) ** 2 + (yr / ay) ** 2)  # 0 centro -> 1 borde
    mascara = r <= 1.0

    if clase == "calidad_promedio":      # Verde / Inmaduro: frio y homogeneo
        base = rng.uniform(15.0, 21.0)
        perfil = base - 1.0 * r
        ruido = rng.normal(0.0, 0.4, (n, n))
    elif clase == "alta_calidad":        # Maduro / Optimo: gradiente radial moderado
        base = rng.uniform(24.0, 31.0)
        perfil = base - 4.0 * (r ** 2)
        ruido = rng.normal(0.0, 0.8, (n, n))
    elif clase == "defectuoso":          # Sobre-maduro: caliente + puntos calientes
        base = rng.uniform(34.0, 41.0)
        perfil = base - 3.0 * (r ** 2)
        ruido = rng.normal(0.0, 1.2, (n, n))
        for _ in range(int(rng.integers(2, 5))):
            hx = cx + rng.uniform(-0.6, 0.6) * ax
            hy = cy + rng.uniform(-0.6, 0.6) * ay
            sigma = rng.uniform(6.0, 14.0)
            perfil = perfil + rng.uniform(3.0, 7.0) * np.exp(
                -(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma ** 2))
            )
    else:
        raise ValueError(f"Clase desconocida: {clase}")

    temp = np.where(mascara, perfil + ruido, temp)

    # --- Ruido impulsivo del sensor (chispeado blanco, ~1% de pixeles) ---
    sal = rng.random((n, n)) < 0.01
    temp[sal] = T_MAX_PALETA

    rel = (temp - T_MIN_PALETA) / (T_MAX_PALETA - T_MIN_PALETA)
    return Image.fromarray(mapa_a_ironbow(rel), mode="RGB")


def generar_particion(nombre_particion, n_por_clase, rng):
    for clase in CLASS_NAMES:
        carpeta = PROCESADOS / nombre_particion / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        for i in range(n_por_clase):
            imagen = generar_imagen(clase, rng)
            imagen.save(carpeta / f"sint_{clase}_{i:04d}.jpg", quality=92)
    print(f"{nombre_particion}: {n_por_clase} imagenes x {len(CLASS_NAMES)} clases")


def main():
    if PROCESADOS.exists():
        shutil.rmtree(PROCESADOS)
    rng = np.random.default_rng(RANDOM_SEED)
    print("Generando dataset termico SINTETICO (simulado, no fruta real)...")
    generar_particion("train", N_TRAIN_POR_CLASE, rng)
    generar_particion("test", N_TEST_POR_CLASE, rng)
    print(f"Dataset sintetico generado en: {PROCESADOS}")


if __name__ == "__main__":
    main()
