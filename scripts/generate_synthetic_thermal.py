"""
Generador de dataset termico SINTETICO para PRE-ENTRENAMIENTO (domain adaptation).

[Plan de Contingencia F2 — Aislamiento estricto]
Este dataset es SIMULADO y su unico rol legal es PRE-ENTRENAR la CNN (adaptacion
de dominio). Queda PROHIBIDO que una sola imagen sintetica entre al set de
evaluacion final. Por eso:

  * Se escribe en `dataset_mangos/synthetic_pretrain/train/<clase>` (solo train),
    NO en `dataset_mangos/processed/` (que contiene el dato REAL) y NO se genera
    ninguna particion `test`.
  * El flujo blindado es:  Sintetico -> pre-train CNN  ==>  Real -> fine-tune +
    Test real bloqueado (ver scripts/train_resnet50.py).

[Plan de Contingencia F2 — Inyeccion estocastica]
El generador YA NO debe escupir patrones perfectos y separables por diseno
(eso producia la circularidad tautologica del 96.7%). Se inyecta:
  * Ruido fractal tipo Perlin (varias octavas) en textura y en la frontera.
  * SOLAPE de los rangos termicos entre clases (colas que se pisan) -> las clases
    dejan de ser trivialmente separables por matiz.
  * Perturbacion estocastica del contorno del fruto.

ADVERTENCIA (honestidad): la precision sobre este dataset NO es exactitud sobre
fruta real. La metrica publicable sale del pipeline real + LOGO (ver F6/F7).

Uso:
    python scripts/generate_synthetic_thermal.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED
# La paleta Ironbow vive ahora en un unico modulo compartido con la augmentation
# fisica, para que generador sintetico y augmentation hablen el mismo idioma de
# color termico (encode/decode consistentes).
from src.augmentation.paleta_ironbow import mapa_a_ironbow

# Salida AISLADA de pre-entrenamiento (nunca sobrescribe el dato real processed/).
SALIDA_SINTETICO = RAIZ / "dataset_mangos" / "synthetic_pretrain"

T_MIN_PALETA = 10.0   # extremo frio de la paleta (azul)
T_MAX_PALETA = 45.0   # extremo caliente de la paleta (blanco)

# Solo pre-entrenamiento: un unico pool (sin test). Mas muestras que antes porque
# ahora es el pool de domain adaptation, no el set evaluado.
N_PRETRAIN_POR_CLASE = 200

# Fraccion de imagenes que recibe ruido realista de sensor.
FRACCION_RUIDO_SENSOR = 0.55

# [F2] SOLAPE de rangos termicos por clase (colas que se pisan). Antes los rangos
# eran disjuntos (15-21 / 24-31 / 34-41) -> separables por diseno. Ahora se pisan
# en los bordes para que la clase no sea deducible solo por el matiz dominante.
RANGO_BASE_POR_CLASE = {
    "calidad_promedio": (15.0, 24.0),   # verde/inmaduro (frio) — cola alta pisa a alta_calidad
    "alta_calidad":     (21.0, 33.0),   # maduro/optimo — se solapa por ambos lados
    "defectuoso":       (30.0, 42.0),   # sobre-maduro (caliente) — cola baja pisa a alta_calidad
}

def _campo_suave(n, rng, escala_px, amplitud):
    """Campo de ruido de baja frecuencia (coherente espacialmente) y normalizado."""
    base = rng.normal(0.0, 1.0, (n, n)).astype(np.float32)
    rango = float(base.max() - base.min()) or 1.0
    img = Image.fromarray((((base - base.min()) / rango) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=escala_px))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr * amplitud


def _perlin_octavas(n, rng, octavas=4, escala_base_px=24.0, amplitud=1.0, persistencia=0.5):
    """
    [F2] Ruido fractal tipo Perlin: suma de octavas de _campo_suave a escalas
    decrecientes con amplitud geometricamente menor. Produce estructura
    multiescala (no un patron liso perfecto), rompiendo la separabilidad de
    diseno del generador anterior.
    """
    acumulado = np.zeros((n, n), dtype=np.float32)
    escala = escala_base_px
    amp = 1.0
    for _ in range(octavas):
        acumulado += _campo_suave(n, rng, escala_px=max(escala, 2.0), amplitud=amp)
        escala *= 0.5
        amp *= persistencia
    acumulado = (acumulado - acumulado.mean()) / (acumulado.std() + 1e-6)
    return acumulado * amplitud


def _degradar_ruido_sensor(imagen_pil, rng):
    """Ruido realista de sensor: desenfoque + gaussiano + distorsion de balance de canales."""
    radio_blur = float(rng.uniform(5.0, 11.0))
    imagen = imagen_pil.filter(ImageFilter.GaussianBlur(radius=radio_blur))
    arreglo = np.asarray(imagen, dtype=np.float32)

    escala_canal = rng.uniform(0.3, 2.3, size=3).astype(np.float32)
    arreglo *= escala_canal[None, None, :]

    arreglo += rng.normal(0.0, rng.uniform(32.0, 65.0), arreglo.shape)
    arreglo += rng.uniform(-65.0, 65.0)
    arreglo = np.clip(arreglo, 0, 255).astype(np.uint8)
    return Image.fromarray(arreglo, mode="RGB")


def generar_imagen(clase, rng):
    """
    Genera una imagen termica sintetica (PIL.Image RGB) de la clase indicada, con
    silueta ovoide, borde difuso, moteado de baja frecuencia y — [F2] — ruido
    fractal multiescala + rango termico solapado + contorno perturbado.
    """
    n = IMAGE_SIZE
    yy, xx = np.mgrid[0:n, 0:n]

    # Fondo ambiente frio con gradiente + vignette.
    t_amb = rng.uniform(11.0, 14.5)
    grad = np.linspace(rng.uniform(-1.5, 0.0), rng.uniform(0.0, 2.5), n)
    dist_centro = np.sqrt(((xx - n / 2) / (n / 2)) ** 2 + ((yy - n / 2) / (n / 2)) ** 2)
    vignette = -1.6 * np.clip(dist_centro - 0.6, 0.0, None)
    temp = t_amb + grad[None, :] + vignette + _campo_suave(n, rng, escala_px=18.0, amplitud=0.5)

    # Fruto: elipse deformada.
    cx, cy = rng.uniform(0.42, 0.58) * n, rng.uniform(0.42, 0.58) * n
    ax, ay = rng.uniform(0.30, 0.40) * n, rng.uniform(0.24, 0.34) * n
    angulo = rng.uniform(-0.5, 0.5)

    xr = (xx - cx) * np.cos(angulo) + (yy - cy) * np.sin(angulo)
    yr = -(xx - cx) * np.sin(angulo) + (yy - cy) * np.cos(angulo)
    theta = np.arctan2(yr, xr)
    ph1, ph2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    deform = (1.0
              + 0.16 * np.sin(theta + ph1)
              + 0.07 * np.sin(2 * theta + ph2)
              + 0.04 * np.sin(3 * theta))
    r = np.sqrt((xr / ax) ** 2 + (yr / ay) ** 2) / deform

    # [F2] Perturbacion estocastica del contorno: empuja el radio con ruido fractal
    # de baja frecuencia -> la frontera deja de ser una elipse analitica perfecta.
    r = r + _perlin_octavas(n, rng, octavas=3, escala_base_px=40.0, amplitud=0.06)

    borde = 0.07
    w = 1.0 / (1.0 + np.exp((r - 1.0) / borde))
    halo = np.exp(-(np.clip(r - 1.0, 0.0, None) ** 2) / (2 * 0.16 ** 2))

    # [F2] Rango termico SOLAPADO por clase (colas que se pisan).
    base_min, base_max = RANGO_BASE_POR_CLASE[clase]
    base = rng.uniform(base_min, base_max)

    if clase == "calidad_promedio":
        perfil = base - 1.2 * (r ** 2)
        textura = _perlin_octavas(n, rng, octavas=4, escala_base_px=16.0, amplitud=0.5)
        halo_temp = 1.5
    elif clase == "alta_calidad":
        perfil = base - 4.5 * (r ** 2)
        textura = _perlin_octavas(n, rng, octavas=4, escala_base_px=12.0, amplitud=1.0)
        halo_temp = 2.6
    elif clase == "defectuoso":
        perfil = base - 3.2 * (r ** 2)
        textura = _perlin_octavas(n, rng, octavas=5, escala_base_px=10.0, amplitud=1.5)
        halo_temp = 3.6
        for _ in range(int(rng.integers(2, 5))):
            hx = cx + rng.uniform(-0.55, 0.55) * ax
            hy = cy + rng.uniform(-0.55, 0.55) * ay
            sigma = rng.uniform(7.0, 15.0)
            perfil = perfil + rng.uniform(3.5, 7.0) * np.exp(
                -(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma ** 2))
            )
    else:
        raise ValueError(f"Clase desconocida: {clase}")

    rim = np.exp(-((1.0 - r) / 0.14) ** 2)
    temp_fruto = perfil + textura - rim * rng.uniform(1.0, 2.2)

    temp_fondo = temp + halo * halo_temp * (1.0 - w)
    temp = temp_fondo * (1.0 - w) + temp_fruto * w

    # Ruido fino + puntos calientes aislados raros.
    temp += rng.normal(0.0, 0.35, (n, n))
    sal = rng.random((n, n)) < 0.0015
    temp[sal] = np.minimum(temp[sal] + rng.uniform(6.0, 12.0), T_MAX_PALETA)

    rel = (temp - T_MIN_PALETA) / (T_MAX_PALETA - T_MIN_PALETA)
    return Image.fromarray(mapa_a_ironbow(rel), mode="RGB")


def generar_pool_pretrain(n_por_clase, rng):
    """Genera el UNICO pool de pre-entrenamiento (sin particion test). -> synthetic_pretrain/train/."""
    n_ruidosas = 0
    for clase in CLASS_NAMES:
        carpeta = SALIDA_SINTETICO / "train" / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        for i in range(n_por_clase):
            imagen = generar_imagen(clase, rng)
            if rng.random() < FRACCION_RUIDO_SENSOR:
                imagen = _degradar_ruido_sensor(imagen, rng)
                n_ruidosas += 1
            imagen.save(carpeta / f"sint_{clase}_{i:04d}.jpg", quality=92)
    total = n_por_clase * len(CLASS_NAMES)
    print(f"pretrain: {total} imagenes x {len(CLASS_NAMES)} clases "
          f"({n_ruidosas} con ruido de sensor, {n_ruidosas/total:.1%})")


def main():
    # Solo se limpia la carpeta SINTETICA; jamas se toca dataset_mangos/processed/.
    if SALIDA_SINTETICO.exists():
        shutil.rmtree(SALIDA_SINTETICO)
    rng = np.random.default_rng(RANDOM_SEED)
    print("Generando pool SINTETICO de PRE-ENTRENAMIENTO (simulado, no fruta real, "
          "PROHIBIDO en test)...")
    generar_pool_pretrain(N_PRETRAIN_POR_CLASE, rng)
    print(f"Pool sintetico de pre-entrenamiento generado en: {SALIDA_SINTETICO / 'train'}")


if __name__ == "__main__":
    main()
