"""
Etiquetado PROGRAMATICO por firma termica (Weak Supervision determinista).

Idea (del plan de solucion): el error del pseudo-etiquetado clasico es poner a la
RED a adivinar sus propias etiquetas -> concordancia circular (fallo #2). Aqui las
etiquetas NO las pone un modelo: las pone una FUNCION MATEMATICA determinista sobre
la firma termica de la matriz. El modelo de Deep Learning nunca se evalua contra lo
que el mismo invento, sino contra estados generados/medidos por reglas fisicas.

[HONESTIDAD — coherencia con F3]  Las camaras FLIR de consumo NO entregan
radiometria absoluta (°C): entregan pseudo-color de 8 bits. Por eso —igual que
`src/domain_physics/thermal_inertia.py`— este modulo trabaja en el INDICE DE
INTENSIDAD TERMICA RELATIVA (ITR) en [0,1], normalizado por imagen, NUNCA en grados
Celsius inventados. Los "umbrales fisicos" son umbrales de FIRMA TERMICA RELATIVA
(intensidad, dispersion espacial, gradiente centro-periferia), que es exactamente
la evidencia que un solo frame estatico deja (PEDTE). Esto mantiene la disciplina
de honestidad del proyecto: no se finge medir temperatura absoluta.

Dos usos:
  1. `etiquetar_por_firma_termica(itr, mascara)` -> etiqueta determinista de una
     matriz ITR real o sintetica, segun umbrales de dominio.
  2. Motor de simulacion parametrica: `simular_estres_termico`, `simular_enfriamiento`,
     `firma_madurez` transforman una matriz base empujandola a un regimen termico y
     devuelven (matriz_nueva, etiqueta), donde la etiqueta la calcula (1) sobre el
     resultado -> auto-etiquetado consistente con la fisica aplicada.

Estas muestras se marcan con procedencia `synthetic_physics` y quedan ESTRICTAMENTE
excluidas de la evaluacion honesta (ver honest_grouped_cv.py y evaluator.py): sirven
para "anabolizar" el entrenamiento, jamas para medir exactitud.
"""
import numpy as np

from src.config import CLASS_NAMES
from src.domain_physics.thermal_inertia import segmentar_fruto, gradiente_centro_periferia

# Umbrales de FIRMA TERMICA RELATIVA (ITR [0,1]), no °C. Calibrados para separar los
# tres regimenes de dominio del proyecto (ver MATURITY_INFO en src/config.py).
ITR_CALIENTE = 0.72          # pixel "caliente" respecto a la propia imagen
FRAC_CALIENTE_DEFECTO = 0.12  # fraccion de fruto caliente -> estres/sobre-maduro
ITR_MEDIO_FRIO = 0.42        # fruto globalmente frio -> verde/inmaduro
GRADIENTE_MADUREZ = 0.06     # gradiente centro-periferia de disipacion -> maduro


def _mascara(itr, mascara):
    if mascara is not None:
        return mascara
    m = segmentar_fruto(itr)
    # Si la segmentacion no encuentra fruto (matriz sintetica sin fondo frio),
    # se usa toda la imagen como region de analisis.
    return m if m.sum() > 0 else np.ones_like(itr, dtype=bool)


def descriptores_firma_termica(itr, mascara=None):
    """
    Descriptores espaciales estaticos de la firma termica de una matriz ITR:
      - itr_medio: intensidad media del fruto (proxy de calor global).
      - sigma_espacial: desviacion espacial (heterogeneidad termica).
      - fraccion_caliente: fraccion del fruto por encima de ITR_CALIENTE (focos).
      - gradiente_centro_periferia: huella PEDTE (centro vs borde).
    Todo en unidades de ITR relativo.
    """
    itr = np.asarray(itr, dtype=np.float32)
    mascara = _mascara(itr, mascara)
    valores = itr[mascara]
    if valores.size == 0:
        return {"itr_medio": float("nan"), "sigma_espacial": float("nan"),
                "fraccion_caliente": float("nan"), "gradiente_centro_periferia": float("nan")}
    return {
        "itr_medio": float(valores.mean()),
        "sigma_espacial": float(valores.std()),
        "fraccion_caliente": float((valores >= ITR_CALIENTE).mean()),
        "gradiente_centro_periferia": gradiente_centro_periferia(itr, mascara),
    }


def etiquetar_por_firma_termica(itr, mascara=None):
    """
    Regla DETERMINISTA (no un modelo) que asigna una de las 3 clases del proyecto a
    partir de la firma termica relativa. Orden de decision por severidad fisica:

      1. Focos calientes / intensidad alta  -> 'defectuoso'  (estres termico /
         sobre-maduro): mucha fraccion caliente o ITR medio alto.
      2. Fruto globalmente frio y uniforme   -> 'calidad_promedio' (verde/inmaduro):
         ITR medio bajo y poco gradiente.
      3. Regimen intermedio con gradiente de disipacion -> 'alta_calidad' (maduro).

    Devuelve el nombre de la clase (str en CLASS_NAMES).
    """
    f = descriptores_firma_termica(itr, mascara)
    grad = 0.0 if np.isnan(f["gradiente_centro_periferia"]) else abs(f["gradiente_centro_periferia"])

    if f["fraccion_caliente"] >= FRAC_CALIENTE_DEFECTO or f["itr_medio"] >= 0.62:
        return "defectuoso"
    if f["itr_medio"] <= ITR_MEDIO_FRIO and grad < GRADIENTE_MADUREZ:
        return "calidad_promedio"
    return "alta_calidad"


# ---------------------------------------------------------------------------
# Motor de simulacion termica parametrica (Physics-Informed, auto-etiquetado)
# ---------------------------------------------------------------------------
def _coords(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return yy, xx


def base_canonica_fruto(n=224, rng=None, nivel=0.14, fondo=0.18, amplitud=0.05):
    """
    Substrato SINTETICO neutro para el motor de simulacion: un fruto ovoide MUY
    suave (baja heterogeneidad y gradiente pequeño), casi uniforme, sobre el que
    cada transformador parametrico imprime DECISIVAMENTE su regimen termico.

    Se mantiene deliberadamente plano: cada generador (estres/enfriamiento/madurez)
    impone su propia firma, asi que el substrato solo aporta un punto de partida
    neutro. Usar un substrato canonico (no una imagen real) garantiza que la firma
    resultante dependa de la FISICA aplicada, no de la estructura fortuita de un
    fruto concreto -> generacion CONTROLABLE y 100% sintetica (no hereda la
    identidad de ningun mango real). Devuelve un mapa ITR [0,1].
    """
    rng = rng or np.random.default_rng()
    yy, xx = _coords(n)
    cx, cy = n / 2, n / 2
    r = np.sqrt(((xx - cx) / (0.42 * n)) ** 2 + ((yy - cy) / (0.34 * n)) ** 2)
    fruto = np.clip(1.0 - r, 0.0, 1.0)                       # 1 en centro, 0 fuera
    ruido = amplitud * rng.standard_normal((n, n)).astype(np.float32)
    return np.clip(nivel * fruto + fondo + ruido, 0.0, 1.0)


def simular_estres_termico(itr_base, rng, n_focos=(3, 6), intensidad=(0.25, 0.45), ganancia=1.15):
    """
    Simula ESTRES TERMICO / sobre-madurez: concentra calor en focos gaussianos y
    sube la ganancia global -> cruza el umbral de fraccion caliente. Modela
    deshidratacion / alta exposicion. Devuelve (itr_nueva [0,1], etiqueta).
    """
    itr = np.asarray(itr_base, dtype=np.float32).copy()
    n = itr.shape[0]
    yy, xx = _coords(n)
    for _ in range(int(rng.integers(*n_focos))):
        cx, cy = rng.uniform(0.3, 0.7) * n, rng.uniform(0.3, 0.7) * n
        sigma = rng.uniform(0.06, 0.14) * n
        itr += float(rng.uniform(*intensidad)) * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    itr = np.clip(itr * ganancia, 0.0, 1.0)
    return itr, etiquetar_por_firma_termica(itr)


def simular_enfriamiento(itr_base, factor_amplitud=0.3, nivel_frio=0.22):
    """
    Simula ENFRIAMIENTO / normalidad (sombra, recoleccion temprana): HOMOGENEIZA el
    campo termico (reduce la amplitud espacial hacia la media, incl. el gradiente
    centro-periferia) y lo desplaza a un nivel frio -> fruto frio y uniforme
    (verde/inmaduro).

    Clave: NO se renormaliza min-max al final (eso re-estiraria el gradiente que la
    difusion acaba de suavizar y arruinaria la firma "uniforme"). Se conserva la
    amplitud REDUCIDA. `factor_amplitud` < 1 escala toda la heterogeneidad (varianza
    y gradiente) por igual, modelando la difusion termica de la sombra. Devuelve
    (itr_nueva [0,1], etiqueta).
    """
    itr = np.asarray(itr_base, dtype=np.float32)
    media = float(itr.mean())
    # Reduce la heterogeneidad hacia la media y recentra en un nivel frio.
    itr = np.clip(nivel_frio + (itr - media) * factor_amplitud, 0.0, 1.0)
    return itr, etiquetar_por_firma_termica(itr)


def firma_madurez(itr_base, rng, realce_centro=0.18, sigma_frac=0.28):
    """
    Inyecta la FIRMA DE MADUREZ: un foco de disipacion central suave que crea un
    gradiente centro->periferia moderado (madurez optima), sin llegar a foco
    caliente de defecto. Devuelve (itr_nueva [0,1], etiqueta).
    """
    itr = np.asarray(itr_base, dtype=np.float32).copy()
    n = itr.shape[0]
    yy, xx = _coords(n)
    cx, cy = n / 2 + rng.uniform(-0.05, 0.05) * n, n / 2 + rng.uniform(-0.05, 0.05) * n
    sigma = sigma_frac * n
    itr = itr + realce_centro * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    # Nivel medio moderado para caer en el regimen intermedio (ni frio ni caliente).
    itr = np.clip(0.30 + 0.40 * (itr - itr.min()) / ((itr.max() - itr.min()) or 1.0), 0.0, 1.0)
    return itr, etiquetar_por_firma_termica(itr)


# Mapa clase -> generador parametrico, para producir un pool balanceado auto-etiquetado.
GENERADORES_POR_CLASE = {
    "defectuoso": lambda base, rng: simular_estres_termico(base, rng),
    "calidad_promedio": lambda base, rng: simular_enfriamiento(base),
    "alta_calidad": lambda base, rng: firma_madurez(base, rng),
}

assert set(GENERADORES_POR_CLASE) == set(CLASS_NAMES), "Generadores fisicos != CLASS_NAMES"
