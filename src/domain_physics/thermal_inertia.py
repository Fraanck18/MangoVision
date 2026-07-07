"""
Fundamento fisico del diagnostico termico no destructivo en mango:
Perfil Espacial de Distribucion Termica Estatica (PEDTE).

[PIVOTE TEORICO — Plan de Contingencia F1]
El dato real disponible es un UNICO frame estatico (no una secuencia). Medir la
tasa de enfriamiento dinamica (dT/dt) de la Ley de Newton es matematicamente
imposible con un solo instante, por lo que este modulo NO calcula ni simula
derivadas temporales. En su lugar explota la EVIDENCIA ESPACIAL que la inercia
termica deja en un solo frame:

  - La inercia termica altera la DISTRIBUCION ESPACIAL del calor interno ante
    cambios ambientales sutiles. Un mango con dano interno o distinta madurez
    retiene el calor de forma HETEROGENEA en su superficie.
  - Esa heterogeneidad es captable en un unico frame mediante el gradiente
    termico centro-periferia (PEDTE) y las anomalias locales respecto a la
    propia media del fruto.

[NOMENCLATURA — Plan de Contingencia F3]
Las camaras de consumo (FLIR One) NO exportan radiometria cruda: entregan un
JPEG de 8 bits en pseudo-color. Recuperar "grados Celsius" de ahi es recuperar
ruido de compresion. Por eso este modulo trabaja con un INDICE DE INTENSIDAD
TERMICA RELATIVA (ITR) en [0, 1], normalizado localmente por imagen, NUNCA con
temperaturas absolutas en °C.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Descriptores espaciales de anomalia (un solo instante, sin dinamica temporal)
# ---------------------------------------------------------------------------

def gradiente_termico_anomalo(itr_region_referencia, itr_region_sospechosa):
    """
    Cuantifica la anomalia espacial como la diferencia absoluta de ITR entre una
    region de referencia (tejido aparentemente sano) y una region sospechosa,
    ambas del MISMO frame. Es una comparacion espacial intra-imagen, no temporal.
    """
    itr_region_referencia = np.asarray(itr_region_referencia, dtype=float)
    itr_region_sospechosa = np.asarray(itr_region_sospechosa, dtype=float)
    return np.abs(itr_region_referencia - itr_region_sospechosa)


def clasificar_anomalia(delta_itr, umbral_alerta_itr=0.05):
    """
    Indica si el gradiente espacial de ITR observado supera el umbral de alerta,
    senalando candidato a necrosis celular o desorden metabolico interno. El
    umbral se expresa en unidades de ITR (relativo [0,1]), no en °C.
    """
    return bool(np.max(delta_itr) >= umbral_alerta_itr)


# ---------------------------------------------------------------------------
# Decodificacion robusta de la imagen pseudo-color -> ITR local (F3)
# ---------------------------------------------------------------------------
#
# En vez de asumir una paleta Ironbow fija y global (fragil ante distintas
# camaras/ajustes y ante la compresion JPEG), se recupera un INDICE relativo por
# imagen mediante normalizacion Min-Max local, y se suaviza inmediatamente con un
# Gaussiano ligero (3x3, sigma=0.5) para mitigar los bloques de cuantificacion
# del JPEG. Se compara cada fruto contra SU PROPIA distribucion, no lecturas
# absolutas entre fotos (que mezclarian condiciones de captura no comparables).

def limpiar_ruido_impulsivo(imagen_rgb, tamano_kernel=5):
    """
    Filtro de mediana contra el "chispeado" de puntos blancos. Una prueba de
    normalidad D'Agostino K^2 sobre el residuo (imagen - GaussianBlur) rechaza la
    hipotesis gaussiana (p~0): el ruido es impulsivo, y la mediana lo elimina en
    vez de esparcirlo como haria un Gaussiano.
    """
    import cv2

    arreglo = np.asarray(imagen_rgb, dtype=np.uint8)
    return cv2.medianBlur(arreglo, tamano_kernel)


def ecualizar_contraste(imagen_rgb):
    """
    Ecualizacion de histograma sobre la luminancia (YCrCb) para resaltar
    gradientes internos sutiles que quedan comprimidos en un rango estrecho de
    intensidad antes de normalizar a ITR.
    """
    import cv2

    arreglo = np.asarray(imagen_rgb, dtype=np.uint8)
    ycrcb = cv2.cvtColor(arreglo, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def normalizar_itr_local(imagen_rgb):
    """
    Normalizacion Min-Max LOCAL (por imagen) del canal de luminancia -> mapa 2D de
    ITR en [0, 1]:  ITR = (I - I_min) / (I_max - I_min).

    No asume ninguna paleta global fija: cada imagen se auto-referencia, de modo
    que el resultado es robusto a diferencias de camara/ajuste y a corrimientos
    globales de brillo. Devuelve un mapa donde 0 = pixel mas frio de ESA imagen y
    1 = mas caliente.
    """
    arreglo = np.asarray(imagen_rgb, dtype=np.float32)
    if arreglo.ndim == 3:
        # Luminancia perceptual (Rec. 601). El pseudo-color Ironbow es monotono
        # frio->caliente, asi que la luminancia preserva el orden termico relativo.
        luminancia = arreglo @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        luminancia = arreglo

    i_min = float(luminancia.min())
    i_max = float(luminancia.max())
    rango = (i_max - i_min) or 1.0
    return (luminancia - i_min) / rango


def suavizar_bloques_jpeg(mapa_itr, sigma=0.5):
    """
    Gaussiano ligero 3x3 (sigma=0.5) aplicado INMEDIATAMENTE despues de decodificar
    a ITR, para atenuar los bloques de cuantificacion 8x8 del JPEG sin borrar los
    gradientes internos reales.
    """
    import cv2

    return cv2.GaussianBlur(mapa_itr.astype(np.float32), (3, 3), sigmaX=sigma, sigmaY=sigma)


def mapa_itr_local(imagen_rgb):
    """Decodificacion completa: luminancia -> Min-Max local -> Gaussiano 3x3. -> ITR [0,1]."""
    return suavizar_bloques_jpeg(normalizar_itr_local(imagen_rgb))


def segmentar_fruto(mapa_itr, umbral_zona_calida=0.55, area_minima_fraccion=0.02):
    """
    Aproxima la mascara del fruto como la mayor componente conexa de pixeles en el
    extremo caliente del ITR (el fruto se ve como una elipse calida central rodeada
    de un fondo mas frio en las fotos del dataset).
    """
    from scipy import ndimage

    mascara_caliente = mapa_itr >= umbral_zona_calida
    etiquetas, num_componentes = ndimage.label(mascara_caliente)
    if num_componentes == 0:
        return np.zeros_like(mascara_caliente, dtype=bool)

    tamanos = ndimage.sum(mascara_caliente, etiquetas, range(1, num_componentes + 1))
    componente_mas_grande = np.argmax(tamanos) + 1

    mascara_fruto = etiquetas == componente_mas_grande
    if mascara_fruto.sum() < area_minima_fraccion * mascara_caliente.size:
        return np.zeros_like(mascara_caliente, dtype=bool)
    return mascara_fruto


def gradiente_centro_periferia(mapa_itr, mascara_fruto):
    """
    Feature nucleo de PEDTE (F1): diferencia de ITR medio entre el CENTRO del fruto
    y su PERIFERIA. La inercia termica hace que un fruto con distinta madurez/dano
    retenga el calor de forma heterogenea, produciendo un gradiente radial
    centro->borde detectable en un unico frame estatico.

    Devuelve (itr_centro_medio - itr_periferia_medio); positivo = centro mas
    caliente que el borde.
    """
    if mascara_fruto.sum() == 0:
        return float("nan")

    ys, xs = np.nonzero(mascara_fruto)
    cy, cx = ys.mean(), xs.mean()
    # Radio normalizado de cada pixel del fruto respecto a su centroide.
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    radio_max = dist.max() or 1.0
    dist_norm = dist / radio_max

    valores = mapa_itr[ys, xs]
    centro = valores[dist_norm <= 0.4]
    periferia = valores[dist_norm >= 0.7]
    if centro.size == 0 or periferia.size == 0:
        return float("nan")
    return float(centro.mean() - periferia.mean())


def extraer_perfil_espacial_termico(imagen_rgb, umbral_desviacion_z=1.5):
    """
    Pipeline PEDTE completo sobre una imagen real (un solo frame):
      1. limpiar_ruido_impulsivo (mediana) -> quita el chispeado impulsivo.
      2. ecualizar_contraste -> realza gradientes internos sutiles.
      3. mapa_itr_local -> Min-Max local + Gaussiano 3x3 (ITR robusto, F3).
      4. segmentar_fruto -> mascara del fruto.
      5. features espaciales estaticas (NO derivadas temporales, F1).

    Devuelve un dict de features interpretables en unidades de ITR relativo, no una
    etiqueta: la decision de clasificar se deja al modelo/umbral que la consuma.
    """
    imagen_limpia = limpiar_ruido_impulsivo(imagen_rgb)
    imagen_realzada = ecualizar_contraste(imagen_limpia)

    mapa_itr = mapa_itr_local(imagen_realzada)
    mascara_fruto = segmentar_fruto(mapa_itr)

    if mascara_fruto.sum() == 0:
        return {
            "fruto_detectado": False,
            "itr_medio_fruto": float("nan"),
            "itr_std_fruto": float("nan"),
            "fraccion_area_anomala_baja": float("nan"),
            "desviacion_maxima_baja": float("nan"),
            "gradiente_centro_periferia": float("nan"),
        }

    itr_fruto = mapa_itr[mascara_fruto]
    media_fruto = itr_fruto.mean()
    std_fruto = itr_fruto.std() + 1e-8

    desviaciones_z = (media_fruto - itr_fruto) / std_fruto  # positivo = mas frio que la media propia
    pixeles_anomalos_bajos = desviaciones_z >= umbral_desviacion_z

    return {
        "fruto_detectado": True,
        "itr_medio_fruto": float(media_fruto),
        "itr_std_fruto": float(std_fruto),
        "fraccion_area_anomala_baja": float(pixeles_anomalos_bajos.mean()),
        "desviacion_maxima_baja": float(desviaciones_z.max()),
        "gradiente_centro_periferia": gradiente_centro_periferia(mapa_itr, mascara_fruto),
    }


# Alias de compatibilidad: el nombre historico apuntaba a un modelo "de Newton";
# ahora es explicitamente el perfil espacial estatico (PEDTE).
extraer_anomalia_termica_imagen = extraer_perfil_espacial_termico
