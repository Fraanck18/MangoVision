"""
Fundamento fisico del diagnostico termico no destructivo en mango.

La camara termica NO realiza una "radiografia" del fruto: el infrarrojo solo
capta radiacion emitida por la superficie. La deteccion de necrosis interna es
posible porque el tejido necrosado altera el calor especifico (c) y la
conductividad termica (k) de la pulpa, lo que modifica la velocidad con la que
el calor migra desde el nucleo hacia la cascara y se disipa al ambiente.

Esa dinamica de disipacion se modela con la Ley de Enfriamiento de Newton:

    dT/dt = -h * (T(t) - T_amb)

cuya solucion analitica es:

    T(t) = T_amb + (T0 - T_amb) * exp(-h * t)

donde:
    T(t)   : temperatura superficial del fruto en el instante t [°C]
    T_amb  : temperatura ambiente de equilibrio [°C]
    T0     : temperatura superficial inicial (post-cosecha/post-refrigeracion) [°C]
    h      : coeficiente de enfriamiento, h = k_eff / (rho * c_eff * V/A) [1/s]

Un tejido necrosado tiene una matriz celular colapsada con mayor contenido de
agua libre y aire intracelular, lo que reduce su calor especifico efectivo
(c_eff) y altera k_eff respecto al tejido sano. Esto provoca que la region
afectada se enfrie o caliente a una tasa h distinta a la del tejido circundante,
generando un gradiente termico superficial anomalo y detectable por la camara
infrarroja, sin necesidad de corte destructivo.
"""
import numpy as np


def temperatura_newton(t, t0, t_amb, h):
    """Evalua T(t) = T_amb + (T0 - T_amb) * exp(-h * t) para un Lote de tiempos t."""
    t = np.asarray(t, dtype=float)
    return t_amb + (t0 - t_amb) * np.exp(-h * t)


def coeficiente_enfriamiento(k_eff, rho, c_eff, volumen, area_superficial):
    """
    Calcula h = k_eff / (rho * c_eff * (V / A)) a partir de propiedades fisicas
    del tejido. Un c_eff menor (tejido necrosado) incrementa h, acelerando la
    respuesta termica de la region afectada respecto al tejido sano.
    """
    if rho <= 0 or c_eff <= 0 or volumen <= 0 or area_superficial <= 0:
        raise ValueError("Las propiedades fisicas deben ser estrictamente positivas.")
    return k_eff / (rho * c_eff * (volumen / area_superficial))


def gradiente_termico_anomalo(temp_region_sana, temp_region_sospechosa):
    """
    Cuantifica la anomalia termica superficial como la diferencia absoluta
    entre la temperatura de una region de referencia sana y una region
    sospechosa de necrosis, ambas medidas en el mismo instante t.
    """
    temp_region_sana = np.asarray(temp_region_sana, dtype=float)
    temp_region_sospechosa = np.asarray(temp_region_sospechosa, dtype=float)
    return np.abs(temp_region_sana - temp_region_sospechosa)


def clasificar_anomalia(delta_temp, umbral_alerta_celsius=0.8):
    """
    Indica si el gradiente termico observado supera el umbral empirico de
    alerta (definido a partir de la calibracion experimental del sensor),
    senalando candidato a necrosis celular o desorden metabolico interno.
    """
    return bool(np.max(delta_temp) >= umbral_alerta_celsius)


# ---------------------------------------------------------------------------
# Extraccion de la anomalia termica directamente desde la imagen pseudo-color
# ---------------------------------------------------------------------------
#
# Las camaras termicas de consumo (FLIR One) no exportan radiometria cruda en
# el JPG: exportan una imagen ya pintada con una paleta de falso color
# (tipo "Ironbow"). No hay forma de recuperar grados Celsius absolutos sin la
# barra de calibracion de cada captura, pero SI se puede recuperar un orden
# relativo frio->caliente a partir del color, y aplicar la misma logica de
# `gradiente_termico_anomalo` / `clasificar_anomalia` pixel a pixel: se
# segmenta el fruto, se calcula SU propia media termica (el "deberia estar a"
# de la Ley de Newton para ESE fruto en ESE instante), y se busca una
# sub-region que se desvie de esa media propia mas de lo esperado por ruido
# del sensor. Esto es deliberadamente distinto a comparar lecturas absolutas
# entre fotos distintas (eso mezcla condiciones ambientales/de captura
# distintas y no es comparable).

_PALETA_IRONBOW_FRIO_A_CALIENTE = np.array([
    (0.00, (0, 0, 160)),     # azul = frio
    (0.15, (0, 200, 255)),   # cian
    (0.35, (0, 200, 0)),     # verde
    (0.55, (255, 230, 0)),   # amarillo
    (0.75, (255, 130, 0)),   # naranja
    (0.90, (230, 0, 0)),     # rojo
    (1.00, (255, 255, 255)),  # blanco = mas caliente
], dtype=object)


def limpiar_ruido_impulsivo(imagen_rgb, tamano_kernel=5):
    """
    Las fotos termicas del dataset muestran un patron de puntos blancos
    dispersos ("chispeado"). Una prueba de normalidad D'Agostino K^2 sobre el
    residuo (imagen - GaussianBlur) lo rechaza con p~0 -- NO es ruido
    gaussiano, es mas consistente con ruido impulsivo. Un filtro gaussiano
    difuminaria ese ruido en vez de eliminarlo (esparce el valor atipico
    sobre los vecinos); el filtro de mediana si lo remueve porque ignora
    valores atipicos aislados en la ventana.
    """
    import cv2

    arreglo = np.asarray(imagen_rgb, dtype=np.uint8)
    return cv2.medianBlur(arreglo, tamano_kernel)


def ecualizar_contraste(imagen_rgb):
    """
    Ecualizacion de histograma (CDF normalizada) sobre el canal de luminancia,
    para resaltar gradientes internos sutiles del fruto que quedan
    comprimidos en un rango estrecho de intensidad antes de decodificar la
    paleta de color a temperatura relativa.
    """
    import cv2

    arreglo = np.asarray(imagen_rgb, dtype=np.uint8)
    ycrcb = cv2.cvtColor(arreglo, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def mapa_temperatura_relativa(imagen_rgb):
    """
    Decodifica una imagen en pseudo-color tipo Ironbow a un mapa 2D de
    temperatura relativa en [0, 1] (0 = extremo frio de la paleta, 1 =
    extremo caliente), por distancia de color mas cercana a la paleta de
    referencia. No es temperatura absoluta en grados.
    """
    arreglo = np.asarray(imagen_rgb, dtype=float)
    valores_paleta = np.stack(_PALETA_IRONBOW_FRIO_A_CALIENTE[:, 0]).astype(float)
    colores_paleta = np.stack(_PALETA_IRONBOW_FRIO_A_CALIENTE[:, 1]).astype(float)

    alto, ancho, _ = arreglo.shape
    pixeles = arreglo.reshape(-1, 3)

    distancias = np.linalg.norm(pixeles[:, None, :] - colores_paleta[None, :, :], axis=2)
    indices_mas_cercanos = np.argmin(distancias, axis=1)
    mapa_plano = valores_paleta[indices_mas_cercanos]
    return mapa_plano.reshape(alto, ancho)


def segmentar_fruto(mapa_temperatura, umbral_zona_calida=0.55, area_minima_fraccion=0.02):
    """
    Aproxima la mascara del fruto como la mayor componente conexa de pixeles
    en el extremo caliente de la paleta (el fruto se ve como una elipse
    calida central rodeada de un fondo mas frio en las fotos del dataset).
    """
    from scipy import ndimage

    mascara_caliente = mapa_temperatura >= umbral_zona_calida
    etiquetas, num_componentes = ndimage.label(mascara_caliente)
    if num_componentes == 0:
        return np.zeros_like(mascara_caliente, dtype=bool)

    tamanos = ndimage.sum(mascara_caliente, etiquetas, range(1, num_componentes + 1))
    componente_mas_grande = np.argmax(tamanos) + 1

    mascara_fruto = etiquetas == componente_mas_grande
    if mascara_fruto.sum() < area_minima_fraccion * mascara_caliente.size:
        return np.zeros_like(mascara_caliente, dtype=bool)
    return mascara_fruto


def extraer_anomalia_termica_imagen(imagen_rgb, umbral_desviacion_z=1.5, area_minima_anomalia_fraccion=0.01):
    """
    Pipeline completo sobre una imagen real: decodifica temperatura relativa,
    segmenta el fruto, calcula su propia media termica (linea base de Newton
    para ESE fruto) y mide si existe una sub-region significativamente mas
    fria que esa media (candidato a necrosis interna / desorden metabolico).

    Devuelve un dict con features interpretables, no una etiqueta: la
    decision de clasificar se deja al modelo/umbral que consuma esta funcion.

    Preprocesamiento aplicado antes de decodificar la paleta (en este orden):
    1. `limpiar_ruido_impulsivo`: filtro de mediana, remueve el chispeado
       impulsivo confirmado por la prueba D'Agostino K^2 (no es gaussiano).
    2. `ecualizar_contraste`: ecualizacion de histograma sobre luminancia,
       para que gradientes internos sutiles no queden comprimidos.
    """
    imagen_limpia = limpiar_ruido_impulsivo(imagen_rgb)
    imagen_realzada = ecualizar_contraste(imagen_limpia)

    mapa_temperatura = mapa_temperatura_relativa(imagen_realzada)
    mascara_fruto = segmentar_fruto(mapa_temperatura)

    if mascara_fruto.sum() == 0:
        return {
            "fruto_detectado": False,
            "temp_media_fruto": float("nan"),
            "temp_std_fruto": float("nan"),
            "fraccion_area_anomala_fria": float("nan"),
            "desviacion_maxima_fria": float("nan"),
        }

    temperaturas_fruto = mapa_temperatura[mascara_fruto]
    media_fruto = temperaturas_fruto.mean()
    std_fruto = temperaturas_fruto.std() + 1e-8

    desviaciones_z = (media_fruto - temperaturas_fruto) / std_fruto  # positivo = mas frio que la media
    pixeles_anomalos_frios = desviaciones_z >= umbral_desviacion_z

    return {
        "fruto_detectado": True,
        "temp_media_fruto": float(media_fruto),
        "temp_std_fruto": float(std_fruto),
        "fraccion_area_anomala_fria": float(pixeles_anomalos_frios.mean()),
        "desviacion_maxima_fria": float(desviaciones_z.max()),
    }
