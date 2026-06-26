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
