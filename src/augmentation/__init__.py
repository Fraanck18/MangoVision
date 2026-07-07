"""
Paquete de DATA AUGMENTATION avanzada para termografia de mango.

Regla que filtra TODA la libreria: las imagenes son pseudo-color Ironbow, donde
el MATIZ es la señal (temperatura relativa / ITR). Por eso NUNCA se perturba el
color de forma arbitraria (ColorJitter, hue, saturation, grayscale destruirian la
etiqueta). Toda tecnica aqui es "termicamente segura":

  * o es puramente geometrica (flip/rotacion/afin/elastica/erasing),
  * o opera en el DOMINIO TERMICO (ITR) y re-codifica con la paleta fisica,
  * o mezcla en el espacio de la etiqueta (MixUp) sin inventar matices.

Ver `docs/DATA_AUGMENTATION.md` para el mapa completo de tecnicas, su
justificacion y donde vive cada una (online / offline / streaming).
"""
from src.augmentation.paleta_ironbow import (
    PALETA_IRONBOW,
    mapa_a_ironbow,
    ironbow_a_itr,
)
from src.augmentation.termica_fisica import (
    AugmentacionTermicaFisica,
    perturbar_itr_fisico,
)
from src.augmentation.regularizadores import (
    DeformacionElastica,
    AutoAugmentGeometrico,
    obtener_transformaciones_entrenamiento_avanzado,
    mixup_lote,
    perdida_mixup,
)
from src.augmentation.diffaugment import DiffAugment

__all__ = [
    "PALETA_IRONBOW",
    "mapa_a_ironbow",
    "ironbow_a_itr",
    "AugmentacionTermicaFisica",
    "perturbar_itr_fisico",
    "DeformacionElastica",
    "AutoAugmentGeometrico",
    "obtener_transformaciones_entrenamiento_avanzado",
    "mixup_lote",
    "perdida_mixup",
    "DiffAugment",
]
