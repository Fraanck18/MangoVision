"""
Logica de inferencia de un mango individual, compartida por el endpoint
sincrono `/diagnostico` (src/api/app.py) y el pipeline de workers en paralelo
(src/pipeline/worker_inference.py), para que ambos caminos produzcan
exactamente el mismo resultado a partir de la misma imagen.
"""
import torch

from src.config import CLASS_NAMES, MATURITY_INFO


def diagnosticar_imagen_pil(imagen_pil, modelo, device, preprocess):
    tensor = preprocess(imagen_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = torch.softmax(logits, dim=1)
        confianza, clase_id = torch.max(probabilidades, dim=1)

    etiqueta = CLASS_NAMES[clase_id.item()]
    info = MATURITY_INFO[etiqueta]
    confianza_valor = confianza.item()

    # [Plan de Contingencia F3] NO se reporta "temperatura en °C": el modelo no mide
    # temperatura, y de un JPEG pseudo-color no se recupera radiometria absoluta. Se
    # expone un INDICE DE INTENSIDAD TERMICA RELATIVA (ITR) en escala 0-100, derivado
    # de la banda relativa de la clase modulada por la confianza. Es un indicador
    # legible, explicitamente RELATIVO, no una medida fisica.
    itr_min, itr_max = info["banda_itr"]
    indice_itr = itr_min + (itr_max - itr_min) * confianza_valor

    return {
        "etiqueta": etiqueta,
        "clase_id": clase_id.item(),
        "confianza": round(confianza_valor, 4),
        "categoria": info["etiqueta"],
        "indice_itr": round(indice_itr, 1),
        "unidad": "ITR (relativo 0-100, no °C)",
        "color": info["color"],
    }
