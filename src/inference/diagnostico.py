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
    temp_min, temp_max = info["rango_temp"]
    temperatura_estimada = temp_min + (temp_max - temp_min) * confianza_valor

    return {
        "etiqueta": etiqueta,
        "clase_id": clase_id.item(),
        "confianza": round(confianza_valor, 4),
        "categoria": info["etiqueta"],
        "temperatura_estimada": round(temperatura_estimada, 1),
        "color": info["color"],
    }
