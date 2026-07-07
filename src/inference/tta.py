"""
Tecnica #9 — Test-Time Augmentation (TTA).

En inferencia (no en entrenamiento) se promedian las predicciones sobre varias
vistas TERMICAMENTE SEGURAS de la MISMA imagen (original + flips + rotaciones de
90°). Promediar el softmax sobre vistas equivalentes reduce la varianza por frame
y suele subir un poco la exactitud sin reentrenar nada. Es especialmente barato en
el pipeline Spark, donde ya se procesa por lotes.

Solo transformaciones que NO cambian la clase de un fruto: flips horizontal/
vertical y rotaciones multiplo de 90° (la orientacion del mango es irrelevante).
No se usa color: seria romper el pseudo-color termico.
"""
import torch


def _vistas_seguras(lote):
    """
    Genera las vistas TTA de un lote (N, C, H, W): identidad, flip-H, flip-V,
    rot90, rot270. Todas preservan la etiqueta en termografia de fruto.
    """
    return [
        lote,
        torch.flip(lote, dims=[3]),          # flip horizontal
        torch.flip(lote, dims=[2]),          # flip vertical
        torch.rot90(lote, k=1, dims=[2, 3]),  # 90°
        torch.rot90(lote, k=3, dims=[2, 3]),  # 270°
    ]


@torch.no_grad()
def probabilidades_tta(modelo, lote, device=None):
    """
    Devuelve las PROBABILIDADES (softmax) promediadas sobre las vistas TTA para un
    lote de tensores ya preprocesados (N, C, H, W). El modelo debe estar en eval().
    """
    if device is not None:
        lote = lote.to(device)
    acumulado = None
    vistas = _vistas_seguras(lote)
    for vista in vistas:
        probs = torch.softmax(modelo(vista), dim=1)
        acumulado = probs if acumulado is None else acumulado + probs
    return acumulado / len(vistas)


@torch.no_grad()
def diagnosticar_imagen_pil_tta(imagen_pil, modelo, device, preprocess):
    """
    Variante de `src.inference.diagnostico.diagnosticar_imagen_pil` con TTA: mismo
    contrato de salida, pero la clase y la confianza salen del softmax promediado
    sobre las vistas seguras. Util en el evaluador online por-frame.
    """
    from src.config import CLASS_NAMES, MATURITY_INFO

    tensor = preprocess(imagen_pil).unsqueeze(0).to(device)
    probs = probabilidades_tta(modelo, tensor)
    confianza, clase_id = torch.max(probs, dim=1)

    etiqueta = CLASS_NAMES[clase_id.item()]
    info = MATURITY_INFO[etiqueta]
    confianza_valor = confianza.item()
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
        "tta": True,
    }
