"""
Auditoria dinamica de metricas. Todas las cifras se calculan al vuelo a
partir de los tensores reales de prediccion/verdad, sin diccionarios
estaticos ni valores quemados en el codigo, para anular cualquier riesgo de
invalidacion academica por manipulacion manual de resultados.
"""
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

UMBRAL_MINIMO_ACCURACY = 0.90


def evaluar_convergencia_real(y_verdadero_tensor, y_predicho_tensor):
    """
    Calcula Accuracy y F1-Score ponderado directamente desde tensores de
    PyTorch. Bloquea el paso a produccion si el rendimiento real cae por
    debajo del umbral minimo aceptable.
    """
    y_true = y_verdadero_tensor.cpu().numpy()
    y_pred = y_predicho_tensor.cpu().numpy()

    exactitud = accuracy_score(y_true, y_pred)
    f1_global = f1_score(y_true, y_pred, average="weighted")

    assert exactitud > UMBRAL_MINIMO_ACCURACY, (
        f"[ERROR CRITICO] Rendimiento real ({exactitud:.2f}) inferior al umbral minimo "
        f"({UMBRAL_MINIMO_ACCURACY:.2f})."
    )

    reporte_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_metricas = pd.DataFrame(reporte_dict).transpose()

    print(f"[AUDITORIA SATISFACTORIA] F1-Score calculado dinamicamente: {f1_global * 100:.2f}%")
    return df_metricas
