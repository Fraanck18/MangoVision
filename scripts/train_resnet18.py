"""
Entrenamiento real de ResNet18MangoQC sobre el dataset preparado en
`dataset_mangos/processed/` (ver scripts/prepare_real_dataset.py).

Corre en CPU (no se asume GPU disponible). Para rescatar la clase intermedia
`calidad_promedio` (que colapsaba a F1=0.00 al ser la firma termica mas sutil,
un punto medio entre alta_calidad y defectuoso) se aplican, en orden:

  1. Weighted Cross Entropy: peso inverso a la frecuencia de cada clase, para
     que el gradiente penalice mas fallar en la clase minoritaria/ignorada.
  2. Oversampling dirigido (WeightedRandomSampler): cada batch se arma
     muestreando con mayor probabilidad la clase debil, garantizando
     representacion equitativa por lote.
  3. Discriminative learning rates + descongelado adicional: se descongela un
     bloque convolucional mas (capas_descongeladas=3) y el backbone aprende a
     una tasa mucho menor (1e-5) que la cabeza de clasificacion (1e-3), para
     afinar features sin destruir lo aprendido en ImageNet.

Uso:
    python scripts/train_resnet18.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.models.resnet18_custom import (
    ResNet18MangoQC,
    obtener_transformaciones_entrenamiento,
    obtener_transformaciones_validacion,
)
from src.models.evaluator import evaluar_convergencia_real

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
PESOS_SALIDA = RAIZ / "dataset_mangos" / "resnet18_mango_qc_multiclase.pt"
ESTADO_ENTRENAMIENTO = RAIZ / "dataset_mangos" / "estado_entrenamiento.json"

EPOCAS = 12
TASA_APRENDIZAJE_CABEZA = 1e-3
TASA_APRENDIZAJE_BACKBONE = 1e-5
CAPAS_DESCONGELADAS = 3
TAMANO_LOTE = 16

# Correcciones de desbalance. Experimento controlado sobre el dataset
# corregido (solo v1i): con AMBAS activadas, accuracy 0.75 / macro-F1 0.66 y
# recall de calidad_promedio (verde) 0.14; desactivandolas, accuracy baja a
# 0.71 / macro-F1 0.57 y el verde colapsa a recall 0.00. Es decir, SI ayudan
# a la clase minoritaria, por eso se dejan activadas por defecto. El techo
# (~0.75) lo impone la escasez de datos: solo ~11 mangos fisicos en total
# (2 verdes en train -> 1 verde nuevo en test), no la funcion de perdida.
USAR_WEIGHTED_CE = True
USAR_OVERSAMPLING = True


def escribir_estado(datos):
    ESTADO_ENTRENAMIENTO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def construir_dataloaders():
    dataset_train = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_entrenamiento())
    dataset_test = ImageFolder(PROCESADOS / "test", transform=obtener_transformaciones_validacion())

    assert dataset_train.classes == CLASS_NAMES, (
        f"Orden de clases de ImageFolder ({dataset_train.classes}) no coincide con CLASS_NAMES ({CLASS_NAMES})"
    )

    if USAR_OVERSAMPLING:
        # Oversampling dirigido: peso por muestra = 1 / frecuencia_de_su_clase,
        # para que las clases minoritarias se muestreen con mayor probabilidad y
        # cada batch quede balanceado pese al desbalance del dataset en disco.
        conteo_por_clase = Counter(dataset_train.targets)
        peso_por_clase = {clase: 1.0 / conteo for clase, conteo in conteo_por_clase.items()}
        pesos_muestra = [peso_por_clase[etiqueta] for etiqueta in dataset_train.targets]
        sampler = WeightedRandomSampler(pesos_muestra, num_samples=len(pesos_muestra), replacement=True)
        loader_train = DataLoader(dataset_train, batch_size=TAMANO_LOTE, sampler=sampler)
    else:
        loader_train = DataLoader(dataset_train, batch_size=TAMANO_LOTE, shuffle=True)

    loader_test = DataLoader(dataset_test, batch_size=TAMANO_LOTE, shuffle=False)
    return loader_train, loader_test, dataset_train


def calcular_pesos_clase(dataset_train, device):
    """Peso inverso a la frecuencia (normalizado) para Weighted Cross Entropy."""
    conteo_por_clase = Counter(dataset_train.targets)
    total = len(dataset_train.targets)
    num_clases = len(CLASS_NAMES)
    pesos = [total / (num_clases * conteo_por_clase[i]) for i in range(num_clases)]
    return torch.tensor(pesos, dtype=torch.float, device=device)


def construir_optimizador(modelo):
    """
    Discriminative learning rates: el backbone descongelado se afina a 1e-5
    (cambios pequenos sobre features genericas preentrenadas) y la cabeza de
    clasificacion entrena a 1e-3 (debe aprender la tarea desde cero).
    """
    params_backbone = [p for p in modelo.backbone.parameters() if p.requires_grad]
    params_cabeza = [p for p in modelo.head.parameters() if p.requires_grad]
    return torch.optim.Adam([
        {"params": params_backbone, "lr": TASA_APRENDIZAJE_BACKBONE},
        {"params": params_cabeza, "lr": TASA_APRENDIZAJE_CABEZA},
    ])


def entrenar_una_epoca(modelo, loader, optimizador, funcion_perdida, device):
    modelo.train()
    perdida_acumulada = 0.0
    for imagenes, etiquetas in loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)

        optimizador.zero_grad()
        logits = modelo(imagenes)
        perdida = funcion_perdida(logits, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_acumulada += perdida.item() * imagenes.size(0)
    return perdida_acumulada / len(loader.dataset)


def evaluar_en_test(modelo, loader, device):
    modelo.eval()
    todas_predicciones, todas_etiquetas = [], []
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device)
            logits = modelo(imagenes)
            predicciones = torch.argmax(logits, dim=1).cpu()
            todas_predicciones.append(predicciones)
            todas_etiquetas.append(etiquetas)
    return torch.cat(todas_etiquetas), torch.cat(todas_predicciones)


def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    inicio = time.time()
    escribir_estado({
        "estado": "en_progreso",
        "epoca_actual": 0,
        "epocas_totales": EPOCAS,
        "historial": [],
        "inicio": inicio,
    })

    loader_train, loader_test, dataset_train = construir_dataloaders()
    print(f"Train: {len(loader_train.dataset)} imagenes | Test: {len(loader_test.dataset)} imagenes")

    modelo = ResNet18MangoQC(capas_descongeladas=CAPAS_DESCONGELADAS).to(device)
    optimizador = construir_optimizador(modelo)

    if USAR_WEIGHTED_CE:
        pesos_clase = calcular_pesos_clase(dataset_train, device)
        print(f"Pesos por clase (Weighted CE) {CLASS_NAMES}: {[round(p, 3) for p in pesos_clase.tolist()]}")
        funcion_perdida = torch.nn.CrossEntropyLoss(weight=pesos_clase)
    else:
        funcion_perdida = torch.nn.CrossEntropyLoss()
    print(f"Weighted CE: {USAR_WEIGHTED_CE} | Oversampling: {USAR_OVERSAMPLING}")

    historial = []
    for epoca in range(1, EPOCAS + 1):
        perdida_promedio = entrenar_una_epoca(modelo, loader_train, optimizador, funcion_perdida, device)
        print(f"Epoca {epoca}/{EPOCAS} - perdida_train: {perdida_promedio:.4f}")
        historial.append({"epoca": epoca, "perdida_train": round(perdida_promedio, 4)})
        escribir_estado({
            "estado": "en_progreso",
            "epoca_actual": epoca,
            "epocas_totales": EPOCAS,
            "historial": historial,
            "inicio": inicio,
        })

    y_true, y_pred = evaluar_en_test(modelo, loader_test, device)
    try:
        df_metricas = evaluar_convergencia_real(y_true, y_pred)
        print(df_metricas)
        reporte = df_metricas.to_dict()
    except AssertionError as error:
        print(f"\n{error}\n")
        from sklearn.metrics import classification_report
        reporte = classification_report(
            y_true.numpy(), y_pred.numpy(), target_names=CLASS_NAMES, zero_division=0, output_dict=True
        )
        print(classification_report(y_true.numpy(), y_pred.numpy(), target_names=CLASS_NAMES, zero_division=0))

    accuracy_test = float((y_true == y_pred).float().mean().item())

    torch.save(modelo.state_dict(), PESOS_SALIDA)
    print(f"Pesos guardados en: {PESOS_SALIDA}")

    escribir_estado({
        "estado": "completado",
        "epoca_actual": EPOCAS,
        "epocas_totales": EPOCAS,
        "historial": historial,
        "inicio": inicio,
        "fin": time.time(),
        "accuracy_test": round(accuracy_test, 4),
        "reporte_test": reporte,
        "pesos": str(PESOS_SALIDA),
    })


if __name__ == "__main__":
    main()
