"""
Entrenamiento real de ResNet18MangoQC sobre el dataset preparado en
`dataset_mangos/processed/` (ver scripts/prepare_real_dataset.py).

Corre en CPU (no se asume GPU disponible). El backbone va congelado salvo
el ultimo bloque convolucional, por lo que el costo de entrenamiento se
concentra en pocos parametros entrenables.

Uso:
    python scripts/train_resnet18.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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

PROCESADOS = RAIZ / "dataset_mangos" / "processed_binary"
PESOS_SALIDA = RAIZ / "dataset_mangos" / "resnet18_mango_qc_binario.pt"

EPOCAS = 12
TASA_APRENDIZAJE = 1e-3
TAMANO_LOTE = 16


def construir_dataloaders():
    dataset_train = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_entrenamiento())
    dataset_test = ImageFolder(PROCESADOS / "test", transform=obtener_transformaciones_validacion())

    assert dataset_train.classes == CLASS_NAMES, (
        f"Orden de clases de ImageFolder ({dataset_train.classes}) no coincide con CLASS_NAMES ({CLASS_NAMES})"
    )

    loader_train = DataLoader(dataset_train, batch_size=TAMANO_LOTE, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=TAMANO_LOTE, shuffle=False)
    return loader_train, loader_test


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

    loader_train, loader_test = construir_dataloaders()
    print(f"Train: {len(loader_train.dataset)} imagenes | Test: {len(loader_test.dataset)} imagenes")

    modelo = ResNet18MangoQC().to(device)
    optimizador = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=TASA_APRENDIZAJE)
    funcion_perdida = torch.nn.CrossEntropyLoss()

    for epoca in range(1, EPOCAS + 1):
        perdida_promedio = entrenar_una_epoca(modelo, loader_train, optimizador, funcion_perdida, device)
        print(f"Epoca {epoca}/{EPOCAS} - perdida_train: {perdida_promedio:.4f}")

    y_true, y_pred = evaluar_en_test(modelo, loader_test, device)
    try:
        df_metricas = evaluar_convergencia_real(y_true, y_pred)
        print(df_metricas)
    except AssertionError as error:
        print(f"\n{error}\n")
        from sklearn.metrics import classification_report
        print(classification_report(y_true.numpy(), y_pred.numpy(), target_names=CLASS_NAMES, zero_division=0))

    torch.save(modelo.state_dict(), PESOS_SALIDA)
    print(f"Pesos guardados en: {PESOS_SALIDA}")


if __name__ == "__main__":
    main()
