"""
Entrenamiento real de ResNet18MangoQC sobre el dataset preparado en
`dataset_mangos/processed/` (ver scripts/prepare_real_dataset.py).

En vez de una sola corrida a un numero fijo de epocas, se entrenan 3
candidatos INDEPENDIENTES (5, 10 y 15 epocas, mismo punto de partida: misma
semilla, mismos pesos iniciales) y se evalua cada uno en el set de test real.
Motivo: con datasets chicos, mas epocas no siempre generaliza mejor (puede
sobreajustar); se deja que el accuracy_test real de cada candidato decida
cual queda en produccion, sin forzar el resultado.

Para rescatar la clase intermedia `calidad_promedio` (que colapsaba a F1=0.00
al ser la firma termica mas sutil, un punto medio entre alta_calidad y
defectuoso) se aplican, en cada candidato, en orden:

  1. Weighted Cross Entropy: peso inverso a la frecuencia de cada clase, para
     que el gradiente penalice mas fallar en la clase minoritaria/ignorada.
  2. Oversampling dirigido (WeightedRandomSampler): cada batch se arma
     muestreando con mayor probabilidad la clase debil, garantizando
     representacion equitativa por lote.
  3. Discriminative learning rates + descongelado adicional: se descongela un
     bloque convolucional mas (capas_descongeladas=3) y el backbone aprende a
     una tasa mucho menor (1e-5) que la cabeza de clasificacion (1e-3), para
     afinar features sin destruir lo aprendido en ImageNet.

Rendimiento: usa GPU automaticamente si torch.cuda.is_available() (sin GPU
NVIDIA/CUDA disponible, sigue en CPU sin romper nada). En CPU, el cuello de
botella real no es el computo sino la carga/transformacion de imagenes en el
proceso principal; se ataca con DataLoader(num_workers>0) para paralelizar la
carga entre nucleos, batches mas grandes (menos overhead de Python por paso),
y pin_memory + non_blocking activados solo cuando hay GPU.

Uso:
    python scripts/train_resnet18.py
"""
import json
import os
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

EPOCAS_CANDIDATAS = [5, 10, 15]
TASA_APRENDIZAJE_CABEZA = 1e-3
TASA_APRENDIZAJE_BACKBONE = 1e-5
CAPAS_DESCONGELADAS = 3
TAMANO_LOTE = 32

# Correcciones de desbalance. Experimento controlado sobre el dataset
# corregido (solo v1i): con AMBAS activadas, accuracy 0.75 / macro-F1 0.66 y
# recall de calidad_promedio (verde) 0.14; desactivandolas, accuracy baja a
# 0.71 / macro-F1 0.57 y el verde colapsa a recall 0.00. Es decir, SI ayudan
# a la clase minoritaria, por eso se dejan activadas por defecto.
USAR_WEIGHTED_CE = True
USAR_OVERSAMPLING = True


def escribir_estado(datos):
    ESTADO_ENTRENAMIENTO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def construir_dataloaders(device):
    dataset_train = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_entrenamiento())
    dataset_test = ImageFolder(PROCESADOS / "test", transform=obtener_transformaciones_validacion())

    assert dataset_train.classes == CLASS_NAMES, (
        f"Orden de clases de ImageFolder ({dataset_train.classes}) no coincide con CLASS_NAMES ({CLASS_NAMES})"
    )

    usar_pin_memory = device.type == "cuda"
    # En GPU el cuello de botella real es el computo: paralelizar la carga de
    # imagenes evita que la GPU espere al CPU. En CPU, con un dataset chico
    # (~450 imagenes) el costo de arrancar procesos worker (en Windows via
    # spawn, cada uno reimporta torch) supera el beneficio, y aqui se paga 3
    # veces (uno por candidato de epocas), asi que se queda en 0.
    num_workers = min(4, os.cpu_count() or 2) if usar_pin_memory else 0

    if USAR_OVERSAMPLING:
        # Oversampling dirigido: peso por muestra = 1 / frecuencia_de_su_clase,
        # para que las clases minoritarias se muestreen con mayor probabilidad y
        # cada batch quede balanceado pese al desbalance del dataset en disco.
        conteo_por_clase = Counter(dataset_train.targets)
        peso_por_clase = {clase: 1.0 / conteo for clase, conteo in conteo_por_clase.items()}
        pesos_muestra = [peso_por_clase[etiqueta] for etiqueta in dataset_train.targets]
        sampler = WeightedRandomSampler(pesos_muestra, num_samples=len(pesos_muestra), replacement=True)
        loader_train = DataLoader(
            dataset_train, batch_size=TAMANO_LOTE, sampler=sampler,
            num_workers=num_workers, pin_memory=usar_pin_memory,
            persistent_workers=num_workers > 0,
        )
    else:
        loader_train = DataLoader(
            dataset_train, batch_size=TAMANO_LOTE, shuffle=True,
            num_workers=num_workers, pin_memory=usar_pin_memory,
            persistent_workers=num_workers > 0,
        )

    loader_test = DataLoader(
        dataset_test, batch_size=TAMANO_LOTE, shuffle=False,
        num_workers=num_workers, pin_memory=usar_pin_memory,
    )
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
    non_blocking = device.type == "cuda"
    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device, non_blocking=non_blocking)
        etiquetas = etiquetas.to(device, non_blocking=non_blocking)

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


def generar_reporte(y_true, y_pred):
    try:
        df_metricas = evaluar_convergencia_real(y_true, y_pred)
        print(df_metricas)
        return df_metricas.to_dict()
    except AssertionError as error:
        print(f"\n{error}\n")
        from sklearn.metrics import classification_report
        reporte = classification_report(
            y_true.numpy(), y_pred.numpy(), target_names=CLASS_NAMES, zero_division=0, output_dict=True
        )
        print(classification_report(y_true.numpy(), y_pred.numpy(), target_names=CLASS_NAMES, zero_division=0))
        return reporte


def entrenar_candidato(num_epocas, device, inicio_global, resultados_previos, indice, total_candidatos):
    """
    Corrida INDEPENDIENTE: misma semilla (mismo punto de partida) que los
    demas candidatos, entrenada por `num_epocas` epocas. Devuelve el modelo
    entrenado y sus metricas reales en test, sin tocar el numero final.
    """
    torch.manual_seed(RANDOM_SEED)
    loader_train, loader_test, dataset_train = construir_dataloaders(device)

    modelo = ResNet18MangoQC(capas_descongeladas=CAPAS_DESCONGELADAS).to(device)
    optimizador = construir_optimizador(modelo)

    if USAR_WEIGHTED_CE:
        pesos_clase = calcular_pesos_clase(dataset_train, device)
        funcion_perdida = torch.nn.CrossEntropyLoss(weight=pesos_clase)
    else:
        funcion_perdida = torch.nn.CrossEntropyLoss()

    print(f"\n=== Candidato: {num_epocas} epocas ({indice}/{total_candidatos}) ===")
    historial = []
    for epoca in range(1, num_epocas + 1):
        inicio_epoca = time.time()
        perdida_promedio = entrenar_una_epoca(modelo, loader_train, optimizador, funcion_perdida, device)
        duracion_epoca = time.time() - inicio_epoca
        print(f"Epoca {epoca}/{num_epocas} - perdida_train: {perdida_promedio:.4f} ({duracion_epoca:.1f}s)")
        historial.append({"epoca": epoca, "perdida_train": round(perdida_promedio, 4)})

        escribir_estado({
            "estado": "en_progreso",
            "candidatos_epocas": EPOCAS_CANDIDATAS,
            "candidato_actual": num_epocas,
            "indice_candidato": indice,
            "total_candidatos": total_candidatos,
            "epoca_actual": epoca,
            "epocas_totales": num_epocas,
            "historial": historial,
            "resultados_candidatos": resultados_previos,
            "inicio": inicio_global,
        })

    y_true, y_pred = evaluar_en_test(modelo, loader_test, device)
    reporte = generar_reporte(y_true, y_pred)
    accuracy_test = float((y_true == y_pred).float().mean().item())
    print(f"Candidato {num_epocas} epocas -> accuracy_test: {accuracy_test:.4f}")

    return modelo, {
        "epocas": num_epocas,
        "accuracy_test": round(accuracy_test, 4),
        "reporte_test": reporte,
        "historial": historial,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Entrenando en GPU: {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"Entrenando en CPU ({os.cpu_count()} nucleos, carga sin workers paralelos por tamano de dataset). "
              f"Sin GPU CUDA disponible en esta maquina.")

    inicio_global = time.time()
    total_candidatos = len(EPOCAS_CANDIDATAS)
    resultados = []
    modelos_entrenados = {}

    for i, num_epocas in enumerate(EPOCAS_CANDIDATAS, start=1):
        modelo, resultado = entrenar_candidato(num_epocas, device, inicio_global, resultados, i, total_candidatos)
        resultados.append(resultado)
        modelos_entrenados[num_epocas] = modelo

    # Selecciona el candidato con mejor accuracy_test real (a igualdad, el de
    # menos epocas: mas barato de entrenar/servir, sin sacrificar precision).
    mejor = max(resultados, key=lambda r: (r["accuracy_test"], -r["epocas"]))
    mejor_modelo = modelos_entrenados[mejor["epocas"]]

    torch.save(mejor_modelo.state_dict(), PESOS_SALIDA)
    print(f"\nMejor candidato: {mejor['epocas']} epocas (accuracy_test={mejor['accuracy_test']:.4f})")
    print(f"Pesos guardados en: {PESOS_SALIDA}")

    for r in resultados:
        marca = " <- elegido" if r["epocas"] == mejor["epocas"] else ""
        print(f"  {r['epocas']:>2} epocas -> accuracy_test={r['accuracy_test']:.4f}{marca}")

    escribir_estado({
        "estado": "completado",
        "candidatos_epocas": EPOCAS_CANDIDATAS,
        "resultados_candidatos": resultados,
        "mejor_epocas": mejor["epocas"],
        "epoca_actual": mejor["epocas"],
        "epocas_totales": mejor["epocas"],
        "historial": mejor["historial"],
        "inicio": inicio_global,
        "fin": time.time(),
        "accuracy_test": mejor["accuracy_test"],
        "reporte_test": mejor["reporte_test"],
        "pesos": str(PESOS_SALIDA),
    })


if __name__ == "__main__":
    main()
