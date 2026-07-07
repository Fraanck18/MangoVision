"""
Entrenamiento del clasificador termico de mango con EVALUACION HONESTA.

[Plan de Contingencia F4 — sin fuga de test]
Se separan TRES conjuntos independientes: Train / Validation / Test.
  * El Test REAL (dataset_mangos/processed/test) queda BLOQUEADO BAJO LLAVE: se
    evalua UNA sola vez, al final, y NUNCA se usa para elegir modelo/epoca.
  * La seleccion de candidato y el checkpointing (best_model.pth) se deciden
    EXCLUSIVAMENTE con la metrica de VALIDATION.
  * La Validation se separa del Train AGRUPANDO por mango fisico (GroupShuffleSplit)
    para que variantes aumentadas del mismo fruto no caigan en ambos lados (fuga).

[Plan de Contingencia F2 — sintetico solo pre-train]
Si existe `dataset_mangos/synthetic_pretrain/train`, se usa SOLO para pre-entrenar
la CNN (domain adaptation) antes del fine-tuning sobre el dato real. Ninguna
imagen sintetica entra a Validation ni a Test.

[Plan de Contingencia F7 — backbone ligero]
El backbone por defecto es ResNet18 (config. con MANGO_BACKBONE). Los pesos de
ResNet50 no son compatibles: el modelo se reentrena desde cero.

Flujo:  [Sintetico] -> pre-train  ==>  [Real 80% Train / Val] -> fine-tune + LOGO
        ==>  [Real 20% Test] -> evaluacion UNICA final.

Uso:
    python scripts/train_resnet50.py
"""
import copy
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.models.resnet50_custom import (
    ResNet50MangoQC,
    obtener_transformaciones_entrenamiento,   # [#1-A] delega en el pipeline avanzado (env-configurable)
    obtener_transformaciones_validacion,
    cargar_backbone_ssl_si_existe,
)
from src.augmentation.regularizadores import mixup_lote, perdida_mixup
from src.models.evaluator import evaluar_convergencia_real
from src.pipeline.historial_entrenamiento import agregar_a_historial, escribir_estado

# [#4] MixUp a nivel de lote. alpha=0 lo desactiva (MANGO_MIXUP_ALPHA).
MIXUP_ALPHA = float(os.getenv("MANGO_MIXUP_ALPHA", "0.2"))

PROCESADOS = RAIZ / "dataset_mangos" / "processed"                     # DATO REAL
SINTETICO_PRETRAIN = RAIZ / "dataset_mangos" / "synthetic_pretrain"    # solo pre-train (F2)
PESOS_SALIDA = RAIZ / "dataset_mangos" / "best_model.pth"

EPOCAS_CANDIDATAS = [5, 10, 15]
PRETRAIN_EPOCAS = int(os.getenv("MANGO_PRETRAIN_EPOCAS", "3"))
VAL_FRACCION = 0.2
TASA_APRENDIZAJE_CABEZA = 1e-3
TASA_APRENDIZAJE_BACKBONE = 1e-5
CAPAS_DESCONGELADAS = 3
TAMANO_LOTE = 32

USAR_WEIGHTED_CE = True
USAR_OVERSAMPLING = True

# Mismo patron de agrupacion por mango fisico que honest_grouped_cv.py, para que
# el split Train/Val no filtre variantes del mismo fruto.
PATRON_GRUPO = re.compile(r"(flir_pseudo_[^.]+|v\di_Mango_\d+|Mango_\d+)")


def _grupo_de(ruta):
    match = PATRON_GRUPO.search(Path(ruta).name)
    return match.group(1) if match else Path(ruta).name


def _num_workers(device):
    return min(4, os.cpu_count() or 2) if device.type == "cuda" else 0


def _sampler_balanceado(targets):
    """WeightedRandomSampler con peso inverso a la frecuencia de clase."""
    conteo = Counter(targets)
    peso_clase = {c: 1.0 / n for c, n in conteo.items()}
    pesos = [peso_clase[t] for t in targets]
    return WeightedRandomSampler(pesos, num_samples=len(pesos), replacement=True)


def separar_train_val(dataset_train):
    """
    [F4] Separa indices de Train y Validation AGRUPANDO por mango fisico. Si hay
    muy pocos grupos para un split agrupado, cae a un split aleatorio simple.
    """
    grupos = [_grupo_de(ruta) for ruta, _ in dataset_train.samples]
    indices = list(range(len(dataset_train)))

    if len(set(grupos)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACCION, random_state=RANDOM_SEED)
        idx_train, idx_val = next(gss.split(indices, groups=grupos))
        if len(idx_val) > 0 and len(idx_train) > 0:
            return list(idx_train), list(idx_val)

    # Fallback: split aleatorio (dataset con grupos insuficientes).
    generador = torch.Generator().manual_seed(RANDOM_SEED)
    perm = torch.randperm(len(indices), generator=generador).tolist()
    corte = max(1, int(len(perm) * VAL_FRACCION))
    return perm[corte:], perm[:corte]


def construir_dataloaders(device):
    """Devuelve loaders de Train / Val (real, agrupado) y Test (real, bloqueado)."""
    dataset_train_full = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_entrenamiento())
    dataset_val_full = ImageFolder(PROCESADOS / "train", transform=obtener_transformaciones_validacion())
    dataset_test = ImageFolder(PROCESADOS / "test", transform=obtener_transformaciones_validacion())

    assert dataset_train_full.classes == CLASS_NAMES, (
        f"Orden de clases ({dataset_train_full.classes}) != CLASS_NAMES ({CLASS_NAMES})"
    )

    idx_train, idx_val = separar_train_val(dataset_train_full)
    subset_train = Subset(dataset_train_full, idx_train)
    subset_val = Subset(dataset_val_full, idx_val)   # sin augmentation en validacion
    targets_train = [dataset_train_full.targets[i] for i in idx_train]

    usar_pin = device.type == "cuda"
    nw = _num_workers(device)

    if USAR_OVERSAMPLING:
        loader_train = DataLoader(
            subset_train, batch_size=TAMANO_LOTE, sampler=_sampler_balanceado(targets_train),
            num_workers=nw, pin_memory=usar_pin, persistent_workers=nw > 0,
        )
    else:
        loader_train = DataLoader(
            subset_train, batch_size=TAMANO_LOTE, shuffle=True,
            num_workers=nw, pin_memory=usar_pin, persistent_workers=nw > 0,
        )

    loader_val = DataLoader(subset_val, batch_size=TAMANO_LOTE, shuffle=False, num_workers=nw, pin_memory=usar_pin)
    loader_test = DataLoader(dataset_test, batch_size=TAMANO_LOTE, shuffle=False, num_workers=nw, pin_memory=usar_pin)
    return loader_train, loader_val, loader_test, targets_train


def construir_loader_pretrain(device):
    """[F2] Loader del pool sintetico de pre-entrenamiento, o None si no existe."""
    carpeta = SINTETICO_PRETRAIN / "train"
    if not carpeta.exists():
        return None
    dataset = ImageFolder(carpeta, transform=obtener_transformaciones_entrenamiento())
    nw = _num_workers(device)
    return DataLoader(dataset, batch_size=TAMANO_LOTE, shuffle=True, num_workers=nw,
                      pin_memory=device.type == "cuda", persistent_workers=nw > 0)


def calcular_pesos_clase(targets, device):
    """Peso inverso a la frecuencia (normalizado) para Weighted Cross Entropy."""
    conteo = Counter(targets)
    total = len(targets)
    num_clases = len(CLASS_NAMES)
    pesos = [total / (num_clases * conteo.get(i, 1)) for i in range(num_clases)]
    return torch.tensor(pesos, dtype=torch.float, device=device)


def construir_optimizador(modelo):
    """Discriminative LR: backbone 1e-5, cabeza 1e-3."""
    params_backbone = [p for p in modelo.backbone.parameters() if p.requires_grad]
    params_cabeza = [p for p in modelo.head.parameters() if p.requires_grad]
    grupos = []
    if params_backbone:
        grupos.append({"params": params_backbone, "lr": TASA_APRENDIZAJE_BACKBONE})
    grupos.append({"params": params_cabeza, "lr": TASA_APRENDIZAJE_CABEZA})
    return torch.optim.Adam(grupos)


def entrenar_una_epoca(modelo, loader, optimizador, funcion_perdida, device, mixup_alpha=0.0):
    modelo.train()
    perdida_acumulada = 0.0
    non_blocking = device.type == "cuda"
    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device, non_blocking=non_blocking)
        etiquetas = etiquetas.to(device, non_blocking=non_blocking)
        optimizador.zero_grad()
        if mixup_alpha > 0:
            # [#4] MixUp: mezcla lineal del lote y de sus etiquetas -> regulariza.
            imagenes, y_a, y_b, lam = mixup_lote(imagenes, etiquetas, alpha=mixup_alpha)
            logits = modelo(imagenes)
            perdida = perdida_mixup(funcion_perdida, logits, y_a, y_b, lam)
        else:
            logits = modelo(imagenes)
            perdida = funcion_perdida(logits, etiquetas)
        perdida.backward()
        optimizador.step()
        perdida_acumulada += perdida.item() * imagenes.size(0)
    return perdida_acumulada / len(loader.dataset)


def evaluar_en_loader(modelo, loader, device):
    modelo.eval()
    todas_predicciones, todas_etiquetas = [], []
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device)
            logits = modelo(imagenes)
            todas_predicciones.append(torch.argmax(logits, dim=1).cpu())
            todas_etiquetas.append(etiquetas)
    return torch.cat(todas_etiquetas), torch.cat(todas_predicciones)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).float().mean().item())


def generar_reporte(y_true, y_pred):
    # El gate relativo (evaluator.py) ya NO lanza AssertionError: imprime el
    # veredicto estructurado y devuelve el DataFrame de metricas. El Test real es
    # 100% humano (processed/test), asi que la metrica del gate ya es "solo real".
    df_metricas = evaluar_convergencia_real(y_true, y_pred)
    print(df_metricas)
    return df_metricas.to_dict()


def preentrenar_estado_base(device, loader_pretrain):
    """
    [F2] Pre-entrena una CNN sobre el pool SINTETICO (domain adaptation) y devuelve
    su state_dict, que servira de punto de partida comun a todos los candidatos.
    Devuelve None si no hay datos sinteticos.
    """
    torch.manual_seed(RANDOM_SEED)
    modelo = ResNet50MangoQC(capas_descongeladas=CAPAS_DESCONGELADAS).to(device)
    # [#2 SSL] Si hay backbone auto-supervisado (SimSiam sobre el stream no
    # etiquetado), arranca de ahi: mejor punto de partida que ImageNet puro.
    hay_ssl = cargar_backbone_ssl_si_existe(modelo)

    if loader_pretrain is None:
        if hay_ssl:
            print("Sin pool sintetico, pero se usa el backbone SSL como estado base.")
            return copy.deepcopy(modelo.state_dict())
        print("Sin pool sintetico ni SSL: se omite el pre-entrenamiento (F2).")
        return None

    print(f"\n=== Pre-entrenamiento sintetico (F2): {PRETRAIN_EPOCAS} epocas ===")
    optimizador = construir_optimizador(modelo)
    funcion_perdida = torch.nn.CrossEntropyLoss()
    for epoca in range(1, PRETRAIN_EPOCAS + 1):
        perdida = entrenar_una_epoca(modelo, loader_pretrain, optimizador, funcion_perdida, device)
        print(f"  Pre-train epoca {epoca}/{PRETRAIN_EPOCAS} - perdida: {perdida:.4f}")
    return copy.deepcopy(modelo.state_dict())


def entrenar_candidato(num_epocas, device, estado_base, loaders, pesos_clase,
                       inicio_global, resultados_previos, indice, total):
    """
    [F4] Entrena por `num_epocas` y hace CHECKPOINT del mejor estado segun VAL
    (nunca test). Devuelve el mejor estado y su accuracy de VALIDATION.
    """
    loader_train, loader_val, _loader_test, _ = loaders
    torch.manual_seed(RANDOM_SEED)
    modelo = ResNet50MangoQC(capas_descongeladas=CAPAS_DESCONGELADAS).to(device)
    if estado_base is not None:
        modelo.load_state_dict(estado_base)  # arranca del pre-entrenamiento sintetico

    optimizador = construir_optimizador(modelo)
    funcion_perdida = (torch.nn.CrossEntropyLoss(weight=pesos_clase)
                       if USAR_WEIGHTED_CE else torch.nn.CrossEntropyLoss())

    print(f"\n=== Candidato: {num_epocas} epocas ({indice}/{total}) ===")
    historial = []
    mejor_val_acc = -1.0
    mejor_estado = copy.deepcopy(modelo.state_dict())
    for epoca in range(1, num_epocas + 1):
        inicio_epoca = time.time()
        perdida = entrenar_una_epoca(modelo, loader_train, optimizador, funcion_perdida, device,
                                     mixup_alpha=MIXUP_ALPHA)
        y_val, p_val = evaluar_en_loader(modelo, loader_val, device)
        val_acc = accuracy(y_val, p_val)
        duracion = time.time() - inicio_epoca
        print(f"Epoca {epoca}/{num_epocas} - perdida_train: {perdida:.4f} - val_acc: {val_acc:.4f} ({duracion:.1f}s)")
        historial.append({"epoca": epoca, "perdida_train": round(perdida, 4), "val_acc": round(val_acc, 4)})

        # CHECKPOINT por VALIDATION (F4): el criterio es val, no test.
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            mejor_estado = copy.deepcopy(modelo.state_dict())

        escribir_estado({
            "estado": "en_progreso",
            "candidatos_epocas": EPOCAS_CANDIDATAS,
            "candidato_actual": num_epocas,
            "indice_candidato": indice,
            "total_candidatos": total,
            "epoca_actual": epoca,
            "epocas_totales": num_epocas,
            "historial": historial,
            "resultados_candidatos": resultados_previos,
            "inicio": inicio_global,
        })

    print(f"Candidato {num_epocas} epocas -> mejor val_acc: {mejor_val_acc:.4f}")
    return mejor_estado, {
        "epocas": num_epocas,
        "val_acc": round(mejor_val_acc, 4),
        "historial": historial,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Entrenando en GPU: {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"Entrenando en CPU ({os.cpu_count()} nucleos). Sin GPU CUDA disponible.")

    inicio_global = time.time()
    total = len(EPOCAS_CANDIDATAS)

    loaders = construir_dataloaders(device)
    loader_train, loader_val, loader_test, targets_train = loaders
    pesos_clase = calcular_pesos_clase(targets_train, device) if USAR_WEIGHTED_CE else None

    escribir_estado({
        "estado": "en_progreso", "candidatos_epocas": EPOCAS_CANDIDATAS,
        "candidato_actual": EPOCAS_CANDIDATAS[0], "indice_candidato": 1, "total_candidatos": total,
        "epoca_actual": 0, "epocas_totales": EPOCAS_CANDIDATAS[0], "historial": [],
        "resultados_candidatos": [], "inicio": inicio_global,
    })

    # [F2] Pre-entrenamiento sintetico (punto de partida comun).
    estado_base = preentrenar_estado_base(device, construir_loader_pretrain(device))

    # [F4] Cada candidato se selecciona por VALIDATION; el test no interviene aqui.
    resultados = []
    estados_candidatos = {}
    for i, num_epocas in enumerate(EPOCAS_CANDIDATAS, start=1):
        estado, resultado = entrenar_candidato(
            num_epocas, device, estado_base, loaders, pesos_clase,
            inicio_global, resultados, i, total,
        )
        resultados.append(resultado)
        estados_candidatos[num_epocas] = estado

    # Seleccion por VAL (a igualdad, menos epocas). El TEST no se usa para elegir.
    mejor = max(resultados, key=lambda r: (r["val_acc"], -r["epocas"]))
    mejor_estado = estados_candidatos[mejor["epocas"]]
    torch.save(mejor_estado, PESOS_SALIDA)
    print(f"\nMejor candidato por VALIDATION: {mejor['epocas']} epocas (val_acc={mejor['val_acc']:.4f})")
    for r in resultados:
        marca = " <- elegido (val)" if r["epocas"] == mejor["epocas"] else ""
        print(f"  {r['epocas']:>2} epocas -> val_acc={r['val_acc']:.4f}{marca}")

    # ===== EVALUACION UNICA FINAL SOBRE EL TEST REAL BLOQUEADO (F4) =====
    print("\n=== Evaluacion UNICA final sobre el Test REAL (bloqueado hasta aqui) ===")
    modelo_final = ResNet50MangoQC(capas_descongeladas=CAPAS_DESCONGELADAS).to(device)
    modelo_final.load_state_dict(mejor_estado)
    y_test, p_test = evaluar_en_loader(modelo_final, loader_test, device)
    reporte_test = generar_reporte(y_test, p_test)
    accuracy_test = accuracy(y_test, p_test)
    print(f"\nLinea Base Real Independiente -> accuracy_test: {accuracy_test:.4f}")

    resumen = {
        "estado": "completado",
        "candidatos_epocas": EPOCAS_CANDIDATAS,
        "resultados_candidatos": resultados,
        "mejor_epocas": mejor["epocas"],
        "seleccion": "por VALIDATION (test bloqueado, evaluado una sola vez)",
        "val_acc": mejor["val_acc"],
        "epoca_actual": mejor["epocas"],
        "epocas_totales": mejor["epocas"],
        "historial": mejor["historial"],
        "inicio": inicio_global,
        "fin": time.time(),
        "accuracy_test": round(accuracy_test, 4),
        "reporte_test": reporte_test,
        "pesos": str(PESOS_SALIDA),
        "backbone": ResNet50MangoQC().arquitectura,
        "preentrenamiento_sintetico": estado_base is not None,
    }
    escribir_estado(resumen)
    agregar_a_historial(resumen)


if __name__ == "__main__":
    main()
