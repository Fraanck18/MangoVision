"""
Validacion cruzada agrupada por foto base para medir generalizacion real del
ResNet18 binario, en vez de confiar en un unico split fijo.

Motivo: con solo 23 fotos fisicas unicas, un solo train/test split (5 fotos
de test) puede dar 100% por suerte de cual foto cayo en test, no porque el
modelo generalice. StratifiedGroupKFold repite el experimento K veces con
distintos grupos de fotos en test cada vez, y se reporta el promedio +/-
desviacion estandar -- una estimacion mucho mas honesta y dificil de
inflar por un split favorable.

Solo usa las 682 imagenes reales (excluye pseudo-etiquetas de FLIR, que no
son ground truth verificado por humanos y no deben usarse para medir
desempeño).

Uso:
    python scripts/cross_validate_resnet18.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import RANDOM_SEED
from src.models.resnet18_custom import (
    ResNet18MangoQC,
    obtener_transformaciones_entrenamiento,
    obtener_transformaciones_validacion,
)

MANIFIESTO = RAIZ / "data" / "hdfs_manifest.json"
MAPA_BINARIO = {0: "sano", 1: "sano", 2: "defectuoso"}  # orden 3-clases original: alta/promedio/defectuoso
CLASES_BINARIAS = ["defectuoso", "sano"]  # alfabetico, igual que ImageFolder/config.py

NUM_FOLDS = 5
EPOCAS_POR_FOLD = 8
SALIDA_JSON = RAIZ / "web" / "data" / "cv_resnet18_binario.json"


class DatasetImagenesEnMemoria(Dataset):
    def __init__(self, rutas, etiquetas, transformaciones):
        self.rutas = rutas
        self.etiquetas = etiquetas
        self.transformaciones = transformaciones

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, indice):
        imagen = Image.open(self.rutas[indice]).convert("RGB")
        return self.transformaciones(imagen), self.etiquetas[indice]


def recolectar_imagenes_reales():
    """Lee el manifiesto y devuelve rutas + etiqueta binaria + foto_base_id, excluyendo pseudo-etiquetas."""
    with open(MANIFIESTO) as f:
        manifiesto = json.load(f)

    rutas, etiquetas_binarias, fotos_base = [], [], []
    for registro in manifiesto["registros"]:
        if registro.get("pseudo_etiquetado"):
            continue
        ruta = RAIZ / registro["ruta_local"]
        if not ruta.exists():
            continue
        clase_binaria = MAPA_BINARIO[registro["etiqueta"]]
        rutas.append(ruta)
        etiquetas_binarias.append(CLASES_BINARIAS.index(clase_binaria))
        fotos_base.append(registro["foto_base_id"])

    return np.array(rutas, dtype=object), np.array(etiquetas_binarias), np.array(fotos_base, dtype=object)


def entrenar_y_evaluar_fold(rutas, etiquetas, indices_train, indices_test, device):
    dataset_train = DatasetImagenesEnMemoria(
        rutas[indices_train], etiquetas[indices_train], obtener_transformaciones_entrenamiento()
    )
    dataset_test = DatasetImagenesEnMemoria(
        rutas[indices_test], etiquetas[indices_test], obtener_transformaciones_validacion()
    )
    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

    modelo = ResNet18MangoQC(num_classes=2).to(device)
    optimizador = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=1e-3)
    funcion_perdida = torch.nn.CrossEntropyLoss()

    modelo.train()
    for _ in range(EPOCAS_POR_FOLD):
        for imagenes, etiquetas_lote in loader_train:
            imagenes, etiquetas_lote = imagenes.to(device), etiquetas_lote.to(device)
            optimizador.zero_grad()
            perdida = funcion_perdida(modelo(imagenes), etiquetas_lote)
            perdida.backward()
            optimizador.step()

    modelo.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imagenes, etiquetas_lote in loader_test:
            logits = modelo(imagenes.to(device))
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            y_true.extend(etiquetas_lote.tolist())

    return np.array(y_true), np.array(y_pred)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)

    rutas, etiquetas, fotos_base = recolectar_imagenes_reales()
    print(f"Imagenes reales (sin pseudo-labels): {len(rutas)}")
    print(f"Fotos base unicas: {len(set(fotos_base))}")

    cv = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Etiqueta dominante por foto base, para que el agrupador estratifique por clase de foto
    etiqueta_por_foto = {}
    for foto, etiqueta in zip(fotos_base, etiquetas):
        etiqueta_por_foto.setdefault(foto, etiqueta)
    etiquetas_para_stratify = np.array([etiqueta_por_foto[f] for f in fotos_base])

    resultados_por_fold = []
    t0 = time.perf_counter()

    for num_fold, (indices_train, indices_test) in enumerate(
        cv.split(rutas, etiquetas_para_stratify, groups=fotos_base), start=1
    ):
        fotos_test_fold = sorted(set(fotos_base[indices_test]))
        print(f"\nFold {num_fold}/{NUM_FOLDS} -- fotos base en test: {fotos_test_fold}")

        y_true, y_pred = entrenar_y_evaluar_fold(rutas, etiquetas, indices_train, indices_test, device)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )

        print(f"  Accuracy fold {num_fold}: {accuracy:.3f}")
        resultados_por_fold.append({
            "fold": num_fold,
            "fotos_base_test": fotos_test_fold,
            "n_imagenes_test": len(y_true),
            "accuracy": float(accuracy),
            "precision_defectuoso": float(precision[0]),
            "recall_defectuoso": float(recall[0]),
            "f1_defectuoso": float(f1[0]),
            "precision_sano": float(precision[1]),
            "recall_sano": float(recall[1]),
            "f1_sano": float(f1[1]),
        })

    tiempo_total = time.perf_counter() - t0

    accuracies = [r["accuracy"] for r in resultados_por_fold]
    resumen = {
        "metodo": "StratifiedGroupKFold por foto_base_id, k=5, sin pseudo-labels",
        "num_folds": NUM_FOLDS,
        "epocas_por_fold": EPOCAS_POR_FOLD,
        "tiempo_total_segundos": round(tiempo_total, 1),
        "accuracy_promedio": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracy_min": float(np.min(accuracies)),
        "accuracy_max": float(np.max(accuracies)),
        "precision_defectuoso_promedio": float(np.mean([r["precision_defectuoso"] for r in resultados_por_fold])),
        "recall_defectuoso_promedio": float(np.mean([r["recall_defectuoso"] for r in resultados_por_fold])),
        "f1_defectuoso_promedio": float(np.mean([r["f1_defectuoso"] for r in resultados_por_fold])),
        "precision_sano_promedio": float(np.mean([r["precision_sano"] for r in resultados_por_fold])),
        "recall_sano_promedio": float(np.mean([r["recall_sano"] for r in resultados_por_fold])),
        "f1_sano_promedio": float(np.mean([r["f1_sano"] for r in resultados_por_fold])),
        "folds": resultados_por_fold,
    }

    print(f"\n=== RESULTADO HONESTO (promedio de {NUM_FOLDS} folds) ===")
    print(f"Accuracy: {resumen['accuracy_promedio']*100:.1f}% +/- {resumen['accuracy_std']*100:.1f}%")
    print(f"Rango: [{resumen['accuracy_min']*100:.1f}%, {resumen['accuracy_max']*100:.1f}%]")
    print(f"Tiempo total: {tiempo_total:.1f}s")

    SALIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_JSON, "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en: {SALIDA_JSON}")


if __name__ == "__main__":
    main()
