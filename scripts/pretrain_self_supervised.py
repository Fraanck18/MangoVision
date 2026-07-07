"""
Tecnica #2 — Pre-entrenamiento AUTO-SUPERVISADO (SimSiam) sobre el stream NO
etiquetado.

Motivo: etiquetado hay casi nada (~11 mangos fisicos), pero el sistema Kafka +
HDFS entrega MILES de frames termicos por segundo SIN etiqueta. Ese rio no
etiquetado es el mayor activo oculto del proyecto. SimSiam aprende
representaciones utiles del dominio termico SIN etiquetas: por cada imagen genera
dos vistas aumentadas y entrena la red para que sus embeddings coincidan
(similitud coseno negativa + stop-gradient, sin negativos ni memory bank).

El backbone resultante se guarda y sirve de PUNTO DE PARTIDA para el fine-tuning
supervisado sobre el diminuto set real (reemplaza/complementa al pre-entrenamiento
sintetico). Los scripts de entrenamiento lo cargan si existe la variable de
entorno MANGO_SSL_BACKBONE apuntando al checkpoint.

[Seguridad termica] Las DOS vistas se generan con augmentation termicamente segura
(geometrica + fisica en dominio ITR), NUNCA con color jitter: en pseudo-color el
matiz es la señal, y romperlo haria que SimSiam aprenda invarianzas equivocadas.

Fuentes de frames no etiquetados (se toman recursivamente, sin mirar carpetas de
clase): por defecto la ingesta cruda del Data Lake simulado + el pool FLIR crudo.
Configurable con MANGO_SSL_DIRS (rutas separadas por ';').

Uso:
    python scripts/pretrain_self_supervised.py
    MANGO_SSL_EPOCAS=5 MANGO_SSL_MAX_IMGS=2000 python scripts/pretrain_self_supervised.py
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import RANDOM_SEED
from src.models.resnet50_custom import ResNet50MangoQC
from src.augmentation.regularizadores import obtener_transformaciones_entrenamiento_avanzado

# Fuentes de imagenes NO etiquetadas (rio del stream). Recursivo, ignora labels.
DIRS_SSL_DEFECTO = [
    RAIZ / "dataset_mangos" / "ingesta" / "hdfs_simulado" / "raw",
    RAIZ / "dataset_mangos" / "extracted" / "flir_one",
]
CHECKPOINT_SSL = RAIZ / "dataset_mangos" / "ssl_backbone.pth"

EPOCAS = int(os.getenv("MANGO_SSL_EPOCAS", "5"))
TAMANO_LOTE = int(os.getenv("MANGO_SSL_LOTE", "64"))
MAX_IMGS = int(os.getenv("MANGO_SSL_MAX_IMGS", "3000"))   # tope para correr en CPU
LR = 1e-3
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _dirs_config():
    env = os.getenv("MANGO_SSL_DIRS")
    if env:
        return [Path(p) for p in env.split(";") if p.strip()]
    return DIRS_SSL_DEFECTO


class DatasetDosVistas(Dataset):
    """Devuelve DOS vistas aumentadas independientes de la misma imagen (SimSiam)."""

    def __init__(self, rutas, transform):
        self.rutas = rutas
        self.transform = transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        img = Image.open(self.rutas[i]).convert("RGB")
        return self.transform(img), self.transform(img)


def recolectar_rutas(dirs, limite):
    rutas = []
    for d in dirs:
        if not Path(d).exists():
            print(f"  [aviso] fuente inexistente, se omite: {d}")
            continue
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in EXTENSIONES:
                rutas.append(p)
                if len(rutas) >= limite:
                    return rutas
    return rutas


class ProyeccionSimSiam(nn.Module):
    """MLP de proyeccion (3 capas) con la ultima BN sin bias, estilo SimSiam."""

    def __init__(self, dim_in, dim=2048):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_in, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim, affine=False),
        )

    def forward(self, x):
        return self.red(x)


class PrediccionSimSiam(nn.Module):
    """MLP predictor (cuello de botella) de SimSiam."""

    def __init__(self, dim=2048, cuello=512):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim, cuello, bias=False), nn.BatchNorm1d(cuello), nn.ReLU(inplace=True),
            nn.Linear(cuello, dim),
        )

    def forward(self, x):
        return self.red(x)


class SimSiam(nn.Module):
    def __init__(self, backbone, dim_features, dim=2048):
        super().__init__()
        self.backbone = backbone
        self.proyeccion = ProyeccionSimSiam(dim_features, dim)
        self.prediccion = PrediccionSimSiam(dim)

    def _rama(self, x):
        z = self.proyeccion(self.backbone(x))
        p = self.prediccion(z)
        return p, z.detach()  # stop-gradient sobre z

    def forward(self, x1, x2):
        p1, z1 = self._rama(x1)
        p2, z2 = self._rama(x2)
        return p1, p2, z1, z2


def perdida_simsiam(p1, p2, z1, z2):
    """Similitud coseno negativa simetrica (D(p1,z2) + D(p2,z1)) / 2."""
    def D(p, z):
        return -F.cosine_similarity(p, z, dim=1).mean()
    return 0.5 * (D(p1, z2) + D(p2, z1))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    inicio = time.time()

    rutas = recolectar_rutas(_dirs_config(), MAX_IMGS)
    if len(rutas) < TAMANO_LOTE * 2:
        print(f"Muy pocas imagenes no etiquetadas ({len(rutas)}) para SSL util. "
              f"Añade fuentes con MANGO_SSL_DIRS. Abortando sin escribir checkpoint.")
        return
    print(f"SimSiam sobre {len(rutas)} frames NO etiquetados del stream "
          f"(fuentes: {[str(d) for d in _dirs_config()]}).")

    # Vistas termicamente seguras (geometrica + fisica; SIN color jitter).
    transform = obtener_transformaciones_entrenamiento_avanzado(p_fisica=0.7, p_elastica=0.5)
    dataset = DatasetDosVistas(rutas, transform)
    nw = min(4, os.cpu_count() or 2) if device.type == "cuda" else 0
    loader = DataLoader(dataset, batch_size=TAMANO_LOTE, shuffle=True, drop_last=True,
                        num_workers=nw, pin_memory=device.type == "cuda")

    # Backbone identico al del clasificador (resnet18 por defecto), sin congelar:
    # aqui SI queremos entrenar todo el extractor de forma auto-supervisada.
    plantilla = ResNet50MangoQC(congelar_backbone=False)
    dim_features = plantilla.head[0].in_features
    modelo = SimSiam(plantilla.backbone, dim_features).to(device)
    modelo.train()

    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    for epoca in range(1, EPOCAS + 1):
        perdida_acum = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizador.zero_grad()
            p1, p2, z1, z2 = modelo(v1, v2)
            perdida = perdida_simsiam(p1, p2, z1, z2)
            perdida.backward()
            optimizador.step()
            perdida_acum += perdida.item()
        promedio = perdida_acum / len(loader)
        # Rango sano de SimSiam: [-1, 0]; mas cerca de -1 = vistas mas alineadas.
        print(f"  epoca {epoca}/{EPOCAS} - perdida SimSiam: {promedio:.4f}")

    # Se guarda SOLO el state_dict del backbone -> cargable en ResNet50MangoQC.backbone.
    torch.save(modelo.backbone.state_dict(), CHECKPOINT_SSL)
    print(f"\nBackbone auto-supervisado guardado en: {CHECKPOINT_SSL}")
    print(f"Uso: MANGO_SSL_BACKBONE={CHECKPOINT_SSL} python scripts/train_head_fast.py")
    print(f"Listo en {time.time()-inicio:.1f}s (CPU).")


if __name__ == "__main__":
    main()
