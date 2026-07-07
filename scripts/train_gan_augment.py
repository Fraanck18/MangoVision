"""
Data Augmentation con GAN condicional (cGAN / DCGAN) para MangoVision.

[Motivo] El dato real es ESCASO y desbalanceado (11 mangos fisicos; clases
flojas: `defectuoso` y la intermedia `calidad_promedio` que colapsa). Este GAN
aprende la distribucion de las firmas termicas reales y GENERA imagenes nuevas
condicionadas por clase, para rellenar los campos flojos (frio / promedio /
caliente) y balancear el entrenamiento sin depender de mas capturas.

[CPU] Se entrena a 64x64 (rapido en CPU) y las muestras se re-escalan a 224 al
guardarlas. Las imagenes GAN van SOLO a `dataset_mangos/gan_augment/train/` ->
se usan en TRAIN, JAMAS en el Test real bloqueado (disciplina F2/F4).

[Honestidad] Un GAN sobre 11 mangos no fabrica identidades de fruta nuevas:
amplia la variedad intra-distribucion y equilibra las clases. El techo real
sigue acotado por lo representativos que sean los mangos de test.

Uso:
    python scripts/train_gan_augment.py
"""
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CLASS_NAMES, RANDOM_SEED
from src.augmentation.diffaugment import DiffAugment

PROCESADOS = RAIZ / "dataset_mangos" / "processed"
SALIDA_GAN = RAIZ / "dataset_mangos" / "gan_augment"

# [#8 DiffAugment] Augmentation diferenciable aplicada por igual a reales y falsas
# para estabilizar el GAN con datos escasos. SOLO geometrica (translation+cutout):
# 'color' se excluye porque romperia el pseudo-color termico. Configurable por env.
DIFFAUG_POLICY = os.getenv("GAN_DIFFAUG_POLICY", "translation,cutout")

IMG = 64
NZ = 100
NGF = 48
NDF = 48
NUM_CLASSES = len(CLASS_NAMES)

EPOCAS = int(os.getenv("GAN_EPOCHS", "160"))
TAMANO_LOTE = 32
LR = 2e-4
# Objetivo de muestras GENERADAS por clase (balance de campos flojos).
N_GENERAR_POR_CLASE = int(os.getenv("GAN_N_POR_CLASE", "160"))


class Generador(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.red = nn.Sequential(
            nn.ConvTranspose2d(NZ + NUM_CLASSES, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8), nn.ReLU(True),                       # 4x4
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4), nn.ReLU(True),                       # 8x8
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2), nn.ReLU(True),                       # 16x16
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF), nn.ReLU(True),                           # 32x32
            nn.ConvTranspose2d(NGF, 3, 4, 2, 1, bias=False), nn.Tanh(),   # 64x64
        )

    def forward(self, z, etiquetas):
        c = self.emb(etiquetas).unsqueeze(-1).unsqueeze(-1)
        return self.red(torch.cat([z, c], dim=1))


class Discriminador(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NUM_CLASSES, IMG * IMG)
        self.red = nn.Sequential(
            nn.Conv2d(3 + 1, NDF, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),      # 32
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2), nn.LeakyReLU(0.2, True),                          # 16
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4), nn.LeakyReLU(0.2, True),                          # 8
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8), nn.LeakyReLU(0.2, True),                          # 4
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
        )

    def forward(self, img, etiquetas):
        mapa = self.emb(etiquetas).view(-1, 1, IMG, IMG)
        return self.red(torch.cat([img, mapa], dim=1)).view(-1)


def _init_pesos(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # -> [-1, 1]
    ])
    dataset = ImageFolder(PROCESADOS / "train", transform=tf)
    assert dataset.classes == CLASS_NAMES, f"{dataset.classes} != {CLASS_NAMES}"
    print(f"GAN condicional | {len(dataset)} imagenes reales | clases {Counter(dataset.targets)}")
    print(f"DiffAugment policy (data-efficient): {DIFFAUG_POLICY!r}")
    loader = DataLoader(dataset, batch_size=TAMANO_LOTE, shuffle=True, drop_last=True)

    G = Generador().to(device).apply(_init_pesos)
    D = Discriminador().to(device).apply(_init_pesos)
    criterio = nn.BCELoss()
    optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoca in range(1, EPOCAS + 1):
        perdidaG = perdidaD = 0.0
        for reales, etiquetas in loader:
            reales, etiquetas = reales.to(device), etiquetas.to(device)
            bs = reales.size(0)
            reales_lbl = torch.full((bs,), 0.9, device=device)   # label smoothing
            falsas_lbl = torch.zeros(bs, device=device)

            # --- Discriminador ---
            # DiffAugment: la MISMA augmentation diferenciable se aplica a la
            # imagen real y a la falsa antes de pasar por D (data-efficient GAN).
            optD.zero_grad()
            out_real = D(DiffAugment(reales, DIFFAUG_POLICY), etiquetas)
            perdida_real = criterio(out_real, reales_lbl)
            z = torch.randn(bs, NZ, 1, 1, device=device)
            etq_fake = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            falsas = G(z, etq_fake)
            out_fake = D(DiffAugment(falsas.detach(), DIFFAUG_POLICY), etq_fake)
            perdida_falsa = criterio(out_fake, falsas_lbl)
            (perdida_real + perdida_falsa).backward()
            optD.step()

            # --- Generador ---
            # Tambien aumentada: el gradiente fluye por DiffAugment, asi el
            # generador no aprende a reproducir los artefactos de la augmentation.
            optG.zero_grad()
            out = D(DiffAugment(falsas, DIFFAUG_POLICY), etq_fake)
            perdida_g = criterio(out, torch.ones(bs, device=device))
            perdida_g.backward()
            optG.step()

            perdidaD += (perdida_real + perdida_falsa).item()
            perdidaG += perdida_g.item()

        if epoca % 20 == 0 or epoca == 1:
            print(f"  epoca {epoca}/{EPOCAS} - D: {perdidaD/len(loader):.3f} - G: {perdidaG/len(loader):.3f}")

    # --- Generacion balanceada por clase -> SOLO train ---
    G.eval()
    if SALIDA_GAN.exists():
        import shutil
        shutil.rmtree(SALIDA_GAN)
    total = 0
    with torch.no_grad():
        for clase_id, clase in enumerate(CLASS_NAMES):
            carpeta = SALIDA_GAN / "train" / clase
            carpeta.mkdir(parents=True, exist_ok=True)
            for i in range(N_GENERAR_POR_CLASE):
                z = torch.randn(1, NZ, 1, 1, device=device)
                etq = torch.tensor([clase_id], device=device)
                img = G(z, etq).squeeze(0).cpu()
                img = (img * 0.5 + 0.5).clamp(0, 1)               # -> [0,1]
                pil = transforms.ToPILImage()(img).resize((224, 224), Image.BICUBIC)
                pil.save(carpeta / f"gan_{clase}_{i:04d}.jpg", quality=92)
                total += 1
    print(f"\nGeneradas {total} imagenes GAN ({N_GENERAR_POR_CLASE}/clase) en: {SALIDA_GAN / 'train'}")
    print("Uso: SOLO para entrenamiento (nunca test). El head fast las incorpora automaticamente.")


if __name__ == "__main__":
    main()
