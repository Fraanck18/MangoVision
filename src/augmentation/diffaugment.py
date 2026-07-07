"""
Tecnica #8 — DiffAugment (Differentiable Augmentation for Data-Efficient GANs,
Zhao et al., NeurIPS 2020).

El cGAN de MangoVision se entrena sobre POQUISIMOS mangos fisicos: un DCGAN normal
sobreajusta el discriminador y colapsa (mode collapse). DiffAugment aplica la MISMA
augmentation DIFERENCIABLE tanto a las imagenes reales como a las falsas, en ambos
pasos (D y G). Como el gradiente fluye a traves de la augmentation, el generador
no aprende a reproducir los artefactos de la augmentation, y el discriminador deja
de memorizar el diminuto set real. Es el arreglo de MENOR esfuerzo para estabilizar
un GAN con datos limitados (~30 lineas, sin cambiar arquitectura).

[Seguridad termica] El policy por defecto es 'translation,cutout' (SOLO
geometrico). Se EXCLUYE 'color' a proposito: en pseudo-color Ironbow, perturbar
brillo/saturacion/contraste empujaria al generador a producir matices fuera de la
paleta termica (temperatura aparente falsa). El policy 'color' queda disponible por
completitud pero NO se recomienda para termografia.

Funciona sobre tensores (N, C, H, W) en rango [-1, 1] (como los del GAN).
"""
import torch
import torch.nn.functional as F


def DiffAugment(x, policy="translation,cutout"):
    """Aplica secuencialmente las politicas indicadas (separadas por coma)."""
    if not policy:
        return x
    for nombre in policy.split(","):
        nombre = nombre.strip()
        for func in FUNCIONES_AUG.get(nombre, []):
            x = func(x)
    return x.contiguous()


def rand_brightness(x):
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + factor


def rand_saturation(x):
    media = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - media) * factor + media


def rand_contrast(x):
    media = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - media) * factor + media


def rand_translation(x, ratio=0.125):
    desplazamiento_x = int(x.size(2) * ratio + 0.5)
    desplazamiento_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-desplazamiento_x, desplazamiento_x + 1, (x.size(0), 1, 1), device=x.device)
    ty = torch.randint(-desplazamiento_y, desplazamiento_y + 1, (x.size(0), 1, 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + tx + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()


def rand_cutout(x, ratio=0.5):
    tam_x = int(x.size(2) * ratio + 0.5)
    tam_y = int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - tam_x % 2), (x.size(0), 1, 1), device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - tam_y % 2), (x.size(0), 1, 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(tam_x, device=x.device),
        torch.arange(tam_y, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - tam_x // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - tam_y // 2, min=0, max=x.size(3) - 1)
    mascara = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mascara[grid_batch, grid_x, grid_y] = 0
    return x * mascara.unsqueeze(1)


FUNCIONES_AUG = {
    "color": [rand_brightness, rand_saturation, rand_contrast],  # NO para termografia
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
