"""
Backbone ResNet50 con Transfer Learning y cabecera personalizada para diagnostico
termico no destructivo de mango.

[Modelo unico — ResNet50] El proyecto usa un UNICO backbone: ResNet50
(IMAGENET1K_V2, ~23 M params, 2048 features). Se elimino la dualidad ResNet18/
ResNet50 para no arrastrar dos modelos distintos; el nombre de clase
`ResNet50MangoQC` queda alineado con el backbone real.

Nota (honestidad): esto revierte el F7 —que habia pivotado a ResNet18 para que la
alta capacidad no memorizara el ruido con un dataset diminuto—. La conclusion
honesta no cambia: la senal real la lleva el brillo termico (ver
`scripts/deteccion_anomalias.py`, `docs/RESULTADOS.md`), no la profundidad de la red.

`MANGO_BACKBONE` sigue existiendo para poder cargar checkpoints antiguos (la carga
es arquitectura-consciente), pero el DEFECTO y todo el entrenamiento nuevo es
`resnet50`.
"""
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

BACKBONE_DEFECTO = os.getenv("MANGO_BACKBONE", "resnet50")


class ResNet50MangoQC(nn.Module):
    """Backbone preentrenado en ImageNet + cabecera regularizada de clasificacion."""

    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.5, congelar_backbone=True,
                 capas_descongeladas=2, arquitectura=None):
        super().__init__()
        self.arquitectura = (arquitectura or BACKBONE_DEFECTO).lower()
        self.backbone, num_features = self._construir_backbone(self.arquitectura)

        if congelar_backbone:
            self._congelar_bloques_iniciales(capas_descongeladas)

        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def _construir_backbone(self, arquitectura):
        """Devuelve (backbone_sin_clasificador, dimension_de_features)."""
        if arquitectura == "resnet18":
            red = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = red.fc.in_features          # 512
            red.fc = nn.Identity()
        elif arquitectura == "resnet50":
            red = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            num_features = red.fc.in_features          # 2048
            red.fc = nn.Identity()
        elif arquitectura == "mobilenet_v3_large":
            red = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            num_features = red.classifier[0].in_features  # 960
            red.classifier = nn.Identity()
        else:
            raise ValueError(
                f"Backbone no soportado: {arquitectura!r}. "
                f"Usa 'resnet18', 'mobilenet_v3_large' o 'resnet50'."
            )
        return red, num_features

    def _congelar_bloques_iniciales(self, capas_descongeladas):
        """
        Fine-tuning parcial: congela el extractor de bajo nivel y deja entrenables
        solo los ultimos N bloques profundos.
        - ResNet: bloques layer1..layer4 (se descongelan los ultimos N).
        - MobileNetV3: sub-modulos de `features` (se descongelan los ultimos N).
        """
        if self.arquitectura.startswith("resnet"):
            bloques_entrenables = {f"layer{4 - i}" for i in range(capas_descongeladas)}
            for nombre, parametro in self.backbone.named_parameters():
                prefijo = nombre.split(".")[0]
                parametro.requires_grad = prefijo in bloques_entrenables
        else:  # mobilenet_v3_large
            for parametro in self.backbone.parameters():
                parametro.requires_grad = False
            bloques = list(self.backbone.features.children())
            for bloque in bloques[-capas_descongeladas:]:
                for parametro in bloque.parameters():
                    parametro.requires_grad = True

    def forward(self, x):
        caracteristicas = self.backbone(x)
        return self.head(caracteristicas)


# Nombre preferido; ResNet50MangoQC se mantiene como alias por compatibilidad.
MangoQCNet = ResNet50MangoQC


def obtener_transformaciones_entrenamiento():
    """
    Pipeline de TRAIN por defecto. [#1-A] Ahora delega en el pipeline AVANZADO
    termicamente seguro (elastica + geometrico + fisica en ITR + erasing), que
    reemplaza el flip+rot basico historico. Todo sigue siendo seguro para
    termografia (NUNCA ColorJitter: en Ironbow el matiz ES la señal).

    `MANGO_AUG_BASICO=1` fuerza el pipeline minimo historico (util para ablation
    "sin augmentation avanzada" contra la LOGO honesta).
    """
    if os.getenv("MANGO_AUG_BASICO", "0") == "1":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    # Import diferido para evitar un ciclo de imports (augmentation importa este modulo).
    from src.augmentation.regularizadores import obtener_transformaciones_entrenamiento_avanzado
    return obtener_transformaciones_entrenamiento_avanzado(
        usar_fisica=os.getenv("MANGO_AUG_FISICA", "1") == "1",
        usar_elastica=os.getenv("MANGO_AUG_ELASTICA", "1") == "1",
        usar_geometrico=os.getenv("MANGO_AUG_GEOMETRICO", "1") == "1",
        usar_erasing=os.getenv("MANGO_AUG_ERASING", "1") == "1",
    )


def obtener_transformaciones_validacion():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Dimension de features (in_features de la primera Linear de la cabeza) por
# arquitectura. Sirve para DETECTAR el backbone de un checkpoint sin asumirlo.
_FEATURES_POR_ARQUITECTURA = {512: "resnet18", 2048: "resnet50", 960: "mobilenet_v3_large"}


def detectar_arquitectura_desde_state_dict(state_dict):
    """
    Deduce la arquitectura del backbone a partir de las shapes del state_dict, en
    vez de ASUMIRLA (fuente del crash #4: cargar pesos ResNet50 en un ResNet18).

    La primera capa de la cabeza es `Linear(num_features, 256)`, asi que
    `head.0.weight` tiene shape (256, num_features). Se mapea num_features ->
    arquitectura. Devuelve el nombre o None si no se puede determinar.
    """
    peso = state_dict.get("head.0.weight")
    if peso is None:
        return None
    num_features = int(peso.shape[1])
    return _FEATURES_POR_ARQUITECTURA.get(num_features)


def cargar_pesos_arquitectura_consciente(ruta, device, num_classes=NUM_CLASSES, tolerancia=0.05):
    """
    Carga un checkpoint construyendo el backbone que REALMENTE le corresponde
    (detectado desde el propio state_dict), no el backbone por defecto del entorno.

    Carga DEFENSIVA: `strict=False` + validacion de cobertura. Si demasiadas claves
    faltan/sobran (mas que `tolerancia` del total), se considera un checkpoint
    incompatible y se lanza ValueError con un mensaje claro, en vez de devolver un
    modelo silenciosamente roto. Devuelve el modelo en `eval()` sobre `device`.
    """
    state_dict = torch.load(ruta, map_location=device)
    arquitectura = detectar_arquitectura_desde_state_dict(state_dict)
    if arquitectura is None:
        raise ValueError(
            f"No se pudo detectar la arquitectura del checkpoint {ruta} "
            f"(falta 'head.0.weight' o dimension de features desconocida)."
        )
    modelo = ResNet50MangoQC(num_classes=num_classes, arquitectura=arquitectura)
    faltantes, inesperadas = modelo.load_state_dict(state_dict, strict=False)
    total = len(modelo.state_dict())
    if (len(faltantes) + len(inesperadas)) > tolerancia * total:
        raise ValueError(
            f"Checkpoint {ruta} incompatible con backbone '{arquitectura}': "
            f"{len(faltantes)} claves faltantes, {len(inesperadas)} inesperadas "
            f"(> {tolerancia:.0%} de {total}). Se rechaza en vez de cargar un modelo roto."
        )
    return modelo.to(device).eval()


def cargar_backbone_ssl_si_existe(modelo, ruta=None, verbose=True):
    """
    [#2 SSL] Si existe un backbone pre-entrenado de forma AUTO-SUPERVISADA
    (SimSiam sobre el stream no etiquetado, ver scripts/pretrain_self_supervised.py),
    carga sus pesos en `modelo.backbone`. La ruta se toma del argumento o de la
    variable de entorno MANGO_SSL_BACKBONE. No-op silencioso si no hay checkpoint.

    Devuelve True si cargo pesos SSL, False en caso contrario.
    """
    ruta = ruta or os.getenv("MANGO_SSL_BACKBONE")
    if not ruta or not os.path.exists(ruta):
        return False
    estado = torch.load(ruta, map_location="cpu")
    faltantes, inesperadas = modelo.backbone.load_state_dict(estado, strict=False)
    if verbose:
        print(f"[SSL] Backbone auto-supervisado cargado desde {ruta} "
              f"(faltantes={len(faltantes)}, inesperadas={len(inesperadas)}).")
    return True


def cargar_modelo_mlflow_registry(nombre_modelo, stage="Production"):
    """Recupera la version etiquetada del modelo desde el Model Registry de MLflow."""
    import mlflow.pytorch

    uri = f"models:/{nombre_modelo}/{stage}"
    return mlflow.pytorch.load_model(uri)


def cargar_imagen_hdfs(uri_hdfs):
    """Lee una imagen termica desde HDFS y la entrega como PIL.Image en RGB."""
    import io
    from PIL import Image
    from pyarrow import fs

    sistema_archivos, ruta = fs.FileSystem.from_uri(uri_hdfs)
    with sistema_archivos.open_input_stream(ruta) as flujo:
        contenido = flujo.read()
    return Image.open(io.BytesIO(contenido)).convert("RGB")


if __name__ == "__main__":
    modelo = ResNet50MangoQC()
    n_params = sum(p.numel() for p in modelo.parameters())
    n_entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    entrada_simulada = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    salida = modelo(entrada_simulada)
    print(f"Backbone: {modelo.arquitectura} | params: {n_params/1e6:.1f}M "
          f"(entrenables: {n_entrenables/1e6:.1f}M)")
    print(f"Forma de salida (batch, num_clases): {salida.shape}")
