"""
ResNet18 con Transfer Learning y cabecera personalizada para diagnostico
termico no destructivo de mango.

Se congelan los bloques convolucionales iniciales (extractor de bordes y
texturas de bajo nivel, genericos para cualquier dominio visual) y se
reentrenan unicamente las capas profundas finales junto con una cabecera de
clasificacion regularizada (BatchNorm1d + Dropout) para mitigar el overfitting
dado el tamano acotado del dataset (1,066 imagenes).
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


class ResNet18MangoQC(nn.Module):
    """ResNet18 preentrenada en ImageNet con cabecera personalizada de clasificacion."""

    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.5, congelar_backbone=True, capas_descongeladas=2):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if congelar_backbone:
            self._congelar_bloques_iniciales(capas_descongeladas)

        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def _congelar_bloques_iniciales(self, capas_descongeladas):
        """Congela conv1/bn1 y todos los layers salvo los ultimos N (fine-tuning parcial)."""
        bloques_entrenables = {f"layer{4 - i}" for i in range(capas_descongeladas)}
        for nombre, parametro in self.backbone.named_parameters():
            prefijo = nombre.split(".")[0]
            parametro.requires_grad = prefijo in bloques_entrenables

    def forward(self, x):
        caracteristicas = self.backbone(x)
        return self.head(caracteristicas)


def obtener_transformaciones_entrenamiento():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def obtener_transformaciones_validacion():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


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
    modelo = ResNet18MangoQC()
    entrada_simulada = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    salida = modelo(entrada_simulada)
    print(f"Forma de salida (batch, num_clases): {salida.shape}")
