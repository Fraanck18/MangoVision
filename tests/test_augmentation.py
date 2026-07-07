"""
Tests de la libreria de DATA AUGMENTATION.

Cubren la propiedad CLAVE de todo el paquete: las tecnicas son TERMICAMENTE
SEGURAS (no reordenan el ranking frio<->caliente que es la etiqueta) y respetan
las formas/contratos esperados por el pipeline.
"""
import numpy as np
import torch
from PIL import Image

from src.config import IMAGE_SIZE, NUM_CLASSES
from src.augmentation.paleta_ironbow import mapa_a_ironbow, ironbow_a_itr
from src.augmentation.termica_fisica import AugmentacionTermicaFisica, perturbar_itr_fisico
from src.augmentation.regularizadores import (
    DeformacionElastica,
    AutoAugmentGeometrico,
    obtener_transformaciones_entrenamiento_avanzado,
    mixup_lote,
    perdida_mixup,
)
from src.augmentation.diffaugment import DiffAugment
from src.inference.tta import probabilidades_tta, _vistas_seguras
from src.streaming.agregacion_temporal import AgregadorTemporal, promediar_frames


def _rampa_ironbow(n=64):
    """Imagen Ironbow con un gradiente ITR conocido (frio arriba -> caliente abajo)."""
    itr = np.linspace(0, 1, n, dtype=np.float32)[:, None].repeat(n, axis=1)
    return itr, Image.fromarray(mapa_a_ironbow(itr), mode="RGB")


# --------------------------- Paleta Ironbow ---------------------------
def test_ironbow_roundtrip_preserva_el_orden_termico():
    """Decodificar una rampa Ironbow recupera el ITR con orden monotono (corr ~1)."""
    itr, img = _rampa_ironbow()
    itr_rec = ironbow_a_itr(np.asarray(img))
    corr = np.corrcoef(itr.ravel(), itr_rec.ravel())[0, 1]
    assert corr > 0.99


def test_ironbow_encode_esta_en_rango_uint8():
    itr = np.linspace(0, 1, 100, dtype=np.float32)[None, :]
    rgb = mapa_a_ironbow(itr)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (1, 100, 3)


# ------------------- Augmentation fisica (dominio ITR) -------------------
def test_perturbacion_fisica_preserva_rango_unitario():
    rng = np.random.default_rng(0)
    itr = rng.random((32, 32)).astype(np.float32)
    out = perturbar_itr_fisico(itr, rng)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape == itr.shape


def test_perturbacion_fisica_no_reordena_la_temperatura():
    """
    La perturbacion fisica es (aprox.) monotona: un ITR bajo sigue siendo mas frio
    que un ITR alto tras aumentar. Se mide con alta correlacion de Spearman/orden;
    aqui basta correlacion de Pearson alta sobre una rampa suave (la etiqueta se
    conserva porque no se invierte el ranking termico).
    """
    rng = np.random.default_rng(3)
    itr = np.linspace(0, 1, 64, dtype=np.float32)[:, None].repeat(64, 1)
    # sin ruido NETD para medir la monotonia de las transformaciones deterministas
    out = perturbar_itr_fisico(itr, rng, netd=0.0)
    assert np.corrcoef(itr.ravel(), out.ravel())[0, 1] > 0.95


def test_augmentacion_fisica_transform_devuelve_pil_rgb():
    _, img = _rampa_ironbow(IMAGE_SIZE)
    out = AugmentacionTermicaFisica(p=1.0, seed=1)(img)
    assert isinstance(out, Image.Image) and out.mode == "RGB"
    assert out.size == img.size


def test_augmentacion_fisica_p_cero_es_identidad():
    _, img = _rampa_ironbow()
    out = AugmentacionTermicaFisica(p=0.0, seed=1)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


# ------------------------- Regularizadores -------------------------
def test_deformacion_elastica_conserva_forma():
    _, img = _rampa_ironbow(IMAGE_SIZE)
    out = DeformacionElastica(p=1.0, seed=2)(img)
    assert out.size == img.size and out.mode == "RGB"


def test_autoaugment_geometrico_conserva_forma():
    _, img = _rampa_ironbow(IMAGE_SIZE)
    out = AutoAugmentGeometrico(seed=5)(img)
    assert out.size == img.size


def test_pipeline_avanzado_produce_tensor_normalizado():
    _, img = _rampa_ironbow(IMAGE_SIZE)
    tf = obtener_transformaciones_entrenamiento_avanzado()
    t = tf(img)
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_mixup_forma_y_lambda():
    x = torch.randn(8, 3, 16, 16)
    y = torch.randint(0, NUM_CLASSES, (8,))
    xm, y_a, y_b, lam = mixup_lote(x, y, alpha=0.2)
    assert xm.shape == x.shape
    assert 0.0 <= lam <= 1.0
    assert y_a.shape == y.shape and y_b.shape == y.shape


def test_mixup_alpha_cero_es_identidad():
    x = torch.randn(4, 3, 8, 8)
    y = torch.arange(4)
    xm, y_a, y_b, lam = mixup_lote(x, y, alpha=0.0)
    assert lam == 1.0
    assert torch.equal(xm, x)


def test_perdida_mixup_interpola_entre_ambas_etiquetas():
    crit = torch.nn.CrossEntropyLoss()
    logits = torch.randn(6, NUM_CLASSES)
    y_a = torch.randint(0, NUM_CLASSES, (6,))
    y_b = torch.randint(0, NUM_CLASSES, (6,))
    l = perdida_mixup(crit, logits, y_a, y_b, 0.3)
    esperado = 0.3 * crit(logits, y_a) + 0.7 * crit(logits, y_b)
    assert torch.allclose(l, esperado)


# --------------------------- DiffAugment ---------------------------
def test_diffaugment_conserva_forma():
    x = torch.randn(4, 3, 64, 64)
    out = DiffAugment(x, "translation,cutout")
    assert out.shape == x.shape


def test_diffaugment_policy_vacia_es_identidad():
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(DiffAugment(x, ""), x)


# ------------------------------ TTA ------------------------------
def test_tta_genera_cinco_vistas_de_igual_forma():
    x = torch.randn(2, 3, 16, 16)
    vistas = _vistas_seguras(x)
    assert len(vistas) == 5
    assert all(v.shape == x.shape for v in vistas)


def test_tta_promedia_probabilidades_validas():
    """El softmax promediado sigue siendo una distribucion (suma 1, no negativa)."""
    class ModeloBobo(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=[2, 3])  # (N, C) desde (N, C, H, W); C=NUM_CLASSES

    modelo = ModeloBobo().eval()
    x = torch.randn(3, NUM_CLASSES, 8, 8)
    probs = probabilidades_tta(modelo, x)
    assert probs.shape == (3, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (probs >= 0).all()


# ----------------------- Agregacion temporal -----------------------
def test_frame_averaging_reduce_ruido():
    base = np.full((8, 8, 3), 120, dtype=np.float32)
    rng = np.random.default_rng(0)
    frames = [np.clip(base + rng.normal(0, 30, base.shape), 0, 255).astype(np.uint8)
              for _ in range(16)]
    promedio = promediar_frames(frames).astype(np.float32)
    # El error PIXEL A PIXEL del promedio (~1/sqrt(N)) es menor que el de un frame
    # suelto. Medir la media global no sirve: promedia el ruido espacialmente.
    err_promedio = np.sqrt(np.mean((promedio - base) ** 2))
    err_frame = np.sqrt(np.mean((frames[0].astype(np.float32) - base) ** 2))
    assert err_promedio < err_frame


def test_agregador_temporal_voto_estabiliza_la_clase():
    """Un frame atipico no debe voltear la decision si la ventana vota mayoria."""
    agg = AgregadorTemporal(num_clases=NUM_CLASSES, ventana=5, modo="voto")
    # cuatro frames dicen clase 0, uno dice clase 2
    for _ in range(4):
        clase, conf, n = agg.actualizar("mango_1", [0.8, 0.1, 0.1])
    clase, conf, n = agg.actualizar("mango_1", [0.1, 0.1, 0.8])
    assert clase == 0  # la mayoria manda pese al frame atipico
    assert n == 5


def test_agregador_temporal_rechaza_dim_incorrecta():
    agg = AgregadorTemporal(num_clases=NUM_CLASSES, ventana=3, modo="media")
    try:
        agg.actualizar("m", [0.5, 0.5])  # faltan clases
        assert False, "deberia haber lanzado ValueError"
    except ValueError:
        pass
