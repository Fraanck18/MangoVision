# Data Augmentation avanzada — MangoVision

Estrategia de aumento de datos para un problema con **muy pocas imágenes térmicas
reales** (~11 mangos físicos, 221 frames de train) y una arquitectura de
**streaming HDFS + Kafka + Spark** que analiza muchos frames por segundo.

Todo el código vive en [`src/augmentation/`](../src/augmentation/), la parte de
inferencia en [`src/inference/tta.py`](../src/inference/tta.py) y
[`src/streaming/agregacion_temporal.py`](../src/streaming/agregacion_temporal.py).

---

## La regla que filtra TODO: es pseudo-color Ironbow, no RGB natural

En una imagen térmica Ironbow **el matiz ES la etiqueta** (temperatura relativa /
ITR — ver [`src/domain_physics/thermal_inertia.py`](../src/domain_physics/thermal_inertia.py)
y el README). Por eso quedan **prohibidas** todas las augmentations que perturban
color:

- ❌ `ColorJitter`, hue/saturation, `RandomGrayscale`, channel-shuffle.
- ❌ Las ops de color de `RandAugment`/`AutoAugment` (posterize, solarize, color,
  contrast, equalize).
- ❌ CutMix agresivo (crea bordes térmicos duros no físicos).

Cada técnica de esta librería es *térmicamente segura*: o es puramente geométrica,
o opera en el **dominio térmico (ITR)** y re-codifica con la paleta física, o
mezcla en el **espacio de la etiqueta** (MixUp) sin inventar matices.

---

## Mapa de técnicas

| # | Técnica | Ataca | Dónde vive | Código |
|---|---------|-------|-----------|--------|
| 1 | Augmentation física en dominio ITR | Overfitting, colapso clase intermedia | Offline (materializada) + online | [`termica_fisica.py`](../src/augmentation/termica_fisica.py) |
| 2 | Pre-entrenamiento auto-supervisado (SimSiam) | Escasez de etiquetas | Offline (usa stream sin etiquetar) | [`pretrain_self_supervised.py`](../scripts/pretrain_self_supervised.py) |
| 3 | Agregación temporal / multi-frame | Ruido por frame, parpadeo de clase | Streaming (Kafka/Spark) | [`agregacion_temporal.py`](../src/streaming/agregacion_temporal.py) |
| 4 | MixUp | Fronteras duras, dataset diminuto | Online (loop de train) | [`regularizadores.py`](../src/augmentation/regularizadores.py) |
| 5 | Random Erasing | Oclusiones / píxeles muertos | Online | [`regularizadores.py`](../src/augmentation/regularizadores.py) |
| 6 | Deformación elástica | Variación de forma del fruto | Online | [`regularizadores.py`](../src/augmentation/regularizadores.py) |
| 7 | Auto-augment geométrico | Búsqueda de política sin color | Online | [`regularizadores.py`](../src/augmentation/regularizadores.py) |
| 8 | DiffAugment (GAN data-efficient) | Mode collapse del cGAN | GAN training | [`diffaugment.py`](../src/augmentation/diffaugment.py) |
| 9 | Test-Time Augmentation (TTA) | Varianza por frame en inferencia | Inferencia (Spark) | [`tta.py`](../src/inference/tta.py) |

---

## 1. Augmentation física en el dominio térmico *(la más potente aquí)*

En vez de perturbar píxeles RGB al azar, se decodifica la paleta Ironbow a un mapa
ITR, se aplican perturbaciones que **un sensor FLIR real introduce sin cambiar la
clase del fruto**, y se re-codifica:

```
Ironbow RGB → decode → mapa ITR [0,1] → perturbar (física) → ITR' → re-encode → Ironbow'
```

Perturbaciones (todas monótonas en ITR → no reordenan frío↔caliente → preservan
la etiqueta):

- **Ganancia + offset** de ITR → deriva de temperatura ambiente / emisividad.
- **Gradiente radial centro→periferia** → coherente con el PEDTE (huella espacial
  de la inercia térmica).
- **Ruido NETD** (gaussiano fino) + **viñeteado** → ruido de detector y caída de
  respuesta hacia los bordes del FPA.
- **Gamma monótono** → respuesta no lineal del sensor.

Reutiliza la misma paleta Ironbow que el generador sintético
([`paleta_ironbow.py`](../src/augmentation/paleta_ironbow.py), fuente única de
verdad para encode/decode).

```bash
# Materializar offline -> dataset_mangos/physics_augment/train/<clase>/
python scripts/materializar_augmentation.py
MANGO_PHYS_POR_IMG=3 python scripts/materializar_augmentation.py
```

`train_head_fast.py` las suma automáticamente al train (`DIRS_AUMENTO`).

## 2. Pre-entrenamiento auto-supervisado (SimSiam) sobre el stream no etiquetado

El mayor activo oculto: Kafka entrega **miles de frames/seg sin etiqueta**. SimSiam
aprende representaciones del dominio térmico sin etiquetas (dos vistas aumentadas →
sus embeddings deben coincidir, similitud coseno negativa + stop-gradient). El
backbone resultante es un mejor punto de partida que ImageNet puro.

Las dos vistas se generan con augmentation **térmicamente segura** (geométrica +
física en dominio ITR), nunca con color jitter.

```bash
python scripts/pretrain_self_supervised.py                 # -> dataset_mangos/ssl_backbone.pth
MANGO_SSL_EPOCAS=5 MANGO_SSL_MAX_IMGS=2000 python scripts/pretrain_self_supervised.py

# Usar el backbone SSL en el fine-tuning supervisado:
MANGO_SSL_BACKBONE=dataset_mangos/ssl_backbone.pth python scripts/train_head_fast.py
```

Los scripts de entrenamiento lo cargan vía `cargar_backbone_ssl_si_existe`
([`resnet50_custom.py`](../src/models/resnet50_custom.py)).

## 3. Agregación temporal / multi-frame *(explota la arquitectura de streaming)*

Como el sensor da muchos frames/seg del mismo fruto:

- **Frame averaging** (`promediar_frames`): promediar N frames alineados reduce el
  ruido NETD ~1/√N antes de clasificar.
- **Voting temporal / EMA de logits** (`AgregadorTemporal`): agrega la predicción
  sobre una ventana deslizante por fruto (modo `ema`, `media` o `voto`) → decisión
  estable, sin parpadeos de clase.

Es post-proceso puro del stream (no reentrena). Punto de integración: la capa
Kafka/streaming ([`evaluador_online.py`](../src/streaming/evaluador_online.py)) o
entre el pipeline Spark y el resultado curado — requiere un `id_fruto` en el evento
para agrupar frames del mismo mango.

## 4–7. Regularizadores online seguros

Se aplican por época en el `DataLoader` de train, vía
`obtener_transformaciones_entrenamiento_avanzado()`, que **reemplaza** al pipeline
básico (solo flip+rotación) de `resnet50_custom.py`:

- **MixUp** (#4): mezcla lineal de dos firmas térmicas y sus etiquetas a nivel de
  lote (`mixup_lote` + `perdida_mixup`). Preferido sobre CutMix. Ataca el colapso
  de la clase intermedia. Activado en `train_resnet50.py` y (en espacio de
  features) en `train_head_fast.py`; `MANGO_MIXUP_ALPHA=0` lo desactiva.
- **Random Erasing** (#5): simula píxeles muertos / oclusiones del sensor.
- **Deformación elástica** (#6): warping de la forma orgánica del fruto.
- **Auto-augment geométrico** (#7): política tipo TrivialAugment **sin ninguna op
  de color**.

Flags de ablation en `obtener_transformaciones_entrenamiento_avanzado(usar_fisica=…,
usar_elastica=…, usar_geometrico=…, usar_erasing=…)`.

## 8. DiffAugment para el cGAN

El cGAN ([`train_gan_augment.py`](../scripts/train_gan_augment.py)) sobre ~11
mangos sufre **mode collapse**. DiffAugment aplica la misma augmentation
diferenciable a reales y falsas en los pasos de D y G → estabiliza el
entrenamiento con datos escasos, sin cambiar la arquitectura.

Policy por defecto **`translation,cutout`** (solo geométrico). Se **excluye
`color`** a propósito (rompería el pseudo-color). Configurable:

```bash
GAN_DIFFAUG_POLICY="translation,cutout" python scripts/train_gan_augment.py
```

## 9. Test-Time Augmentation (TTA)

En inferencia se promedia el softmax sobre vistas equivalentes de la misma imagen
(identidad + flips + rot90/rot270 — todas preservan la clase). Reduce la varianza
por frame. Barato en el pipeline Spark (ya va por lotes):

```bash
MANGO_TTA=1 spark-submit … src/pipeline/spark_inference.py
```

`diagnosticar_imagen_pil_tta` ofrece la variante por-frame para el evaluador online.

---

## Dónde vive cada cosa (online / offline / streaming)

| Tipo | Dónde | Por qué |
|------|-------|---------|
| Geométricas ligeras, MixUp, erasing | **Online** (DataLoader de train) | Baratas, distintas cada época |
| Física, elástica, GAN, sintético, SSL | **Offline** → materializar a disco/HDFS Parquet | Caras + reproducibles; evita el *small-file problem* de HDFS |
| TTA | **Spark `mapInPandas`** | Promedia predicciones por frame |
| Voting temporal / frame averaging | Capa **Kafka/streaming** | Agrega sobre la ventana de frames del mismo fruto |
| SSL | Consume el stream **no etiquetado**, escribe pesos | Convierte el río sin etiqueta en señal |

---

## Disciplina anti-fuga (F4) — crítico con un dataset diminuto

- **Nada sintético/GAN/físico entra a Validation ni a Test.** Solo se suman a
  train (`DIRS_AUMENTO`).
- **Grupo por mango físico**: las variantes físicas materializadas conservan en su
  nombre el token de grupo del mango de origen (`Mango_10`, `v3i_Mango_2`,
  `flir_pseudo_*`). `train_head_fast.features_de_aumento` **excluye** del train las
  variantes de cualquier mango que cayó en Validation → no hay fuga de identidad de
  fruto entre train y val.
- El **Test real sigue bloqueado** y se evalúa una sola vez.

## Honestidad

- Un GAN sobre 11 mangos **no fabrica identidades de fruta nuevas**: amplía la
  variedad intra-distribución y equilibra clases. El techo real sigue acotado por
  lo representativos que sean los mangos de test.
- La augmentation física recupera el **orden térmico relativo** (ITR), no
  temperatura absoluta (imposible desde pseudo-color, F3).
- SSL sobre pocos miles de frames del stream ayuda al arranque, no sustituye datos
  etiquetados representativos.

## 10. Etiquetado programático por física (Weak Supervision determinista)

El error del pseudo-etiquetado clásico es poner a la **red** a adivinar sus propias
etiquetas → concordancia circular (fallo #2). Alternativa honesta: las etiquetas las
fija una **función matemática determinista** sobre la firma térmica, no un modelo.

[`src/domain_physics/etiquetado_fisico.py`](../src/domain_physics/etiquetado_fisico.py):
- `etiquetar_por_firma_termica(itr)` — regla determinista sobre descriptores ITR
  (intensidad media, fracción caliente, gradiente centro-periferia) → clase.
- Motor de simulación paramétrica sobre un substrato canónico:
  `simular_estres_termico`→`defectuoso` (focos de calor + ganancia),
  `simular_enfriamiento`→`calidad_promedio` (homogeneiza y enfría),
  `firma_madurez`→`alta_calidad` (foco de disipación central). Cada transformación
  **auto-etiqueta** con la regla sobre el resultado — la etiqueta la pone la física,
  no la red.

**Honestidad (F3):** todo en **ITR relativo [0,1]**, no en °C (las FLIR de consumo
no dan radiometría absoluta). Las muestras se marcan `provenance="synthetic_physics"`
y quedan **excluidas** del LOGO honesto y del gate (ver
[`FIXES_CRITICOS.md`](FIXES_CRITICOS.md)).

## Materialización offline como ETL de Spark (raw→parquet)

[`scripts/etl_raw_a_parquet_spark.py`](../scripts/etl_raw_a_parquet_spark.py) es la
materialización offline de las técnicas #1 y #10 dentro del pipeline de Big Data:
lee el lote nuevo (Kafka o local), emite por cada imagen 1 original + K físicas
(+ sintéticas auto-etiquetadas opcionales), y escribe `raw_parquet/` con columnas
`imagen_bytes`, `lote_ingesta`, `grupo_fisico`, `provenance`. Con esto el pipeline
Spark (`spark_inference.py`) corre E2E y el volumen se multiplica ×(K+1).

```bash
spark-submit scripts/etl_raw_a_parquet_spark.py --fuente local --k-aug 3 --sinteticos-por-base 1
```

## 11. Augmentation por RÉGIMEN DE TEMPERATURA (expandir la diversidad térmica)

El dataset no son "solo 11 mangos": cada imagen es una **matriz térmica** que se
puede llevar a **otras temperaturas**. Como el proyecto clasifica por **régimen
térmico** (`calidad_promedio`=frío/verde, `alta_calidad`=medio/maduro,
`defectuoso`=caliente/sobre-maduro), una versión enfriada o calentada de una imagen
es una **muestra térmica nueva y válida**.

[`src/augmentation/temperatura.py`](../src/augmentation/temperatura.py):
- `desplazar_temperatura_itr(itr, delta, ganancia)` — filtro de nivel térmico en ITR.
- `variantes_temperatura(imagen, rng, n)` — barrido **regime-targeted** (frío
  homogéneo / medio / caliente con focos) que cubre las tres clases incluso desde
  bases reales muy estructuradas, **reetiquetando cada variante por la regla física
  determinista** (no por el modelo → sin circularidad).

Entrenamiento + evaluación:
[`scripts/entrenar_temperatura.py`](../scripts/entrenar_temperatura.py) genera K
variantes por imagen real, entrena la cabeza sobre `real+temp(train)` y evalúa —
**sin fuga de identidad de mango** (split por mango físico)— en dos conjuntos.

### Objetivo del sistema: leer el ESTADO TÉRMICO del fruto

Confirmado con el dominio: el sistema debe clasificar el **régimen térmico** que ve
el sensor infrarrojo en la faja (frío/medio/caliente), definido por la firma
térmica — consistente con las bandas de `MATURITY_INFO`. Esa es la tarea de
producción, y la augmentation de temperatura la optimiza directamente.

> [!danger] Este número es un CONTROL DE TAUTOLOGÍA, no un resultado
> [`scripts/train_termico.py`](../scripts/train_termico.py) entrena un clasificador
> de "régimen térmico" con variantes de temperatura reetiquetadas por física y
> alcanza **91.5 %** sobre mangos de test no vistos. **NO lo presentes como
> evaluación de calidad de fruta.** Un **baseline trivial de brillo medio** (una
> sola feature, sin CNN) da **90.6 %** sobre los mismos mangos: el CNN aporta
> **+0.9 pp**. La etiqueta se generó a partir del nivel ITR y el modelo solo lo
> vuelve a leer → **tautología**. Ver [`INTEGRIDAD.md`](INTEGRIDAD.md) §3.

**Uso honesto de esta maquinaria:** las variantes de temperatura sirven como
**sonda de sensibilidad declarada** (¿el detector de anomalías flagea estados
físicamente plausibles?) o como expansión label-preserving del manifold normal —
**nunca** como conjunto de evaluación ni como accuracy de fruta.

Reproducir el control (con su baseline trivial al lado):
`python scripts/train_termico.py --k 6 --epocas 80`.

## Resultados medidos (LOGO honesto del CNN) — sin adornos

[`scripts/honest_cv_cnn.py`](../scripts/honest_cv_cnn.py) evalúa el clasificador CNN
con LOGO por mango físico (cabeza sobre backbone congelado), **promediado sobre 3
seeds** para domar la varianza del head-training. Medido sobre los 11 mangos
humanos, solo contra verdad real:

| Config (seed-averaged) | acc LOGO | Δ | std entre mangos |
|---|---|---|---|
| baseline (solo real) | 74.6% | — | ±37.7 |
| +física (K=3) | 74.6% | +0.0 | ±35.4 |
| +sintético (en la cabeza) | 70.8% | −3.8 | ±32.2 |
| +sintético+MixUp | 71.2% | −3.4 | ±35.0 |

**Lectura honesta (crítica):**
- Con **solo 11 mangos**, la métrica LOGO tiene **~±36pp de varianza entre mangos**.
  Ningún delta de augmentation es estadísticamente distinguible del ruido: en runs
  de una sola seed aparecían "ganancias" de +4pp que **se evaporan al promediar
  seeds** (eran suerte de partición).
- Lo único reproducible: la **augmentation física** (label-preserving) es
  **neutra en media y reduce algo la varianza** (37.7→35.4) — segura de mantener.
- Volcar el **pool 100% sintético directamente en el train de la cabeza** tiende a
  **bajar la media** (−3.8): su brecha de dominio contamina el ajuste. El sintético
  rinde en su uso correcto —**pre-entrenamiento** y luego fine-tuning sobre real
  (flujo de `train_resnet50.py`)—, no dumpeado como features de la cabeza.
- **El cuello de botella real es el número de mangos físicos distintos, no el
  toolkit.** La augmentation no fabrica identidades de fruta nuevas (principio de
  honestidad ya documentado). Para subir la media de forma real hacen falta más
  mangos, o fine-tuning del backbone con augmentation online (que deja que la
  augmentation remodele las features, no solo añada puntos en un espacio fijo).

Reproducir: `python scripts/honest_cv_cnn.py --epocas 50` (imprime la tabla).

## Ablation honesto del currículo

[`scripts/ablation_augmentation.py`](../scripts/ablation_augmentation.py) entrena la
cabeza rápida bajo cada configuración de augmentation (flags `MANGO_AUG_*`,
`MANGO_MIXUP_ALPHA`, `MANGO_SSL_BACKBONE`) y compara la accuracy sobre validación
**agrupada por mango** (dato real no visto). Regla: conservar una técnica solo si
mueve la métrica real, no la de train. La métrica publicable final es el LOGO de
[`honest_grouped_cv.py`](../scripts/honest_grouped_cv.py).

## Tests

```bash
pytest tests/test_augmentation.py tests/test_etiquetado_fisico.py \
       tests/test_etl_augmentation.py tests/test_cv_honesta.py \
       tests/test_metrics.py -v
```

[`tests/test_augmentation.py`](../tests/test_augmentation.py) verifica la propiedad
clave (no se reordena el ranking térmico) y los contratos de forma de cada técnica;
los demás cubren el etiquetado físico determinista, el core del ETL, el LOGO
restringido a verdad humana y el gate relativo.
