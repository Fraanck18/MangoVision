# Correcciones de fallos críticos

Implementación del plan de solución de los 4 fallos críticos detectados en el
análisis. Cada uno se aborda como lo que es, y la mayoría se apoya en el toolkit de
data augmentation (ver [`DATA_AUGMENTATION.md`](DATA_AUGMENTATION.md)).

Principio de honestidad transversal: las cámaras FLIR de consumo **no** dan
radiometría absoluta (°C), dan pseudo-color. Todo el trabajo "físico" nuevo opera
en el **Índice de Intensidad Térmica Relativa (ITR ∈ [0,1])**, no en grados
inventados — coherente con F3 y con `src/domain_physics/thermal_inertia.py`.

---

## #4 — Trampa de pesos legacy (crash garantizado) → carga arquitectura-consciente

**Antes:** [`app.py`](../src/api/app.py) caía a `resnet50_mango_qc_multiclase.pt`
(que existe) y lo cargaba con `load_state_dict` estricto en un modelo cuyo backbone
por defecto es ResNet18 → crash en el arranque.

**Ahora:**
- Se **eliminó** el fallback legacy.
- `detectar_arquitectura_desde_state_dict` + `cargar_pesos_arquitectura_consciente`
  ([`resnet50_custom.py`](../src/models/resnet50_custom.py)) **detectan** el backbone
  desde las shapes del checkpoint (`head.0.weight` → 512=resnet18 / 2048=resnet50 /
  960=mobilenet) y construyen el modelo correcto. Carga defensiva (`strict=False` +
  validación de cobertura): un checkpoint incompatible se **rechaza con ValueError**
  claro (el startup lo captura y deja `modelo=None`, visible en `/health`), no
  revienta el proceso.
- El proyecto usa un **modelo único ResNet50** (`best_model.pth`, 2048 features); la
  carga arquitectura-consciente lo detecta y lo carga solo. La detección sigue
  soportando otras arquitecturas por si aparece un checkpoint antiguo, pero el
  entrenamiento nuevo es siempre ResNet50. También lo usan `pseudo_label_flir.py` y
  `cross_validate_pseudo_labels.py`.

## #3 — Spark no corría E2E → etapa ETL raw→parquet con augmentation

**Antes:** [`spark_inference.py`](../src/pipeline/spark_inference.py) leía
`raw_parquet/` pero **nadie lo escribía** → fallaba en la primera lectura.

**Ahora:** [`scripts/etl_raw_a_parquet_spark.py`](../scripts/etl_raw_a_parquet_spark.py)
materializa ese Parquet, y de paso es la augmentation offline:
- Fuente del lote nuevo: Kafka (topic `mango-etl-raw`) o carpeta local (demo).
  Partition pruning natural por `lote_ingesta`.
- Por cada imagen real emite (vía `mapInPandas` → `expandir_particion` en
  [`etl_augmentation.py`](../src/pipeline/etl_augmentation.py)): **1 original** +
  **K físicas** (label-preserving) + opcional **sintéticas auto-etiquetadas**.
- Escribe `raw_parquet/` con la imagen como columna `imagen_bytes` + `lote_ingesta`
  + **`grupo_fisico`** + **`provenance`** (`real` | `phys_aug` | `synthetic_physics`).
  El `grupo_fisico` viaja como **columna** (no solo en el nombre) → el anti-fuga
  sobrevive al empaquetado Parquet.
- `spark_inference.py` ahora lee la columna `imagen_bytes` (alineado con su propio
  docstring, que ya decía "imágenes empaquetadas como bytes en Parquet"; el código
  leía URIs — se corrigió esa incoherencia), con fallback a URI.

Efecto: `ejecutar_pipeline_batch()` corre, y el volumen que alimenta a Spark se
multiplica ×(K+1) con muestras físicamente válidas → palanca directa del #1.

```bash
spark-submit scripts/etl_raw_a_parquet_spark.py --fuente local --k-aug 3 --sinteticos-por-base 1
spark-submit src/pipeline/spark_inference.py            # ahora sí encuentra datos
```

## #2 — La "CV honesta" medía contra pseudo-etiquetas del modelo → LOGO restringido

**Antes:** `recolectar()` mezclaba train+test en un pool LOGO plano; los
`flir_pseudo_*` (etiquetas que el modelo se auto-asignó) caían como `y_true` cuando
su mango quedaba held-out → concordancia circular. Rompía la promesa del docstring
de `merge_pseudo_labels.py`.

**Ahora** ([`honest_grouped_cv.py`](../scripts/honest_grouped_cv.py)):
- `procedencia_de(nombre)` clasifica cada imagen: `humana | pseudo | phys_aug | gan
  | synthetic_physics`.
- **LOGO restringido** (`folds_logo_restringido`): el grupo held-out es **siempre
  humano** y su test contiene **solo** imágenes humanas; pseudo/aumentado/sintético
  pueden expandir el **train** de cualquier fold, nunca ser el grupo evaluado. Las
  variantes aumentadas de un mango held-out (que comparten su token de grupo) quedan
  fuera de su train (anti-fuga).
- Se reporta `n_grupos_humanos_evaluables` aparte de `n_grupos`.

Es la regla de oro de Noisy Student / weak supervision: se **entrena** con la
expansión, se **mide** contra verdad humana.

## #1 — Gate 0.90 incompatible con la línea base honesta

### Mitad A — Augmentation para subir el techo (la palanca)
El 0.72 no es un destino fijo. Currículo termo-seguro, medido siempre con la LOGO
honesta (ver [`DATA_AUGMENTATION.md`](DATA_AUGMENTATION.md) y
[`scripts/ablation_augmentation.py`](../scripts/ablation_augmentation.py)):
pre-entrenamiento SSL sobre FLIR sin etiqueta → pipeline avanzado (elástica +
geométrica + física + erasing, ahora **default** en `resnet50_custom.py`) → MixUp
en ITR contra el colapso de `calidad_promedio` → **ablation honesto** (cada técnica
se acepta solo si mueve la métrica sobre dato real, no la de train).

### Mitad B — Gate real, no teatro
**Antes:** `assert exactitud > 0.90` ([evaluator.py]) — inalcanzable, siempre
lanzaba, los scripts la atrapaban y descartaban.

**Ahora** ([`evaluator.py`](../src/models/evaluator.py)):
- Umbral **derivado de la línea base honesta documentada** en disco
  (`web/data/cv_honesta.json` → `accuracy_media`), no un número aspiracional:
  `umbral = baseline + margen`. `leer_baseline_honesta()` con override por env.
- `evaluar_gate(...)` devuelve un **veredicto estructurado**
  (`{aprobado, accuracy, umbral_usado, baseline, motivo, ...}`) en vez de lanzar
  una excepción que se traga. Ahora el gate realmente puede bloquear o aprobar.
- Se mide **solo sobre muestras reales** (`es_real`): se entrena con expansión, se
  mide contra verdad humana.
- `tests/test_metrics.py` reescrito: verifica que **bloquea** bajo `baseline+margen`
  y **aprueba** al superarlo, no que `UMBRAL == 0.90`.

Principio: un umbral por encima de lo alcanzable no es rigor, es una alarma
desconectada. La augmentation sube el listón; el gate hace que subirlo signifique
algo.

---

## Blindaje: datos sintéticos SIEMPRE fuera de la evaluación

El etiquetado programático por física ([`etiquetado_fisico.py`](../src/domain_physics/etiquetado_fisico.py))
genera matrices ITR **auto-etiquetadas por regla determinista** (no por un modelo):
`simular_estres_termico`→defectuoso, `simular_enfriamiento`→calidad_promedio,
`firma_madurez`→alta_calidad, sobre un substrato canónico. Se marcan
`provenance="synthetic_physics"` y con `grupo_fisico` sintético → **nunca**
colisionan con un mango real y quedan **estrictamente excluidas** del LOGO honesto y
del gate. Se usan para "anabolizar" el entrenamiento; la exactitud se mide
brutalmente honesta solo frente a datos reales.

## Tests

```bash
pytest tests/test_metrics.py tests/test_cv_honesta.py \
       tests/test_etiquetado_fisico.py tests/test_etl_augmentation.py -v
```
