# Declaración de Integridad — MangoVision

Este documento es el contrato de honestidad del proyecto. Declara, sin adornos,
**qué dato es real y qué es sintético, qué mide cada número, contra qué baseline se
compara, y qué NO está demostrado.** Un evaluador debe poder auditar cualquier cifra
del proyecto leyendo esto.

> **Principio rector:** *La física te da las imágenes; la realidad te da la verdad.
> Puedes ENTRENAR con lo que generas desde la ley térmica, pero solo puedes
> EVALUARTE con dato real — aunque sea con un puñado de mangos.*

---

## 1. Alcance real del proyecto (qué es y qué NO es)

**Es:** *Clasificación de calidad del mango mediante análisis de imágenes térmicas*,
planteada como **detección de anomalías térmicas** — cuánto se desvía la firma
térmica de un fruto respecto de la población NORMAL.

**NO es:** un diagnóstico de enfermedad ni una detección de necrosis interna
específica. El sistema no identifica *qué* patología tiene un mango; señala que su
firma térmica **se aparta de lo normal**, y esa desviación es el indicador de
calidad. Cualquier redacción previa que prometa "detectar necrosis/desórdenes
metabólicos internos" es un **sobre-alcance** y queda corregida por este documento.

## 2. Procedencia del dato (real vs. generado)

| Procedencia | Qué es | Uso PERMITIDO | Uso PROHIBIDO |
|---|---|---|---|
| **real (humano)** | Roboflow v1i + FLIR verificadas | Entrenar **y** evaluar | — |
| **phys_aug** | perturbación física label-preserving de imagen real | Solo TRAIN | Evaluar |
| **synthetic_physics** | matriz generada por regla física, auto-etiquetada | Solo TRAIN / sonda | Evaluar |
| **temp (variantes de temperatura)** | recoloreo a otro régimen térmico | Solo TRAIN / sonda | **Evaluar / reportar como accuracy de fruta** |
| **pseudo** | etiqueta puesta por el propio modelo | Solo TRAIN | Evaluar |
| **gan** | muestra generada por cGAN | Solo TRAIN | Evaluar |

Regla de oro (implementada en `scripts/honest_grouped_cv.py`, LOGO restringido):
**el grupo evaluado siempre es de procedencia `real`; el resto solo expande el train.**

## 3. El control de tautología (obligatorio)

Todo resultado sobre imágenes térmicas debe reportarse **junto a un baseline
trivial**: un clasificador de **brillo medio** (una sola feature, sin CNN, sin
textura). Si el modelo no supera ese baseline de forma clara y reproducible, **no
aporta nada** y no debe presentarse como resultado.

> **Hallazgo medido (crítico):** el clasificador de *régimen térmico* entrenado con
> augmentation de temperatura da **91.5 %**, pero el **baseline trivial de brillo da
> 90.6 %** sobre los mismos mangos no vistos. El CNN aporta **+0.9 pp**. Conclusión:
> ese 91.5 % es una **tautología** — la etiqueta se generó por el nivel ITR y el
> modelo solo lo vuelve a leer. **No es evidencia de evaluación de calidad de fruta
> y no debe presentarse como tal.** (`scripts/entrenar_temperatura.py`, `train_termico.py`.)

## 4. Qué mide cada número del proyecto

| Número | Sobre qué dato | Qué prueba | Qué NO prueba |
|---|---|---|---|
| ~96.7 % (histórico, retirado) | sintético | pipeline + CNN aprende la física simulada | nada sobre fruta real |
| **91.5 % régimen térmico** | temp sintético | recupera una etiqueta de intensidad | **nada** (baseline trivial = 90.6 %) |
| **60.9 % LOGO binario** (`cv_honesta.json`) | real, held-out por mango | generalización honesta a mangos nuevos | recall defectuoso solo **0.20** |
| **AUC 0.87 (brillo trivial)** vs 0.65 (PEDTE) | real, LOGO por mango | los defectuosos son **térmicamente más brillantes** (señal real, no circular) | que el PEDTE/CNN aporte (no supera al trivial) |
| Sensibilidad a sondas físicas 100 % | sintético declarado | el detector reacciona a anomalías plausibles | nada sobre fruta real |

> **Resultado del detector de anomalías (`scripts/deteccion_anomalias.py`):** sobre
> fruta REAL, held-out por mango, un **baseline trivial de brillo separa defectuoso de
> normal a AUC 0.87** (etiqueta humana → no circular), y el **detector PEDTE one-class
> NO lo supera (0.65)**. Conclusión honesta: existe señal térmica real y simple (los
> defectuosos son más brillantes); la sofisticación (PEDTE/CNN) **no está justificada
> por el dato** a esta escala.

## 5. Qué NO está demostrado (y hay que decirlo)

- **No hay evidencia creíble** de que el sistema detecte calidad/defecto real de
  fruta. La única métrica honesta sobre el objetivo (LOGO binario) es **60.9 % con
  recall de defectuoso 0.20** — apenas supera predecir la clase mayoritaria y se le
  escapa el ~80 % de los defectos.
- La "ley térmica" que genera imágenes sintéticas **no ha sido validada contra un
  solo mango real cortado**. Mientras no exista ese ancla, la simulación es una
  **hipótesis no falsable**, no una verdad de evaluación.
- La narrativa de "inercia térmica" opera sobre **un único frame estático** (PEDTE,
  correlación espacial), no sobre la dinámica de enfriamiento (dT/dt).

## 6. Enfoque honesto recomendado (detección de anomalías)

Como los fallos son raros y la normalidad abundante, el planteamiento correcto es
**one-class / detección de anomalías**:

1. Entrenar el modelo de NORMALIDAD solo con mangos buenos **reales** (definidos
   por un criterio externo a la propia imagen: frescura / dentro de especificación).
2. Puntuar la desviación de cada fruto respecto de lo normal.
3. Métrica siempre honesta: **tasa de falsas alarmas sobre normal real held-out**
   (LOGO por mango). La sensibilidad se estima con las pocas anomalías reales + las
   sondas físicas **declaradas como sintéticas**.

La condición innegociable: el conjunto "normal" se define por algo **independiente
de la imagen térmica**; si no, se re-introduce la circularidad. En el repo, `normal =
alta_calidad` (etiqueta humana de calidad, criterio externo).

**Implementado:** `src/anomaly/` (detector one-class sobre PEDTE + baseline trivial) y
`scripts/deteccion_anomalias.py` (falsas alarmas LOGO + AUC + sondas físicas
declaradas). Guardarraíl del baseline trivial cubierto por `tests/test_anomaly.py`.

## 7. Requisito mínimo para volver "hipótesis" en "resultado"

**5–10 mangos reales cortados y verificados** como ancla de validación. Con ellos se
calibra que la ley física coincide con la realidad y se puede reportar un número
honesto. Sin ese ancla, ningún número térmico del proyecto es defendible como
evaluación de calidad de fruta.

---

## Checklist anti-fraude (antes de reportar cualquier cifra)

- [ ] ¿Se evaluó sobre dato **real**, held-out por **mango físico**?
- [ ] ¿Está el **baseline trivial de brillo** al lado, y el modelo lo supera?
- [ ] ¿Se declara qué es **sintético/sonda** y qué es **real**?
- [ ] ¿Se muestra el **balance de clases** y el **recall por clase** (no solo accuracy)?
- [ ] ¿La cifra alta NO proviene de etiquetas que el propio sistema generó?

Enlaces: [`DATA_AUGMENTATION.md`](DATA_AUGMENTATION.md) · [`FIXES_CRITICOS.md`](FIXES_CRITICOS.md)
