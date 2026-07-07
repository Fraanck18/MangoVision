# Resultados medidos — MangoVision (ledger honesto)

Registro reproducible de **cada número** del proyecto: el comando que lo genera, qué
mide, y su **baseline trivial al lado**. Complementa el contrato
[`INTEGRIDAD.md`](INTEGRIDAD.md) (reglas) con el **ledger** (cifras). Regla:
*ningún número térmico se presenta sin su baseline trivial de brillo.*

Fecha de la última corrida: **2026-07-04**. Todo se reproduce con el `.venv` del repo.

---

## ✅ Resultado honesto presentable — Detección de anomalías

`python scripts/deteccion_anomalias.py` → `web/data/anomalias.json`

Enfoque: `normal = alta_calidad` (etiqueta HUMANA de calidad, criterio externo a la
imagen → no circular). LOGO por mango (4 mangos normales).

| Métrica (fruta REAL, held-out por mango) | Detector PEDTE | **Baseline trivial (brillo)** |
|---|---|---|
| **AUC** normal vs defectuoso | 0.645 | **0.874** |
| Falsas alarmas sobre normal held-out | 15.6 % | 7.5 % |
| Sensibilidad a sondas físicas *(sintéticas declaradas, no evaluación)* | 100 % | — |

**Lectura:** existe **señal térmica real y NO circular** — los `defectuoso` son
térmicamente **más brillantes** y un score de brillo los separa del normal a **AUC
0.87** (etiqueta humana). El detector PEDTE **no supera** al trivial → la señal es
simple y se reporta como tal. Es el **opuesto del 91.5 % tautológico**.

## ✅ Evaluación honesta LOGO (física + LogReg, binario)

`python scripts/honest_grouped_cv.py` → `web/data/cv_honesta.json`

LOGO restringido a procedencia humana, 11 mangos, `sano` vs `defectuoso`:
- **Accuracy media 60.9 % ± 32.7 %** · precision defectuoso 0.36 · **recall defectuoso 0.20** · F1 0.23.
- Honesto pero **modesto**: apenas supera la clase mayoritaria y se le escapa ~80 %
  del defecto. La alta σ refleja los pocos mangos.

## ❌ Control de tautología — Régimen térmico (NO es resultado)

`python scripts/train_termico.py --k 6 --epocas 80` → `web/data/thermal_cv.json`

| | Accuracy (mangos no vistos) |
|---|---|
| CNN "régimen térmico" | 91.5 % (recall [0.83, 0.93, 1.00]) |
| **Baseline trivial de brillo** | **90.6 %** |

**Diferencia +0.9 pp → tautología.** La etiqueta se generó por el nivel ITR y el
modelo solo lo relee. **No es evaluación de calidad de fruta.** Ver [`INTEGRIDAD.md`](INTEGRIDAD.md) §3.

## ➖ Ablación de data augmentation (no sube el techo honesto)

`python scripts/honest_cv_cnn.py --epocas 50`

LOGO del CNN, **promediado sobre 3 seeds**. Baseline (solo real) ≈ 74.6 % ± 37.7.
La augmentation física/sintética/MixUp **no produce ganancia reproducible** al
controlar la varianza de seed; solo estabiliza. Cuello de botella: **nº de mangos**,
no las técnicas.

## 🚫 Retirados (no son fruta real)
- **~96.7 %** (histórico): sobre dataset **sintético** — prueba pipeline + que la CNN
  aprende la física simulada, no exactitud de fruta.
- **100 %** antiguo: fuga de datos por mango (variantes del mismo fruto a ambos lados).

---

## Guardarraíl permanente
El baseline trivial de brillo es un **test** (`tests/test_anomaly.py`) y una columna
obligatoria en todo reporte. Implementación: `src/anomaly/baseline_trivial.py`.

## Pendiente para volver hipótesis en resultado validado
**Ancla real:** 5–10 mangos cortados y verificados para validar que la física
coincide con la realidad. Ver [`INTEGRIDAD.md`](INTEGRIDAD.md) §7.
