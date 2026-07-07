"""
Job BATCH: reporte agregado por lote/cargamento (al llegar el camion a destino).

Es el rol legitimo del batch en este proyecto: NO re-clasifica por "mas precision"
(seria redundante — el modelo pesado no supera al brillo trivial), sino que produce
el AGREGADO del envio a partir de lo que el STREAM ya verifico y clasifico:
  - cuantos mangos, % anomalos (detector PEDTE Y baseline trivial),
  - fraccion degradada, distribucion de clases,
  - tendencia de anomalia en el viaje -> alerta de cadena de frio.

Escribe `web/data/reporte_lotes.json` y lo imprime.

Uso:
    python scripts/reporte_lote_batch.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.streaming.reporte_lotes import calcular_reporte_lotes

SALIDA = RAIZ / "web" / "data" / "reporte_lotes.json"


def main():
    reporte = calcular_reporte_lotes()
    if reporte["estado"] == "sin_datos":
        print("Sin datos de stream todavia (arranca la API y deja que ingesten lotes).")
        return

    print(f"=== Reporte de cargamento (batch, {reporte['n_lotes']} lotes, "
          f"{reporte['total_imagenes']} imagenes) ===")
    for l in reporte["lotes"]:
        print(f"  {l['lote_ingesta']}: {l['n_imagenes']} img | "
              f"anomalia det {l['tasa_anomalia_detector_pct']}% / trivial {l['tasa_anomalia_trivial_pct']}% | "
              f"degradadas {l['fraccion_degradada_pct']}%")
    print(f"\nTendencia anomalia (trivial) en el viaje: Δ {reporte['delta_anomalia_viaje_pct']:+.1f} pp")
    if reporte["alerta_cadena_frio"]:
        print("⚠ ALERTA: la tasa de anomalia sube marcadamente en el viaje -> posible ruptura de cadena de frio.")
    print(f"\nNota: {reporte['nota_honestidad']}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en: {SALIDA}")


if __name__ == "__main__":
    main()
