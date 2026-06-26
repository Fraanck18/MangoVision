"""
Demo end-to-end de la cadena de ingesta: sensor en camion -> buffer local ->
staging -> HDFS (simulado en disco local), tal como se diagramo.

Simula: el camion captura 3 lecturas SIN señal (se acumulan en el buffer),
luego recupera conexion en una parada y se drena todo a staging, y
finalmente se aterriza en el Data Lake particionado por lote_ingesta.

Uso:
    python scripts/demo_ingestion_cadena_frio.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.ingestion.edge_buffer import capturar_lectura, listar_pendientes
from src.ingestion.uploader_batch import drenar_buffer_a_staging, subir_staging_a_hdfs


def main():
    imagen_simulada = b"\xff\xd8\xff\xe0FAKEJPEGBYTES"

    print("Camion EN RUTA, sin señal: capturando 3 lecturas del sensor termico...")
    for i in range(3):
        id_lectura = capturar_lectura(imagen_simulada, {"sensor_id": "camion-07-sensor-termico"})
        print(f"  Lectura {i+1} guardada en buffer local: {id_lectura}")

    print(f"\nPendientes en buffer local: {len(listar_pendientes())}")

    print("\nIntentando subir SIN conexion (debe quedarse todo en el buffer)...")
    movidas = drenar_buffer_a_staging(verificador_conexion=lambda: False)
    print(f"  Movidas a staging: {len(movidas)} (esperado: 0)")

    print("\nCamion llega a una parada con señal: drenando buffer a staging...")
    movidas = drenar_buffer_a_staging(verificador_conexion=lambda: True)
    print(f"  Movidas a staging: {len(movidas)}")
    print(f"  Pendientes en buffer local tras drenar: {len(listar_pendientes())}")

    print("\nSubiendo de staging al Data Lake (HDFS simulado), particionado por lote_ingesta...")
    resultado = subir_staging_a_hdfs()
    print(f"  Lote de ingesta: {resultado['lote_ingesta']}")
    print(f"  Imagenes aterrizadas: {len(resultado['subidas'])}")
    print(f"  Carpeta destino: {resultado['carpeta_destino']}")


if __name__ == "__main__":
    main()
