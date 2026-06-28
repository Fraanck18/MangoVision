"""
Preparacion del dataset real de MangoVision a partir de los exports de
Roboflow (`v3i_multiclass`, `v1i_multiclass`) descargados en
`dataset_mangos/extracted/`.

Unifica las dos taxonomias de etiquetas reales en las 3 clases finales del
proyecto (alta_calidad / calidad_promedio / defectuoso) y reparte las
imagenes en train/test agrupando por FOTO BASE (no por archivo aumentado),
para evitar fuga de datos: todas las variantes aumentadas de una misma foto
quedan en el mismo lado del split.

Uso:
    python scripts/prepare_real_dataset.py
"""
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
EXTRAIDOS = RAIZ / "dataset_mangos" / "extracted"
PROCESADOS = RAIZ / "dataset_mangos" / "processed"
MANIFIESTO_SALIDA = RAIZ / "data" / "hdfs_manifest.json"

CLASES_FINALES = ["alta_calidad", "calidad_promedio", "defectuoso"]

# IMPORTANTE: solo se usa v1i_multiclass. La fuente v3i_multiclass se descarto
# a proposito: su taxonomia es de CALIDAD (fresh/good/spoil), un eje conceptual
# distinto al de MADUREZ (Raw/Ripened/defective) de v1i. Fusionar ambas
# contaminaba las clases (p.ej. "alta_calidad" mezclaba mangos frescos de v3i
# con mangos verdes inmaduros de v1i, visualmente opuestos), lo que hacia que
# el modelo no pudiera separar alta_calidad de calidad_promedio (F1=0.00).
#
# El mapeo de v1i respeta la semantica de MADUREZ del dashboard (MATURITY_INFO
# en src/config.py):
#   Raw (verde/inmaduro)  -> calidad_promedio -> "Verde / Inmaduro"
#   Ripened (maduro)      -> alta_calidad     -> "Maduro / Optimo"
#   defective (danado)    -> defectuoso       -> "Sobre-Maduro"
FUENTES = {
    "v1i": {
        "carpeta": EXTRAIDOS / "v1i_multiclass",
        "mapa_columnas": {"Raw": "calidad_promedio", "Ripened": "alta_calidad", "defective": "defectuoso"},
    },
}

PATRON_FOTO_BASE = re.compile(r"^(Mango_\d+)_jpg")


def leer_etiquetas_split(carpeta_split, mapa_columnas):
    """Lee un _classes.csv de Roboflow y devuelve {nombre_archivo: clase_final}."""
    ruta_csv = carpeta_split / "_classes.csv"
    etiquetas = {}
    if not ruta_csv.exists():
        return etiquetas

    with open(ruta_csv, newline="") as f:
        lector = csv.DictReader(f, skipinitialspace=True)
        for fila in lector:
            nombre_archivo = fila["filename"].strip()
            for columna_origen, clase_final in mapa_columnas.items():
                if fila.get(columna_origen, "0").strip() == "1":
                    etiquetas[nombre_archivo] = clase_final
                    break
    return etiquetas


def recolectar_registros():
    """Recorre cada fuente Roboflow y arma una lista de registros con foto_base_id, ruta, clase."""
    registros = []
    for nombre_fuente, info in FUENTES.items():
        carpeta = info["carpeta"]
        for nombre_split in ("train", "test", "valid"):
            carpeta_split = carpeta / nombre_split
            if not carpeta_split.exists():
                continue
            etiquetas = leer_etiquetas_split(carpeta_split, info["mapa_columnas"])
            for nombre_archivo, clase_final in etiquetas.items():
                ruta_imagen = carpeta_split / nombre_archivo
                if not ruta_imagen.exists():
                    continue
                match = PATRON_FOTO_BASE.match(nombre_archivo)
                foto_base = match.group(1) if match else nombre_archivo
                foto_base_id = f"{nombre_fuente}_{foto_base}"
                registros.append({
                    "ruta_origen": ruta_imagen,
                    "foto_base_id": foto_base_id,
                    "clase": clase_final,
                    "fuente": nombre_fuente,
                })
    return registros


def construir_split_sin_fuga(registros, test_size=0.2, random_state=42):
    """Asigna train/test a nivel de foto_base_id (estratificado por clase mayoritaria de cada foto)."""
    clases_por_foto = defaultdict(list)
    for registro in registros:
        clases_por_foto[registro["foto_base_id"]].append(registro["clase"])

    foto_ids = sorted(clases_por_foto.keys())
    clase_dominante_por_foto = [max(set(clases_por_foto[fid]), key=clases_por_foto[fid].count) for fid in foto_ids]

    fotos_train, fotos_test = train_test_split(
        foto_ids, test_size=test_size, random_state=random_state, stratify=clase_dominante_por_foto
    )
    fotos_train, fotos_test = set(fotos_train), set(fotos_test)

    for registro in registros:
        registro["particion"] = "train" if registro["foto_base_id"] in fotos_train else "test"
    return registros


def materializar_dataset(registros):
    """Copia las imagenes a dataset_mangos/processed/<particion>/<clase>/ y construye el manifiesto."""
    if PROCESADOS.exists():
        shutil.rmtree(PROCESADOS)

    manifiesto_registros = []
    for i, registro in enumerate(registros):
        carpeta_destino = PROCESADOS / registro["particion"] / registro["clase"]
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        nombre_destino = f"{registro['fuente']}_{registro['ruta_origen'].name}"
        ruta_destino = carpeta_destino / nombre_destino
        shutil.copyfile(registro["ruta_origen"], ruta_destino)

        manifiesto_registros.append({
            "id_muestra": f"mango_{i:05d}",
            "foto_base_id": registro["foto_base_id"],
            "fuente_original": registro["fuente"],
            "uri_hdfs": f"hdfs://nodo-maestro:9000/data-lake/mangos/raw/{registro['particion']}/{registro['clase']}/{nombre_destino}",
            "ruta_local": str(ruta_destino.relative_to(RAIZ)).replace("\\", "/"),
            "etiqueta": CLASES_FINALES.index(registro["clase"]),
            "particion": registro["particion"],
        })
    return manifiesto_registros


def escribir_manifiesto(manifiesto_registros):
    manifiesto = {
        "descripcion": (
            "Manifiesto de linaje del dataset real de MangoVision. Unifica los exports de "
            "Roboflow v3i_multiclass (fresh/good/spoil) y v1i_multiclass (Raw/Ripened/defective) "
            "en 3 clases finales. Split realizado por foto base (sin fuga entre train/test)."
        ),
        "version_dataset": "2.0.0-real",
        "total_imagenes": len(manifiesto_registros),
        "esquema_etiquetas": {str(i): clase for i, clase in enumerate(CLASES_FINALES)},
        "estrategia_particion": "train_test_split estratificado por foto base, test_size=0.2, random_state=42",
        "registros": manifiesto_registros,
    }
    MANIFIESTO_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFIESTO_SALIDA, "w") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)


def main():
    registros = recolectar_registros()
    print(f"Registros etiquetados encontrados: {len(registros)}")

    registros = construir_split_sin_fuga(registros)
    n_train = sum(1 for r in registros if r["particion"] == "train")
    n_test = sum(1 for r in registros if r["particion"] == "test")
    print(f"Train: {n_train} imagenes | Test: {n_test} imagenes")

    fotos_unicas = {r["foto_base_id"] for r in registros}
    print(f"Fotos base unicas (antes de augmentation): {len(fotos_unicas)}")

    manifiesto_registros = materializar_dataset(registros)
    escribir_manifiesto(manifiesto_registros)
    print(f"Manifiesto real escrito en: {MANIFIESTO_SALIDA}")
    print(f"Imagenes materializadas en: {PROCESADOS}")


if __name__ == "__main__":
    main()
