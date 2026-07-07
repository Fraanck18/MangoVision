"""
Tests del LOGO RESTRINGIDO a procedencia humana (fallo #2: sin fuga por
auto-etiquetas).

Se verifica la invariante central: el grupo held-out (test) es SIEMPRE humano y su
test contiene SOLO imagenes humanas; los datos pseudo/aumentados/sinteticos pueden
expandir el train pero NUNCA ser el grupo evaluado, y las variantes aumentadas de
un mango held-out quedan fuera de su propio train (anti-fuga).
"""
import numpy as np

from scripts.honest_grouped_cv import procedencia_de, folds_logo_restringido


def test_procedencia_clasifica_por_nombre():
    assert procedencia_de("v1i_Mango_3_jpg.rf.abc.jpg") == "humana"
    assert procedencia_de("flir_pseudo_ThermalMango_2.jpg") == "pseudo"
    assert procedencia_de("phys_Mango_3_stem_a0.jpg") == "phys_aug"
    assert procedencia_de("gan_alta_calidad_0001.jpg") == "gan"
    assert procedencia_de("synthphys_defectuoso_0.jpg") == "synthetic_physics"
    assert procedencia_de("sint_alta_calidad_0003.jpg") == "synthetic_physics"


def _escenario():
    # 2 mangos humanos (M1, M2) con 2 fotos cada uno + expansion no humana.
    grupos = np.array([
        "M1", "M1", "M2", "M2",          # humanas
        "M1", "M2",                      # phys_aug (comparten token de grupo del mango)
        "synthphys_defectuoso_0",        # sintetico
        "flir_pseudo_x",                 # pseudo (su propio grupo)
    ])
    procedencias = np.array([
        "humana", "humana", "humana", "humana",
        "phys_aug", "phys_aug",
        "synthetic_physics",
        "pseudo",
    ])
    return grupos, procedencias


def test_solo_grupos_humanos_son_held_out():
    grupos, procedencias = _escenario()
    folds = list(folds_logo_restringido(grupos, procedencias))
    grupos_evaluados = {g for _, _, g in folds}
    assert grupos_evaluados == {"M1", "M2"}  # ni sintetico ni pseudo evaluados


def test_test_fold_es_solo_humano():
    grupos, procedencias = _escenario()
    for idx_train, idx_test, g in folds_logo_restringido(grupos, procedencias):
        assert all(procedencias[i] == "humana" for i in idx_test)
        assert all(grupos[i] == g for i in idx_test)


def test_variantes_aumentadas_del_held_out_no_entran_al_train():
    """Si M1 es held-out, su phys_aug (grupo 'M1') NO puede estar en el train (anti-fuga)."""
    grupos, procedencias = _escenario()
    for idx_train, idx_test, g in folds_logo_restringido(grupos, procedencias):
        assert all(grupos[i] != g for i in idx_train)


def test_expansion_no_humana_si_puede_entrenar_en_otros_folds():
    """El sintetico/pseudo (de otros grupos) SI expande el train del fold."""
    grupos, procedencias = _escenario()
    folds = list(folds_logo_restringido(grupos, procedencias))
    # En el fold que evalua M1, el train debe incluir muestras no humanas de otros grupos.
    idx_train, _, _ = next(f for f in folds if f[2] == "M1")
    procs_train = {procedencias[i] for i in idx_train}
    assert "synthetic_physics" in procs_train or "pseudo" in procs_train
