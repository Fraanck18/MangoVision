/**
 * FUENTE UNICA de la escala termica del dashboard (F8).
 *
 * El backend pivoto de °C a ITR relativo (Plan de Contingencia F3). Antes, el
 * frontend duplicaba constantes de escala (buckets °C, umbrales 21/32) y quedo
 * DESINCRONIZADO. Este modulo centraliza la escala en un solo lugar:
 *   - `ESCALA` refleja `src/config.py:MATURITY_INFO` (fallback para renderizar
 *     antes del fetch y si el backend no responde).
 *   - `/config/escala` (backend) es la verdad AUTORITATIVA; `cargarEscala()` la
 *     trae y re-renderiza.
 *
 * Un test (tests/test_frontend_escala.py) verifica que este fallback coincide con
 * config.py -> la escala no puede volver a desincronizarse en silencio.
 *
 * TODO aqui es ITR relativo 0-100 (NUNCA grados Celsius).
 */
const ESCALA = {
  unidad: "ITR (relativo 0-100, no °C)",
  rango: [0, 100],
  // Debe COINCIDIR con MATURITY_INFO en src/config.py (cubierto por test).
  clases: {
    calidad_promedio: { etiqueta: "Verde / Inmaduro", banda_itr: [15, 35], color: "#2f6fed" },
    alta_calidad:     { etiqueta: "Maduro / Óptimo",  banda_itr: [40, 70], color: "#16a34a" },
    defectuoso:       { etiqueta: "Sobre-Maduro",     banda_itr: [75, 100], color: "#dc2626" },
  },
};

// Buckets del histograma en ITR 0-100 (10 tramos). NO son °C.
const BUCKETS_ITR = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

/** Umbrales de color DERIVADOS de los huecos entre bandas (no hardcodeados aparte). */
function umbralesColorITR() {
  const p = ESCALA.clases.calidad_promedio.banda_itr;
  const a = ESCALA.clases.alta_calidad.banda_itr;
  const d = ESCALA.clases.defectuoso.banda_itr;
  return [(p[1] + a[0]) / 2, (a[1] + d[0]) / 2]; // p.ej. 37.5 y 72.5
}

/** Color de barra/punto segun el ITR (0-100), usando las bandas de la escala unica. */
function colorPorITR(itr) {
  const [u1, u2] = umbralesColorITR();
  if (itr < u1) return ESCALA.clases.calidad_promedio.color;
  if (itr < u2) return ESCALA.clases.alta_calidad.color;
  return ESCALA.clases.defectuoso.color;
}

/** Estado legible de una zona segun su ITR. */
function estadoZonaPorITR(itr) {
  const [u1, u2] = umbralesColorITR();
  if (itr < u1) return { texto: "Inmadura", clase: "frio" };
  if (itr < u2) return { texto: "Óptima", clase: "optimo" };
  return { texto: "Anómala", clase: "critico" };
}

/**
 * Semilla neutra por clase = punto medio de su banda ITR. Es un placeholder de
 * ESCALA (no un valor °C fabricado) para pintar algo antes del primer analisis.
 */
function semillaMediaPorClaseITR() {
  const s = {};
  for (const [clase, info] of Object.entries(ESCALA.clases)) {
    s[clase] = (info.banda_itr[0] + info.banda_itr[1]) / 2;
  }
  return s;
}

/**
 * Trae la escala AUTORITATIVA del backend (/config/escala) y, si difiere, la
 * aplica y re-renderiza vía `alActualizar`. Silencioso si el backend no responde
 * (se usa el fallback). Requiere `consultarEscala` (apiClient.js).
 */
async function cargarEscala(alActualizar) {
  try {
    const remota = await consultarEscala();
    if (remota && remota.clases) {
      ESCALA.unidad = remota.unidad || ESCALA.unidad;
      ESCALA.rango = remota.rango || ESCALA.rango;
      ESCALA.clases = remota.clases;
      if (typeof alActualizar === "function") alActualizar();
    }
  } catch {
    /* backend sin conexion: se conserva el fallback local */
  }
}
