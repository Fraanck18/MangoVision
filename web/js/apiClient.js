/**
 * Unico punto de acceso al backend FastAPI (src/api/app.py). Sin esto el
 * dashboard queda desconectado del modelo real y muestra valores fijos.
 */
const API_BASE = window.location.hostname === "" || window.location.protocol === "file:"
  ? "http://localhost:8000"
  : `${window.location.protocol}//${window.location.hostname}:8000`;

async function consultarSalud() {
  const respuesta = await fetch(`${API_BASE}/health`);
  if (!respuesta.ok) throw new Error(`/health respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function diagnosticarImagen(archivo) {
  const formulario = new FormData();
  formulario.append("archivo", archivo);
  const respuesta = await fetch(`${API_BASE}/diagnostico`, { method: "POST", body: formulario });
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}));
    throw new Error(detalle.detail || `/diagnostico respondio HTTP ${respuesta.status}`);
  }
  return respuesta.json();
}

async function iniciarEntrenamiento() {
  const respuesta = await fetch(`${API_BASE}/entrenamiento/iniciar`, { method: "POST" });
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}));
    throw new Error(detalle.detail || `/entrenamiento/iniciar respondio HTTP ${respuesta.status}`);
  }
  return respuesta.json();
}

async function consultarEstadoEntrenamiento() {
  const respuesta = await fetch(`${API_BASE}/entrenamiento/estado`);
  if (!respuesta.ok) throw new Error(`/entrenamiento/estado respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function detenerEntrenamiento() {
  const respuesta = await fetch(`${API_BASE}/entrenamiento/detener`, { method: "POST" });
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}));
    throw new Error(detalle.detail || `/entrenamiento/detener respondio HTTP ${respuesta.status}`);
  }
  return respuesta.json();
}

async function consultarHistorialEntrenamiento() {
  const respuesta = await fetch(`${API_BASE}/entrenamiento/historial`);
  if (!respuesta.ok) throw new Error(`/entrenamiento/historial respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

function urlReporteCSV() {
  return `${API_BASE}/entrenamiento/reporte.csv`;
}

async function consultarEscala() {
  const respuesta = await fetch(`${API_BASE}/config/escala`);
  if (!respuesta.ok) throw new Error(`/config/escala respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function obtenerPreferencias() {
  const respuesta = await fetch(`${API_BASE}/config/preferencias`);
  if (!respuesta.ok) throw new Error(`/config/preferencias respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function guardarPreferencias(preferencias) {
  const respuesta = await fetch(`${API_BASE}/config/preferencias`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preferencias),
  });
  if (!respuesta.ok) throw new Error(`/config/preferencias (POST) respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function consultarEstadoHDFS() {
  const respuesta = await fetch(`${API_BASE}/hdfs/estado`);
  if (!respuesta.ok) throw new Error(`/hdfs/estado respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function analizarLotesHDFS() {
  const respuesta = await fetch(`${API_BASE}/hdfs/analizar`, { method: "POST" });
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}));
    throw new Error(detalle.detail || `/hdfs/analizar respondio HTTP ${respuesta.status}`);
  }
  return respuesta.json();
}

async function consultarMetricasVivo() {
  const respuesta = await fetch(`${API_BASE}/metricas/vivo`);
  if (!respuesta.ok) throw new Error(`/metricas/vivo respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function consultarLotesImagenes() {
  const respuesta = await fetch(`${API_BASE}/imagenes/lotes`);
  if (!respuesta.ok) throw new Error(`/imagenes/lotes respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}

async function consultarImagenesDeLote(nombreLote) {
  const respuesta = await fetch(`${API_BASE}/imagenes/lote/${encodeURIComponent(nombreLote)}`);
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}));
    throw new Error(detalle.detail || `/imagenes/lote respondio HTTP ${respuesta.status}`);
  }
  return respuesta.json();
}

/** Convierte la URL relativa que entrega el backend en una URL absoluta servible por <img>. */
function urlImagenLote(rutaRelativa) {
  return `${API_BASE}${rutaRelativa}`;
}

async function simularFalloSensor(lotes = 3) {
  const respuesta = await fetch(`${API_BASE}/simular/fallo-sensor?lotes=${lotes}`, { method: "POST" });
  if (!respuesta.ok) throw new Error(`/simular/fallo-sensor respondio HTTP ${respuesta.status}`);
  return respuesta.json();
}
