/**
 * Orquestador del dashboard. Cada accion real (importar + analizar imagen,
 * iniciar entrenamiento) llama al backend FastAPI (ver apiClient.js) y
 * actualiza `estadoDashboard`, que es la unica fuente de verdad para todos
 * los renders (uiRenderer.js, mango3dRenderer.js, chartsRenderer.js).
 */
const estadoDashboard = {
  tempPromedio: 28.4,
  indiceMadurez: 74,
  varianza: 2.1,
  anomalias: 0,
  precisionModelo: 0,
  conteoClases: { calidad_promedio: 0, alta_calidad: 0, defectuoso: 0 },
  temperaturasMedias: { calidad_promedio: 18.2, alta_calidad: 28.4, defectuoso: 38.7 },
  totalMuestras: 0,
  tiempoProcMs: 0,
  zonas: { A: 27.1, B: 29.3, C: 32.8, D: 18.5, E: 38.2, F: 26.6 },
  metricaVivo: null,
};

let archivoSeleccionado = null;
let cicloPollingEntrenamiento = null;
let cicloPollingHDFS = null;
let cicloPollingMetrica = null;

document.addEventListener("DOMContentLoaded", () => {
  renderTodo();

  document.getElementById("btn-importar").addEventListener("click", () => abrirSelectorImagen());
  document.getElementById("btn-agregar-muestra").addEventListener("click", () => abrirSelectorImagen());
  document.getElementById("input-imagen").addEventListener("change", manejarSeleccionImagen);
  document.getElementById("btn-analizar").addEventListener("click", manejarClickAnalizar);
  document.getElementById("btn-entrenar").addEventListener("click", manejarClickEntrenar);
  document.getElementById("btn-exportar").addEventListener("click", exportarEstadoComoJSON);
  document.getElementById("btn-fallo-sensor").addEventListener("click", manejarFalloSensor);

  document.querySelectorAll("#selector-modo button").forEach((boton) => {
    boton.addEventListener("click", () => cambiarModoVisualizacion(boton.dataset.modo));
  });

  verificarConexionBackend();
  actualizarEstadoEntrenamiento();
  actualizarEstadoHDFS();
  actualizarMetricaVivo();
  cicloPollingEntrenamiento = setInterval(actualizarEstadoEntrenamiento, 4000);
  cicloPollingHDFS = setInterval(actualizarEstadoHDFS, 5000);
  cicloPollingMetrica = setInterval(actualizarMetricaVivo, 5000);
});

function renderTodo() {
  renderTarjetasStats(estadoDashboard);
  renderClasificacionPorLote(estadoDashboard);
  renderParametrosAnalisis(estadoDashboard);
  renderZonasTemperatura(estadoDashboard);
  renderMangos3D();
  renderGraficaDistribucion();
}

function abrirSelectorImagen() {
  document.getElementById("input-imagen").click();
}

function manejarSeleccionImagen(evento) {
  const archivo = evento.target.files[0];
  if (!archivo) return;
  archivoSeleccionado = archivo;
  document.getElementById("btn-analizar").disabled = false;
  document.getElementById("meta-sesion").textContent =
    `Sesión activa · Lote #MNG-2024-047 · Imagen lista: ${archivo.name}`;
  mostrarToast(`Imagen "${archivo.name}" importada. Pulsa Analizar.`);
}

/**
 * "Analizar" hace dos cosas en cada click, igual que un sistema en tiempo
 * real: (1) si hay una imagen importada manualmente, la diagnostica, y
 * (2) siempre revisa el Data Lake (HDFS simulado) y procesa EN PARALELO,
 * entre varios workers, todos los lotes que el sensor del camion haya
 * aterrizado desde el ultimo click (sea 5 min o 1 hora despues).
 */
async function manejarClickAnalizar() {
  const boton = document.getElementById("btn-analizar");
  boton.disabled = true;
  boton.textContent = "⏳ Analizando...";

  try {
    if (archivoSeleccionado) {
      await analizarImagenImportada();
    }
    await analizarLotesPendientesHDFS();
  } catch (error) {
    mostrarToast(`Error al analizar: ${error.message}`, true);
  } finally {
    boton.disabled = false;
    boton.textContent = "▶ Analizar";
  }
}

async function analizarImagenImportada() {
  const inicio = performance.now();
  const diagnostico = await diagnosticarImagen(archivoSeleccionado);
  const tiempoSeg = (performance.now() - inicio) / 1000;

  aplicarDiagnosticoAlEstado(diagnostico, tiempoSeg);
  actualizarClasificacionHolografica(diagnostico);
  acumularMuestraGrafica(diagnostico.temperatura_estimada);

  const urlImagen = URL.createObjectURL(archivoSeleccionado);
  renderMuestraReciente(diagnostico, urlImagen);
  archivoSeleccionado = null;

  mostrarToast(`Imagen importada clasificada como "${diagnostico.categoria}" (${(diagnostico.confianza * 100).toFixed(1)}%).`);
}

async function analizarLotesPendientesHDFS() {
  const resultado = await analizarLotesHDFS();

  if (resultado.lotes_procesados.length === 0) {
    renderTodo();
    mostrarToast("No hay lotes nuevos en el Data Lake todavía.");
    return;
  }

  resultado.resultados.forEach((diagnostico) => {
    aplicarDiagnosticoAlEstado(diagnostico, diagnostico.tiempo_inferencia_s);
    acumularMuestraGrafica(diagnostico.temperatura_estimada);
  });
  actualizarClasificacionHolografica(resultado.resultados[resultado.resultados.length - 1]);
  renderTodo();

  document.getElementById("meta-sesion").textContent =
    `Sesión activa · Lote #MNG-2024-047 · Última actualización hace unos segundos`;

  const speedup = resultado.tiempo_secuencial_estimado_s > 0
    ? (resultado.tiempo_secuencial_estimado_s / resultado.tiempo_total_s).toFixed(1)
    : "1.0";
  mostrarToast(
    `${resultado.lotes_procesados.length} lote(s) HDFS · ${resultado.resultados.length} imágenes · ` +
    `${resultado.num_workers} workers en paralelo · ${resultado.tiempo_total_s}s (${speedup}x más rápido que secuencial).`
  );
}

function aplicarDiagnosticoAlEstado(diagnostico, tiempoSeg) {
  const clase = diagnostico.etiqueta;
  const n = estadoDashboard.conteoClases[clase];
  estadoDashboard.temperaturasMedias[clase] =
    (estadoDashboard.temperaturasMedias[clase] * n + diagnostico.temperatura_estimada) / (n + 1);
  estadoDashboard.conteoClases[clase] += 1;
  estadoDashboard.totalMuestras += 1;

  const temps = Object.values(estadoDashboard.temperaturasMedias);
  estadoDashboard.tempPromedio = temps.reduce((a, b) => a + b, 0) / temps.length;
  estadoDashboard.varianza = Math.sqrt(
    temps.reduce((acc, t) => acc + (t - estadoDashboard.tempPromedio) ** 2, 0) / temps.length
  );
  estadoDashboard.indiceMadurez = Math.round(
    (estadoDashboard.conteoClases.alta_calidad / estadoDashboard.totalMuestras) * 100
  );
  estadoDashboard.anomalias = estadoDashboard.conteoClases.defectuoso;
  estadoDashboard.precisionModelo = diagnostico.confianza * 100;
  estadoDashboard.tiempoProcMs = tiempoSeg;

  const zonaRotativa = Object.keys(estadoDashboard.zonas)[estadoDashboard.totalMuestras % 6];
  estadoDashboard.zonas[zonaRotativa] = diagnostico.temperatura_estimada;
}

async function verificarConexionBackend() {
  const pill = document.getElementById("pill-estado-backend");
  try {
    const salud = await consultarSalud();
    if (salud.modelo_cargado) {
      pill.innerHTML = `<span class="dot"></span> EN VIVO (${salud.origen_modelo})`;
      pill.classList.remove("error");
    } else {
      pill.innerHTML = `<span class="dot"></span> BACKEND SIN MODELO`;
      pill.classList.add("error");
    }
  } catch {
    pill.innerHTML = `<span class="dot"></span> BACKEND DESCONECTADO`;
    pill.classList.add("error");
  }
}

async function manejarClickEntrenar() {
  const boton = document.getElementById("btn-entrenar");
  boton.disabled = true;
  try {
    await iniciarEntrenamiento();
    mostrarToast("Entrenamiento por épocas iniciado en el backend.");
  } catch (error) {
    mostrarToast(`No se pudo iniciar el entrenamiento: ${error.message}`, true);
  } finally {
    boton.disabled = false;
  }
}

async function actualizarEstadoEntrenamiento() {
  const pill = document.getElementById("estado-entrenamiento-pill");
  const barra = document.getElementById("entrenamiento-barra-relleno");
  const texto = document.getElementById("entrenamiento-texto");

  try {
    const estado = await consultarEstadoEntrenamiento();
    pill.className = `estado-pill ${estado.estado}`;
    pill.textContent = estado.estado.replace("_", " ");

    if (estado.estado === "sin_iniciar") {
      barra.style.width = "0%";
      texto.textContent = "Sin entrenamientos registrados todavía.";
      return;
    }

    const pct = Math.round((estado.epoca_actual / estado.epocas_totales) * 100);
    barra.style.width = `${pct}%`;

    if (estado.estado === "en_progreso") {
      const ultima = estado.historial[estado.historial.length - 1];
      texto.textContent = `Época ${estado.epoca_actual}/${estado.epocas_totales} · pérdida_train: ${ultima ? ultima.perdida_train : "—"}`;
    } else if (estado.estado === "completado") {
      texto.textContent = `Completado · accuracy_test: ${(estado.accuracy_test * 100).toFixed(1)}% sobre ${estado.epocas_totales} épocas`;
    }
  } catch {
    pill.className = "estado-pill error";
    pill.textContent = "sin conexión";
    texto.textContent = "No se pudo consultar /entrenamiento/estado.";
  }
}

async function actualizarEstadoHDFS() {
  const pill = document.getElementById("hdfs-pill");
  const pendientes = document.getElementById("hdfs-pendientes");
  const proxima = document.getElementById("hdfs-proxima");

  try {
    const estado = await consultarEstadoHDFS();
    pendientes.textContent = estado.total_lotes_pendientes;
    pill.className = `estado-pill ${estado.total_lotes_pendientes > 0 ? "en_progreso" : "completado"}`;
    pill.textContent = estado.total_lotes_pendientes > 0 ? "lotes esperando análisis" : "al día";
    proxima.textContent = `cada ${Math.round(estado.intervalo_ingesta_s / 60)} min (automático)`;
  } catch {
    pill.className = "estado-pill error";
    pill.textContent = "sin conexión";
    pendientes.textContent = "—";
    proxima.textContent = "—";
  }
}

async function manejarFalloSensor() {
  const boton = document.getElementById("btn-fallo-sensor");
  boton.disabled = true;
  try {
    const r = await simularFalloSensor(3);
    mostrarToast(`❄ Fallo de sensor simulado por ${r.lotes_afectados} lotes — observa caer la precisión y dispararse el drift.`, true);
  } catch (error) {
    mostrarToast(`No se pudo simular el fallo: ${error.message}`, true);
  } finally {
    setTimeout(() => { boton.disabled = false; }, 2000);
  }
}

async function actualizarMetricaVivo() {
  try {
    estadoDashboard.metricaVivo = await consultarMetricasVivo();
  } catch {
    estadoDashboard.metricaVivo = null;
  }
  renderTarjetasStats(estadoDashboard);
}

function exportarEstadoComoJSON() {
  const blob = new Blob([JSON.stringify(estadoDashboard, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const enlace = document.createElement("a");
  enlace.href = url;
  enlace.download = "distribucion_termica.json";
  enlace.click();
  URL.revokeObjectURL(url);
}

function mostrarToast(mensaje, esError = false) {
  const toast = document.getElementById("toast");
  toast.textContent = mensaje;
  toast.classList.toggle("error", esError);
  toast.classList.remove("oculto");
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => toast.classList.add("oculto"), 3500);
}
