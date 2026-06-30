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
  inicializarSelectorTema();

  document.getElementById("btn-importar").addEventListener("click", () => abrirSelectorImagen());
  document.getElementById("btn-agregar-muestra").addEventListener("click", () => abrirSelectorImagen());
  document.getElementById("input-imagen").addEventListener("change", manejarSeleccionImagen);
  document.getElementById("btn-analizar").addEventListener("click", manejarClickAnalizar);
  document.getElementById("btn-entrenar").addEventListener("click", manejarClickEntrenar);
  document.getElementById("btn-detener-entrenamiento").addEventListener("click", manejarClickDetenerEntrenamiento);
  document.getElementById("btn-exportar-reporte").addEventListener("click", exportarReporteCSV);
  document.getElementById("btn-ver-historial").addEventListener("click", alternarHistorialEntrenamiento);
  document.getElementById("btn-exportar").addEventListener("click", exportarEstadoComoJSON);
  document.getElementById("btn-fallo-sensor").addEventListener("click", manejarFalloSensor);

  document.querySelectorAll(".nav-link[data-vista]").forEach((boton) => {
    boton.addEventListener("click", () => navegarVista(boton.dataset.vista));
  });
  document.querySelectorAll("#selector-modo button").forEach((boton) => {
    boton.addEventListener("click", () => window.cambiarModoVisualizacion?.(boton.dataset.modo));
  });
  document.querySelectorAll("#holo-clases button").forEach((boton) => {
    boton.addEventListener("click", () => window.seleccionarClase3D?.(boton.dataset.clase));
  });
  document.getElementById("holo-escala-z").addEventListener("input", (e) => window.setEscalaZ3D?.(e.target.value));
  document.getElementById("holo-autorot").addEventListener("change", (e) => window.setAutoRotar3D?.(e.target.checked));

  // Mapa Termico: boton actualizar y filtros
  document.getElementById("btn-mt-actualizar").addEventListener("click", () => renderMapaTermico());
  document.querySelectorAll("#mt-filtro-estado button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#mt-filtro-estado button").forEach((x) => x.classList.remove("activo"));
      b.classList.add("activo");
      renderMapaTermico(b.dataset.filtro);
    });
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
  renderGraficaDistribucion();
}

/**
 * Navegacion entre vistas del sidebar. Las tres vistas principales son:
 * "dashboard", "mapa-termico" y "vista3d". Cada vista tiene su propio div
 * con id correspondiente. La Vista 3D se inicializa de forma perezosa.
 */
function navegarVista(vista) {
  const vistas = ["vista-dashboard", "vista-mapa-termico", "vista-3d"];
  const mapaVista = {
    "dashboard": "vista-dashboard",
    "mapa-termico": "vista-mapa-termico",
    "vista3d": "vista-3d",
  };
  const idActiva = mapaVista[vista] || "vista-dashboard";

  vistas.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("oculto", id !== idActiva);
  });

  document.querySelectorAll(".nav-link[data-vista]").forEach((b) => {
    b.classList.toggle("activo", b.dataset.vista === vista);
  });

  if (vista === "vista3d") {
    window.renderMangos3D?.();
    requestAnimationFrame(() => window.ajustarVisor3D?.());
  }
  if (vista === "mapa-termico") {
    renderMapaTermico();
  }
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
  window.actualizarClasificacionHolografica?.(diagnostico);
  acumularMuestraGrafica(diagnostico.temperatura_estimada);

  const urlImagen = URL.createObjectURL(archivoSeleccionado);
  renderMuestraReciente(diagnostico, urlImagen);
  // El holograma 3D usa la imagen REAL importada como mapa de calor (relieve Z).
  window.cargarImagenHolograma?.(urlImagen);
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
  window.actualizarClasificacionHolografica?.(resultado.resultados[resultado.resultados.length - 1]);
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
  const candidatos = document.getElementById("entrenamiento-candidatos");
  const btnEntrenar = document.getElementById("btn-entrenar");
  const btnDetener = document.getElementById("btn-detener-entrenamiento");
  const btnExportarReporte = document.getElementById("btn-exportar-reporte");

  try {
    const estado = await consultarEstadoEntrenamiento();
    pill.className = `estado-pill ${estado.estado}`;
    pill.textContent = estado.estado.replace("_", " ");

    const enProgreso = estado.estado === "en_progreso" || estado.estado === "iniciando";
    btnEntrenar.disabled = enProgreso;
    btnDetener.disabled = !enProgreso;
    btnExportarReporte.disabled = !estado.reporte_test;

    if (estado.estado === "sin_iniciar") {
      barra.style.width = "0%";
      texto.textContent = "Sin entrenamientos registrados todavía.";
      candidatos.innerHTML = "";
      return;
    }

    if (estado.estado === "iniciando") {
      barra.style.width = "0%";
      texto.textContent = "Arrancando el proceso de entrenamiento…";
      barra.parentElement.classList.remove("oculto");
      return;
    }

    const pct = Math.round((estado.epoca_actual / estado.epocas_totales) * 100);
    barra.style.width = `${pct}%`;

    if (estado.estado === "en_progreso") {
      const ultima = estado.historial[estado.historial.length - 1];
      const progresoCandidato = estado.indice_candidato
        ? `Entrenamiento ${estado.indice_candidato}/${estado.total_candidatos} (${estado.candidato_actual} épocas) · `
        : "";
      texto.textContent = `${progresoCandidato}Época ${estado.epoca_actual}/${estado.epocas_totales} · pérdida_train: ${ultima ? ultima.perdida_train : "—"}`;
      barra.parentElement.classList.remove("oculto");
    } else if (estado.estado === "completado") {
      texto.textContent = `${estado.resultados_candidatos.length} entrenamientos independientes completados.`;
      barra.parentElement.classList.add("oculto");
    } else if (estado.estado === "detenido") {
      texto.textContent = "Entrenamiento detenido manualmente antes de completar los 3 candidatos.";
      barra.parentElement.classList.add("oculto");
    }

    renderCandidatosEntrenamiento(estado, candidatos);
  } catch {
    pill.className = "estado-pill error";
    pill.textContent = "sin conexión";
    texto.textContent = "No se pudo consultar /entrenamiento/estado.";
  }
}

async function manejarClickDetenerEntrenamiento() {
  const boton = document.getElementById("btn-detener-entrenamiento");
  boton.disabled = true;
  try {
    await detenerEntrenamiento();
    mostrarToast("Entrenamiento detenido. Esa corrida quedó registrada en el historial.", true);
    actualizarEstadoEntrenamiento();
  } catch (error) {
    mostrarToast(`No se pudo detener el entrenamiento: ${error.message}`, true);
  }
}

function exportarReporteCSV() {
  const enlace = document.createElement("a");
  enlace.href = urlReporteCSV();
  enlace.download = "reporte_clasificacion.csv";
  enlace.click();
}

let historialVisible = false;

async function alternarHistorialEntrenamiento() {
  const contenedor = document.getElementById("entrenamiento-historial");
  historialVisible = !historialVisible;
  contenedor.classList.toggle("oculto", !historialVisible);
  if (!historialVisible) return;

  try {
    const { corridas } = await consultarHistorialEntrenamiento();
    if (corridas.length === 0) {
      contenedor.innerHTML = `<span class="entrenamiento-info">Todavía no hay corridas completadas en el historial.</span>`;
      return;
    }
    contenedor.innerHTML = corridas.slice().reverse().map((corrida) => {
      const fecha = corrida.fin ? new Date(corrida.fin * 1000).toLocaleString() : "—";
      const detalle = corrida.estado === "detenido"
        ? "detenida manualmente"
        : `mejor: ${corrida.mejor_epocas} épocas · accuracy_test: ${(corrida.accuracy_test * 100).toFixed(1)}%`;
      return `
        <div class="entrenamiento-historial-item">
          <span>${fecha}</span>
          <span class="estado-pill ${corrida.estado}">${corrida.estado}</span>
          <span>${detalle}</span>
        </div>`;
    }).join("");
  } catch (error) {
    contenedor.innerHTML = `<span class="entrenamiento-info">No se pudo cargar el historial: ${error.message}</span>`;
  }
}

/**
 * Cada cantidad de epocas (5/10/15) corre como un entrenamiento INDEPENDIENTE
 * (modelo nuevo, mismo punto de partida) y se muestra aqui como su propio
 * resultado completo -- igual que el panel mostraba un unico entrenamiento de
 * 12 epocas antes, ahora son 3 entrenamientos separados, lado a lado. Solo
 * uno de los tres (el de mejor accuracy_test real) queda guardado como el
 * modelo que sirve el backend; se marca con la insignia "EN PRODUCCIÓN".
 */
function renderCandidatosEntrenamiento(estado, contenedor) {
  const resultados = estado.resultados_candidatos;
  if (!resultados || resultados.length === 0) {
    contenedor.innerHTML = "";
    return;
  }
  const mejorEpocas = estado.mejor_epocas;
  contenedor.innerHTML = resultados.map((r) => {
    const enProduccion = r.epocas === mejorEpocas;
    return `
      <div class="resultado-entrenamiento ${enProduccion ? "en-produccion" : ""}">
        <div class="resultado-cabecera">
          <span class="resultado-titulo">Entrenamiento · ${r.epocas} épocas</span>
          ${enProduccion ? '<span class="resultado-badge">★ EN PRODUCCIÓN</span>' : ""}
        </div>
        <span class="estado-pill completado">completado</span>
        <div class="resultado-accuracy">accuracy_test: <strong>${(r.accuracy_test * 100).toFixed(1)}%</strong></div>
      </div>`;
  }).join("");
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

/**
 * Selector de tema (Claro / Oscuro / Neón). El script inline en <head> ya
 * aplico el tema de localStorage antes de pintar (evita parpadeo); aqui se
 * sincroniza ademas con el backend (`/config/preferencias`), que es la
 * fuente de verdad real -- localStorage queda solo como cache para que el
 * script inline no tenga que esperar una llamada de red en cada carga.
 */
function inicializarSelectorTema() {
  const select = document.getElementById("selector-tema");
  const temaActual = document.documentElement.dataset.tema || "claro";
  select.value = temaActual;
  select.addEventListener("change", () => cambiarTema(select.value));

  obtenerPreferencias()
    .then((preferencias) => {
      if (preferencias.tema && preferencias.tema !== select.value) {
        select.value = preferencias.tema;
        aplicarTema(preferencias.tema);
      }
    })
    .catch(() => { /* backend sin conexion: se queda con lo que ya aplico localStorage */ });
}

function aplicarTema(tema) {
  document.documentElement.dataset.tema = tema;
  localStorage.setItem("thermomango_tema", tema);
}

function cambiarTema(tema) {
  aplicarTema(tema);
  guardarPreferencias({ tema }).catch(() => { /* se queda persistido en localStorage al menos */ });
}

function mostrarToast(mensaje, esError = false) {
  const toast = document.getElementById("toast");
  toast.textContent = mensaje;
  toast.classList.toggle("error", esError);
  toast.classList.remove("oculto");
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => toast.classList.add("oculto"), 3500);
}
