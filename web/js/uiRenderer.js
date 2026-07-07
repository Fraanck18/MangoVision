/**
 * Renderizado de las tarjetas de estado, clasificacion por lote, parametros
 * de analisis, zonas de temperatura y lista de muestras recientes. Todo se
 * alimenta de `estadoDashboard` (ver main.js), que se actualiza con cada
 * respuesta real del backend (/diagnostico, /entrenamiento/estado).
 */
const ZONAS_BASE = [
  { id: "A", nombre: "ZONA A" },
  { id: "B", nombre: "ZONA B" },
  { id: "C", nombre: "ZONA C" },
  { id: "D", nombre: "ZONA D" },
  { id: "E", nombre: "ZONA E" },
  { id: "F", nombre: "ZONA F" },
];

// [F10] Estado de zona por ITR relativo (0-100), delegado a la escala unica
// (escala.js). Antes usaba umbrales °C (21/33/38) que no aplican a ITR.
function estadoZonaPorTemp(itr) {
  return estadoZonaPorITR(itr);
}

function renderTarjetasStats(estado) {
  const cont = document.getElementById("tarjetas-stats");
  if (!cont) return;

  const { tempPromedio, indiceMadurez, varianza, anomalias } = estado;

  cont.innerHTML = `
    <div class="tarjeta-stat">
      <div class="etiqueta">ITR PROMEDIO 🌡</div>
      <div class="valor">${tempPromedio.toFixed(1)} ITR</div>
      <div class="sub positivo">índice relativo (no °C)</div>
    </div>
    <div class="tarjeta-stat">
      <div class="etiqueta">ÍNDICE MADUREZ 🔘</div>
      <div class="valor">${Math.round(indiceMadurez)}/100</div>
      <div class="sub positivo">${indiceMadurez >= 60 ? "Óptimo" : "En desarrollo"}</div>
      <div class="barra-mini"><div style="width:${Math.min(100, indiceMadurez)}%"></div></div>
    </div>
    <div class="tarjeta-stat">
      <div class="etiqueta">VARIANZA ITR ⚡</div>
      <div class="valor">±${varianza.toFixed(1)} ITR</div>
      <div class="sub">Normal</div>
    </div>
    <div class="tarjeta-stat">
      <div class="etiqueta">ANOMALÍAS ⚠</div>
      <div class="valor">${anomalias} zonas</div>
      <div class="sub ${anomalias > 0 ? "alerta" : "positivo"}">${anomalias > 0 ? "Requiere revisión" : "Sin hallazgos"}</div>
    </div>
    ${renderTarjetaVerificacion(estado.metricaVivo)}
    ${renderTarjetaPrecision(estado.metricaVivo)}
  `;
}

/**
 * Tarjeta de VERIFICACIÓN de anomalía (paso previo a clasificar, streaming).
 * Muestra SIEMPRE la tasa del detector PEDTE junto al baseline trivial de brillo
 * (guardarraíl de honestidad): en dato real el trivial es la referencia (AUC 0.87).
 */
function renderTarjetaVerificacion(mv) {
  if (!mv || mv.estado !== "ok" || mv.tasa_anomalia_trivial === undefined) {
    return `
    <div class="tarjeta-stat">
      <div class="etiqueta">VERIFICACIÓN (anomalía) 🔎</div>
      <div class="valor">— %</div>
      <div class="sub">Esperando stream…</div>
    </div>`;
  }
  return `
    <div class="tarjeta-stat">
      <div class="etiqueta">VERIFICACIÓN (anomalía) 🔎</div>
      <div class="valor">${mv.tasa_anomalia_trivial.toFixed(1)} %</div>
      <div class="sub">trivial (referencia) · detector PEDTE ${mv.tasa_anomalia_detector.toFixed(1)} %</div>
    </div>`;
}

function renderTarjetaPrecision(mv) {
  if (!mv || mv.estado !== "ok") {
    return `
    <div class="tarjeta-stat">
      <div class="etiqueta">CLASIFICACIÓN · accuracy ventana 🎯</div>
      <div class="valor">— %</div>
      <div class="sub">Esperando stream Kafka…</div>
    </div>`;
  }

  const flechas = { upward: "↑", downward: "↓", DOWNWARD_SHARP: "↓↓", stable: "→" };
  const flecha = flechas[mv.trend] || "→";

  if (mv.drift_alert) {
    return `
    <div class="tarjeta-stat">
      <div class="etiqueta">CLASIFICACIÓN · accuracy ventana 🎯</div>
      <div class="valor">${mv.rolling_accuracy.toFixed(1)} %</div>
      <div class="sub alerta">⚠ DRIFT — posible condensación FLIR</div>
      <div class="barra-mini"><div style="width:${Math.min(100, mv.rolling_accuracy)}%; background:var(--red)"></div></div>
    </div>`;
  }

  const clase = mv.rolling_accuracy >= 90 ? "positivo" : mv.rolling_accuracy >= 80 ? "" : "alerta";
  return `
    <div class="tarjeta-stat">
      <div class="etiqueta">CLASIFICACIÓN · accuracy ventana ${mv.ventana} 🎯</div>
      <div class="valor">${mv.rolling_accuracy.toFixed(1)} %</div>
      <div class="sub ${clase}">${flecha} ${mv.trend} · ${mv.muestras_totales} eval · conf ${mv.confianza_media}</div>
      <div class="barra-mini"><div style="width:${Math.min(100, mv.rolling_accuracy)}%"></div></div>
    </div>`;
}

function renderClasificacionPorLote(estado) {
  const cont = document.getElementById("lista-clasificacion");
  if (!cont) return;

  const total = Object.values(estado.conteoClases).reduce((a, b) => a + b, 0) || 1;
  const orden = [
    { id: "calidad_promedio", nombre: "Verde / Inmaduro", color: "#2f6fed" },
    { id: "alta_calidad", nombre: "Maduro / Óptimo", color: "#1f9d6b" },
    { id: "defectuoso", nombre: "Sobre-Maduro", color: "#dc2626" },
  ];

  cont.innerHTML = orden.map((clase) => {
    const conteo = estado.conteoClases[clase.id] || 0;
    const pct = Math.round((conteo / total) * 100);
    const tempMedia = estado.temperaturasMedias[clase.id];
    return `
      <div class="item-clasif">
        <div class="fila-top">
          <span class="nombre"><span class="punto" style="background:${clase.color}"></span>${clase.nombre}</span>
          <span class="pct" style="color:${clase.color}">${pct}%</span>
        </div>
        <div class="barra"><div style="width:${pct}%; background:${clase.color}"></div></div>
        <div class="detalle">${conteo} unidades · ITR medio: ${tempMedia.toFixed(1)}</div>
      </div>
    `;
  }).join("");
}

function renderParametrosAnalisis(estado) {
  const cont = document.getElementById("grilla-parametros");
  if (!cont) return;

  const parametros = [
    ["RESOLUCIÓN IR", "320 × 240 px"],
    ["FPS CAPTURA", "9 fps"],
    ["SENSOR", "FLIR Lepton 3.5"],
    ["EMISIVIDAD", "0.97"],
    ["ESCALA", "ITR 0–100 (relativa)"],
    ["ALGORITMO", "CNN ResNet50"],
    ["MUESTRAS LOTE", `${estado.totalMuestras} und.`],
    ["TIEMPO PROC.", `${estado.tiempoProcMs.toFixed(1)} s / imagen`],
  ];

  cont.innerHTML = parametros.map(([etq, val]) => `
    <div class="parametro"><div class="etq">${etq}</div><div class="val">${val}</div></div>
  `).join("");
}

function renderZonasTemperatura(estado) {
  const cont = document.getElementById("grilla-zonas");
  if (!cont) return;

  cont.innerHTML = ZONAS_BASE.map((zona) => {
    const temp = estado.zonas[zona.id];
    const info = estadoZonaPorTemp(temp);
    return `
      <div class="zona-card estado-${info.clase}">
        <div class="zona-nombre">${zona.nombre}</div>
        <div class="zona-temp">${temp.toFixed(1)} ITR</div>
        <div class="zona-estado">${info.texto}</div>
      </div>
    `;
  }).join("");
}

function renderMuestraReciente(diagnostico, urlImagen) {
  const tira = document.getElementById("tira-muestras");
  const boton = document.getElementById("btn-agregar-muestra");
  if (!tira || !boton) return;

  const colorClase = diagnostico.color;
  const thumb = document.createElement("div");
  thumb.className = "muestra-thumb";
  thumb.style.background = colorClase;
  thumb.title = `${diagnostico.categoria} · ${(diagnostico.confianza * 100).toFixed(1)}%`;
  thumb.innerHTML = urlImagen
    ? `<img src="${urlImagen}" alt="${diagnostico.categoria}">`
    : `${diagnostico.indice_itr.toFixed(0)} ITR`;

  tira.insertBefore(thumb, boton);
}
