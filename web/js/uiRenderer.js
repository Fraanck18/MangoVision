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

function estadoZonaPorTemp(temp) {
  if (temp < 21) return { texto: "Fría", clase: "frio" };
  if (temp < 33) return { texto: "Óptima", clase: "optimo" };
  if (temp < 38) return { texto: "Alerta", clase: "alerta" };
  return { texto: "Crítica", clase: "critico" };
}

function renderTarjetasStats(estado) {
  const cont = document.getElementById("tarjetas-stats");
  if (!cont) return;

  const { tempPromedio, indiceMadurez, varianza, anomalias, precisionModelo } = estado;

  cont.innerHTML = `
    <div class="tarjeta-stat">
      <div class="etiqueta">TEMP. PROMEDIO 🌡</div>
      <div class="valor">${tempPromedio.toFixed(1)} °C</div>
      <div class="sub positivo">↑ +1.2°C vs ayer</div>
    </div>
    <div class="tarjeta-stat">
      <div class="etiqueta">ÍNDICE MADUREZ 🔘</div>
      <div class="valor">${Math.round(indiceMadurez)}/100</div>
      <div class="sub positivo">${indiceMadurez >= 60 ? "Óptimo" : "En desarrollo"}</div>
      <div class="barra-mini"><div style="width:${Math.min(100, indiceMadurez)}%"></div></div>
    </div>
    <div class="tarjeta-stat">
      <div class="etiqueta">VARIANZA TÉRMICA ⚡</div>
      <div class="valor">±${varianza.toFixed(1)} °C</div>
      <div class="sub">Normal</div>
    </div>
    <div class="tarjeta-stat">
      <div class="etiqueta">ANOMALÍAS ⚠</div>
      <div class="valor">${anomalias} zonas</div>
      <div class="sub ${anomalias > 0 ? "alerta" : "positivo"}">${anomalias > 0 ? "Requiere revisión" : "Sin hallazgos"}</div>
    </div>
    ${renderTarjetaPrecision(estado.metricaVivo)}
  `;
}

function renderTarjetaPrecision(mv) {
  if (!mv || mv.estado !== "ok") {
    return `
    <div class="tarjeta-stat">
      <div class="etiqueta">PRECISIÓN MODELO (ventana) 🎯</div>
      <div class="valor">— %</div>
      <div class="sub">Esperando stream Kafka…</div>
    </div>`;
  }

  const flechas = { upward: "↑", downward: "↓", DOWNWARD_SHARP: "↓↓", stable: "→" };
  const flecha = flechas[mv.trend] || "→";

  if (mv.drift_alert) {
    return `
    <div class="tarjeta-stat">
      <div class="etiqueta">PRECISIÓN MODELO (ventana) 🎯</div>
      <div class="valor">${mv.rolling_accuracy.toFixed(1)} %</div>
      <div class="sub alerta">⚠ DRIFT — posible condensación FLIR</div>
      <div class="barra-mini"><div style="width:${Math.min(100, mv.rolling_accuracy)}%; background:var(--red)"></div></div>
    </div>`;
  }

  const clase = mv.rolling_accuracy >= 90 ? "positivo" : mv.rolling_accuracy >= 80 ? "" : "alerta";
  return `
    <div class="tarjeta-stat">
      <div class="etiqueta">PRECISIÓN MODELO (ventana ${mv.ventana}) 🎯</div>
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
        <div class="detalle">${conteo} unidades · Temp. media: ${tempMedia.toFixed(1)}°C</div>
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
    ["RANGO TEMP.", "-20 ~ 150°C"],
    ["ALGORITMO", "CNN ResNet-50"],
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
        <div class="zona-temp">${temp.toFixed(1)}°C</div>
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
    : `${diagnostico.temperatura_estimada.toFixed(0)}°`;

  tira.insertBefore(thumb, boton);
}
