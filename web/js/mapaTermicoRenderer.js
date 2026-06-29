/**
 * Renderizador del Mapa Térmico por Camión y Lote.
 *
 * Genera datos simulados de la flota de camiones en la cadena de frío,
 * cada uno transportando múltiples lotes de mango. Los datos térmicos
 * se generan proceduralmente con distribuciones coherentes con el modelo
 * físico de thermal_inertia.py, sin modificar el backend.
 */

// ---------------------------------------------------------------------------
// 1. Generación de datos simulados de la flota
// ---------------------------------------------------------------------------
const RUTAS = ["Ruta Norte", "Ruta Sur", "Ruta Este"];
const ORIGENES = [
  "Finca La Esperanza",
  "Finca San Pedro",
  "Finca El Roble",
];

function _rand(min, max) { return min + Math.random() * (max - min); }
function _randInt(min, max) { return Math.floor(_rand(min, max + 1)); }

function clasificarPorTemp(temp) {
  if (temp < 24) return { clase: "verde", etiqueta: "Verde", color: "#2f6fed" };
  if (temp < 33) return { clase: "maduro", etiqueta: "Maduro", color: "#1f9d6b" };
  return { clase: "sobremaduro", etiqueta: "Sobre-maduro", color: "#dc2626" };
}

function generarMuestra() {
  const temp = _rand(15, 43);
  const { clase, etiqueta, color } = clasificarPorTemp(temp);
  const confianza = _rand(85, 99.5);
  return { temp: +temp.toFixed(1), clase, etiqueta, color, confianza: +confianza.toFixed(1) };
}

function generarLote(camionId, loteIdx) {
  const numMuestras = _randInt(18, 32);
  const muestras = Array.from({ length: numMuestras }, () => generarMuestra());
  const temps = muestras.map((m) => m.temp);
  const tempProm = temps.reduce((a, b) => a + b, 0) / temps.length;
  const tempMin = Math.min(...temps);
  const tempMax = Math.max(...temps);

  const conteo = { verde: 0, maduro: 0, sobremaduro: 0 };
  muestras.forEach((m) => conteo[m.clase]++);
  const claseMayoritaria = Object.entries(conteo).sort((a, b) => b[1] - a[1])[0][0];

  // Determinar alertas a nivel de lote
  const alertas = [];
  const pctSobremaduro = (conteo.sobremaduro / numMuestras) * 100;
  if (pctSobremaduro > 40) alertas.push({ tipo: "Crítico", msg: `${pctSobremaduro.toFixed(0)}% sobre-maduros detectados.` });
  else if (pctSobremaduro > 20) alertas.push({ tipo: "Alerta", msg: `Sobre-maduración incipiente detectada.` });
  if (tempMax > 40) alertas.push({ tipo: "Crítico", msg: `T° crítica ${tempMax.toFixed(1)}°C. Riesgo de pérdida.` });

  return {
    id: `${camionId}-L${String(loteIdx + 1).padStart(2, "0")}`,
    nombre: `Lote ${String.fromCharCode(65 + loteIdx)}-${String(loteIdx + 1).padStart(2, "0")}`,
    muestras,
    numMuestras,
    tempProm: +tempProm.toFixed(1),
    tempMin: +tempMin.toFixed(1),
    tempMax: +tempMax.toFixed(1),
    conteo,
    claseMayoritaria,
    alertas,
  };
}

function generarCamion(idx) {
  const id = `TRK-${String(idx + 1).padStart(2, "0")}`;
  const numLotes = _randInt(3, 5);
  const lotes = Array.from({ length: numLotes }, (_, i) => generarLote(id, i));

  const totalMuestras = lotes.reduce((a, l) => a + l.numMuestras, 0);
  const tempProm = lotes.reduce((a, l) => a + l.tempProm * l.numMuestras, 0) / totalMuestras;
  const tempMin = Math.min(...lotes.map((l) => l.tempMin));
  const tempMax = Math.max(...lotes.map((l) => l.tempMax));

  const conteoGlobal = { verde: 0, maduro: 0, sobremaduro: 0 };
  lotes.forEach((l) => {
    conteoGlobal.verde += l.conteo.verde;
    conteoGlobal.maduro += l.conteo.maduro;
    conteoGlobal.sobremaduro += l.conteo.sobremaduro;
  });

  const alertasCamion = lotes.flatMap((l) => l.alertas.map((a) => ({ ...a, lote: l.nombre, loteId: l.id })));
  const estadoCamion = alertasCamion.some((a) => a.tipo === "Crítico") ? "Crítico"
    : alertasCamion.length > 0 ? "Alerta" : "Óptimo";

  return {
    id, nombre: `Camión ${idx + 1}`,
    ruta: RUTAS[idx % RUTAS.length],
    origen: ORIGENES[idx % ORIGENES.length],
    lotes, numLotes, totalMuestras,
    tempProm: +tempProm.toFixed(1), tempMin: +tempMin.toFixed(1), tempMax: +tempMax.toFixed(1),
    conteo: conteoGlobal,
    alertas: alertasCamion,
    estado: estadoCamion,
  };
}

let datosFlota = null;

function obtenerDatosFlota() {
  datosFlota = Array.from({ length: 3 }, (_, i) => generarCamion(i));
  return datosFlota;
}


// ---------------------------------------------------------------------------
// 2. Render principal
// ---------------------------------------------------------------------------
function renderMapaTermico(filtro) {
  filtro = filtro || "todos";
  const flota = datosFlota || obtenerDatosFlota();

  renderListaCamiones(flota);
  renderResumenGlobal(flota);
  renderCamionesDetalle(flota, filtro);
  renderMetricas(flota);
  renderAlertas(flota);
  renderDistribucionCamion(flota);

  const actEl = document.getElementById("mt-ultima-act");
  if (actEl) {
    const ahora = new Date();
    actEl.textContent = `Última actualización: ${ahora.toLocaleTimeString()} · En vivo`;
  }
}


// ---------------------------------------------------------------------------
// 3. Columna izquierda: Lista de camiones + Resumen
// ---------------------------------------------------------------------------
function renderListaCamiones(flota) {
  const cont = document.getElementById("mt-lista-camiones");
  if (!cont) return;
  cont.innerHTML = flota.map((c) => {
    const colorEstado = c.estado === "Crítico" ? "var(--red)" : c.estado === "Alerta" ? "var(--amber)" : "var(--accent)";
    return `
      <div class="mt-camion-item">
        <div class="mt-camion-item-head">
          <strong>${c.id}</strong>
          <span class="mt-estado-badge" style="color:${colorEstado};border-color:${colorEstado};">${c.estado}</span>
        </div>
        <div class="mt-camion-item-sub">${c.numLotes} lotes · ${c.totalMuestras} muestras</div>
      </div>`;
  }).join("");
}

function renderResumenGlobal(flota) {
  const cont = document.getElementById("mt-resumen-global");
  if (!cont) return;
  const totalMuestras = flota.reduce((a, c) => a + c.totalMuestras, 0);
  const totalLotes = flota.reduce((a, c) => a + c.numLotes, 0);
  const tempGlobal = flota.reduce((a, c) => a + c.tempProm * c.totalMuestras, 0) / totalMuestras;
  const totalAlertas = flota.reduce((a, c) => a + c.alertas.length, 0);

  const conteoG = { verde: 0, maduro: 0, sobremaduro: 0 };
  flota.forEach((c) => {
    conteoG.verde += c.conteo.verde;
    conteoG.maduro += c.conteo.maduro;
    conteoG.sobremaduro += c.conteo.sobremaduro;
  });
  const pctM = ((conteoG.maduro / totalMuestras) * 100).toFixed(0);
  const pctV = ((conteoG.verde / totalMuestras) * 100).toFixed(0);
  const pctS = ((conteoG.sobremaduro / totalMuestras) * 100).toFixed(0);

  cont.innerHTML = `
    <div class="mt-resumen-fila"><span>Total muestras</span><strong>${totalMuestras}</strong></div>
    <div class="mt-resumen-fila"><span>Lotes activos</span><strong>${totalLotes}</strong></div>
    <div class="mt-resumen-fila"><span>T° prom. global</span><strong>${tempGlobal.toFixed(1)} °C</strong></div>
    <div class="mt-resumen-fila"><span>Alertas activas</span><strong style="color:var(--red)">${totalAlertas}</strong></div>
    <div class="mt-resumen-barras">
      <div class="mt-resumen-barra-row"><span style="color:var(--accent)">Maduro</span><div class="mt-barra-bg"><div class="mt-barra-fill" style="width:${pctM}%;background:var(--accent)"></div></div><span>${pctM}%</span></div>
      <div class="mt-resumen-barra-row"><span style="color:var(--blue)">Verde</span><div class="mt-barra-bg"><div class="mt-barra-fill" style="width:${pctV}%;background:var(--blue)"></div></div><span>${pctV}%</span></div>
      <div class="mt-resumen-barra-row"><span style="color:var(--red)">S-Maduro</span><div class="mt-barra-bg"><div class="mt-barra-fill" style="width:${pctS}%;background:var(--red)"></div></div><span>${pctS}%</span></div>
    </div>`;
}


// ---------------------------------------------------------------------------
// 4. Columna central: Tarjetas de camión con lotes y heatmap
// ---------------------------------------------------------------------------
function colorCelda(temp) {
  if (temp < 24) return "#2563eb";
  if (temp < 27) return "#16a34a";
  if (temp < 30) return "#22c55e";
  if (temp < 33) return "#84cc16";
  if (temp < 36) return "#eab308";
  if (temp < 39) return "#f97316";
  return "#ef4444";
}

function renderCamionesDetalle(flota, filtro) {
  const cont = document.getElementById("mt-camiones-detalle");
  if (!cont) return;

  cont.innerHTML = flota.map((camion) => {
    const colorEstado = camion.estado === "Crítico" ? "var(--red)" : camion.estado === "Alerta" ? "var(--amber)" : "var(--accent)";
    const numAlertas = camion.alertas.length;
    const alertaHtml = numAlertas > 0 ? `<span class="mt-estado-badge" style="color:${colorEstado};border-color:${colorEstado}">${numAlertas} alerta${numAlertas > 1 ? "s" : ""}</span>` : `<span class="mt-estado-badge" style="color:var(--accent);border-color:var(--accent)">Óptimo</span>`;

    const lotesHtml = camion.lotes.map((lote) => {
      // Filtrar por estado si aplica
      if (filtro !== "todos") {
        const claseFiltro = filtro;
        if (lote.claseMayoritaria !== claseFiltro) return "";
      }

      // Heatmap: cada muestra es una celda coloreada
      const celdasHtml = lote.muestras.map((m) => {
        const bg = colorCelda(m.temp);
        return `<div class="mt-celda" style="background:${bg}" title="${m.temp}°C · ${m.etiqueta} · Conf: ${m.confianza}%">${m.temp.toFixed(0)}°</div>`;
      }).join("");

      const pctM = ((lote.conteo.maduro / lote.numMuestras) * 100).toFixed(0);
      const pctV = ((lote.conteo.verde / lote.numMuestras) * 100).toFixed(0);
      const pctS = ((lote.conteo.sobremaduro / lote.numMuestras) * 100).toFixed(0);

      const loteAlertaHtml = lote.alertas.length > 0 ? `<span class="mt-alerta-lote">⚠</span>` : "";

      return `
        <div class="mt-lote-fila">
          <div class="mt-lote-id">${lote.nombre} ${loteAlertaHtml}</div>
          <div class="mt-heatmap-grid">${celdasHtml}</div>
          <div class="mt-lote-resumen">
            <span style="color:var(--accent)">${pctM}% M</span>
            <span style="color:var(--blue)">${pctV}% V</span>
            ${parseInt(pctS) > 0 ? `<span style="color:var(--red)">${pctS}% S</span>` : ""}
          </div>
        </div>`;
    }).join("");

    if (!lotesHtml.trim()) return "";

    return `
      <div class="panel mt-camion-card">
        <div class="mt-camion-card-head">
          <div>
            <div class="mt-camion-card-titulo">${camion.nombre} — ${camion.ruta}</div>
            <div class="mt-camion-card-sub">Origen: ${camion.origen} · ${camion.numLotes} lotes · ${camion.totalMuestras} muestras</div>
          </div>
          <div class="mt-camion-card-stats">
            <span>${camion.tempProm}°C</span>
            <span>${camion.tempMax}°C</span>
            ${alertaHtml}
          </div>
        </div>
        <div class="mt-lotes-container">${lotesHtml}</div>
      </div>`;
  }).join("");
}


// ---------------------------------------------------------------------------
// 5. Columna derecha: Métricas, alertas, distribución
// ---------------------------------------------------------------------------
function renderMetricas(flota) {
  const cont = document.getElementById("mt-metricas");
  if (!cont) return;
  const totalMuestras = flota.reduce((a, c) => a + c.totalMuestras, 0);
  const tempMin = Math.min(...flota.map((c) => c.tempMin));
  const tempMax = Math.max(...flota.map((c) => c.tempMax));
  const tempProm = flota.reduce((a, c) => a + c.tempProm * c.totalMuestras, 0) / totalMuestras;

  const confianzas = flota.flatMap((c) => c.lotes.flatMap((l) => l.muestras.map((m) => m.confianza)));
  const confProm = confianzas.reduce((a, b) => a + b, 0) / confianzas.length;

  cont.innerHTML = `
    <div class="mt-metrica-fila"><span>T° mínima</span><strong style="color:var(--blue)">${tempMin.toFixed(1)} °C</strong></div>
    <div class="mt-metrica-fila"><span>T° promedio</span><strong>${tempProm.toFixed(1)} °C</strong></div>
    <div class="mt-metrica-fila"><span>T° máxima</span><strong style="color:var(--red)">${tempMax.toFixed(1)} °C</strong></div>
    <div class="mt-metrica-fila"><span>Lotes OK</span><strong style="color:var(--accent)">${flota.reduce((a, c) => a + c.lotes.filter((l) => l.alertas.length === 0).length, 0)} / ${flota.reduce((a, c) => a + c.numLotes, 0)}</strong></div>
    <div class="mt-metrica-fila"><span>Confianza prom.</span><strong>${confProm.toFixed(1)}%</strong></div>
    <div class="mt-metrica-fila"><span>Resolución</span><strong>320 × 240</strong></div>`;
}

function renderAlertas(flota) {
  const cont = document.getElementById("mt-alertas");
  if (!cont) return;
  const alertas = flota.flatMap((c) => c.alertas.map((a) => ({ ...a, camion: c.id })));

  if (alertas.length === 0) {
    cont.innerHTML = `<p class="holo-hint">Sin alertas activas. Todos los lotes dentro de parámetros.</p>`;
    return;
  }

  cont.innerHTML = alertas.slice(0, 5).map((a) => {
    const esCritico = a.tipo === "Crítico";
    return `
      <div class="mt-alerta-card ${esCritico ? "critico" : "alerta"}">
        <div class="mt-alerta-titulo">${a.camion} · ${a.lote}</div>
        <div class="mt-alerta-msg">${a.msg}</div>
        <span class="mt-estado-badge" style="color:${esCritico ? "var(--red)" : "var(--amber)"};border-color:${esCritico ? "var(--red)" : "var(--amber)"}">${a.tipo}</span>
      </div>`;
  }).join("");
}

function renderDistribucionCamion(flota) {
  const cont = document.getElementById("mt-distribucion-camion");
  if (!cont) return;

  cont.innerHTML = flota.map((c) => {
    const pctM = ((c.conteo.maduro / c.totalMuestras) * 100).toFixed(0);
    const pctV = ((c.conteo.verde / c.totalMuestras) * 100).toFixed(0);
    const pctS = ((c.conteo.sobremaduro / c.totalMuestras) * 100).toFixed(0);

    return `
      <div class="mt-dist-camion">
        <div class="mt-dist-head"><strong>${c.id}</strong><span>${c.totalMuestras} muestras</span></div>
        <div class="mt-dist-barra">
          <div style="width:${pctM}%;background:var(--accent)" title="Maduro ${pctM}%"></div>
          <div style="width:${pctV}%;background:var(--blue)" title="Verde ${pctV}%"></div>
          <div style="width:${pctS}%;background:var(--red)" title="Sobre-maduro ${pctS}%"></div>
        </div>
        <div class="mt-dist-leyenda">
          <span style="color:var(--accent)">${pctM}% M</span>
          <span style="color:var(--blue)">${pctV}% V</span>
          <span style="color:var(--red)">${pctS}% S</span>
        </div>
      </div>`;
  }).join("");
}
