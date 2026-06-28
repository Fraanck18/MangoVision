/**
 * Visualizacion holografica de los 3 estados termicos (clases reales del
 * modelo: calidad_promedio, alta_calidad, defectuoso). Las esferas son CSS
 * puro (radial-gradient + scanlines + particulas), sin librerias externas.
 * `actualizarClasificacionHolografica` resalta la clase devuelta por el
 * backend en /diagnostico tras una inferencia real.
 */
const ESTADOS_TERMICOS = [
  {
    id: "calidad_promedio",
    nombre: "Verde / Inmaduro",
    color: "#2f6fed",
    rango: [15, 21],
    descripcion: "Temperatura baja, sin actividad metabólica significativa",
  },
  {
    id: "alta_calidad",
    nombre: "Maduro / Óptimo",
    color: "#1f9d6b",
    rango: [22, 32],
    descripcion: "Temperatura ideal, distribución uniforme de calor",
  },
  {
    id: "defectuoso",
    nombre: "Sobre-Maduro",
    color: "#dc2626",
    rango: [33, 42],
    descripcion: "Alta temperatura, zonas de descomposición detectadas",
  },
];

let modoVisualizacionActual = "holografico";
let temperaturasActuales = { calidad_promedio: 18.2, alta_calidad: 28.4, defectuoso: 38.7 };
let claseActivaId = "alta_calidad";

function renderMangos3D() {
  const contenedor = document.getElementById("holo-grilla");
  if (!contenedor) return;

  contenedor.innerHTML = ESTADOS_TERMICOS.map((estado) => `
    <div class="holo-tarjeta modo-${modoVisualizacionActual} ${estado.id === claseActivaId ? "activa" : ""}"
         id="holo-tarjeta-${estado.id}" style="--glow:${estado.color}">
      <div class="holo-stage">
        <div class="holo-esfera" style="background: radial-gradient(circle at 32% 26%, ${estado.color}cc 0%, ${estado.color} 55%, #04141f 100%); box-shadow: 0 0 30px 6px ${estado.color}55, inset 0 0 20px ${estado.color}55;">
          <div class="holo-particulas"></div>
        </div>
        <div class="holo-base"></div>
      </div>
      <span class="holo-etiqueta" style="background:${estado.color}22; color:${estado.color}">${estado.nombre.toUpperCase()}</span>
      <div class="holo-temp" id="holo-temp-${estado.id}" style="color:${estado.color}">${temperaturasActuales[estado.id].toFixed(1)}°C</div>
      <p class="holo-desc">${estado.descripcion}</p>
    </div>
  `).join("");
}

function cambiarModoVisualizacion(modo) {
  modoVisualizacionActual = modo;
  document.querySelectorAll("#selector-modo button").forEach((boton) => {
    boton.classList.toggle("activo", boton.dataset.modo === modo);
  });
  renderMangos3D();
}

/** Aplica el resultado real de /diagnostico: resalta la clase predicha y actualiza su temperatura estimada. */
function actualizarClasificacionHolografica(diagnostico) {
  claseActivaId = diagnostico.etiqueta;
  temperaturasActuales[diagnostico.etiqueta] = diagnostico.temperatura_estimada;
  renderMangos3D();
}
