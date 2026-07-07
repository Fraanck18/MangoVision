/**
 * Visor holografico 3D interactivo (Three.js + WebGL) para la firma termica del
 * mango. Reemplaza las esferas CSS estaticas por un modelo tridimensional real
 * navegable en XYZ (OrbitControls: arrastrar = rotar, rueda = zoom, clic
 * derecho = desplazar).
 *
 * Tecnica (adaptada del modelo THERMO-HOLO al contexto MangoVision):
 *  1. Firma termica procedural: se dibuja en un canvas 2D oculto un mapa de
 *     calor por clase de madurez (verde frio/homogeneo, maduro con gradiente
 *     radial, sobre-maduro con hot-spots), coherente con el generador fisico
 *     de scripts/generate_synthetic_thermal.py. Si el usuario importa una
 *     imagen real y la analiza, esa imagen se usa como mapa de calor real.
 *  2. Z-Depth Mapping: el brillo de cada celda (0..1) se eleva en el eje Y,
 *     formando relieves termicos -> los hot-spots del sobre-maduro se ven como
 *     montañas, las zonas frias del verde quedan planas.
 *  3. Modos de representacion (mapeados a los toggles Espectro IR / Holografico
 *     / Diferencial que ya existian en la UI):
 *       - holografico  -> nube de puntos (THREE.Points) teñida por la clase.
 *       - ir           -> malla solida con paleta Ironbow por altura.
 *       - diferencial  -> topografia: malla oscura + wireframe neon encima.
 *  4. Los 3 estados de madurez actuan como los 3 estados de alerta: la clase
 *     predicha por el backend tiñe el holograma y, en sobre-maduro, se añade
 *     un leve jitter de "interferencia termica".
 */
import * as THREE from "three";
import { OrbitControls } from "./../vendor/OrbitControls.js";

// [F10] Rangos en ITR relativo 0-100 (NO °C). Coinciden con las bandas de la
// escala unica (escala.js / MATURITY_INFO). El backend no mide grados Celsius.
const ESTADOS = {
  calidad_promedio: { nombre: "Verde / Inmaduro", color: 0x2f6fed, perfil: "verde", rango: "ITR 15–35",
    desc: "Firma térmica baja y homogénea: intensidad relativa baja, sin actividad de maduración significativa." },
  alta_calidad:     { nombre: "Maduro / Óptimo",  color: 0x16a34a, perfil: "maduro", rango: "ITR 40–70",
    desc: "Gradiente radial moderado con centro de mayor intensidad relativa: punto óptimo de consumo." },
  defectuoso:       { nombre: "Sobre-Maduro",     color: 0xdc2626, perfil: "sobremaduro", rango: "ITR 75–100",
    desc: "Hot-spots locales de alta intensidad relativa: anomalías térmicas detectadas." },
};

// Paleta Ironbow (igual que src/domain_physics/thermal_inertia.py): frio -> caliente.
const IRONBOW = [
  [0.00, [0.00, 0.00, 0.25]],
  [0.20, [0.20, 0.00, 0.55]],
  [0.45, [0.75, 0.10, 0.55]],
  [0.65, [1.00, 0.45, 0.10]],
  [0.85, [1.00, 0.85, 0.15]],
  [1.00, [1.00, 1.00, 1.00]],
];

const RES = 80;            // resolucion de la malla termica (RES x RES vertices)
const TAM = 10;            // tamaño del plano en unidades de mundo

const estado3d = {
  escena: null, camara: null, renderer: null, controles: null,
  objeto: null, animId: null, reloj: null,
  modo: "holografico",
  claseActiva: "alta_calidad",
  escalaZ: 1.4,
  autoRotar: true,
  alturas: null,            // Float32Array RES*RES con el heightmap [0..1] actual
  fpsUltimo: 0, fpsContador: 0, fpsAcum: 0,
  iniciado: false,
};


// ---------------------------------------------------------------------------
// 1. Generacion de la firma termica (heightmap [0..1])
// ---------------------------------------------------------------------------
function generarFirmaProcedural(perfil) {
  const alturas = new Float32Array(RES * RES);
  // hot-spots aleatorios para sobre-maduro (anomalias termicas)
  const hotspots = [];
  if (perfil === "sobremaduro") {
    const n = 2 + Math.floor(Math.random() * 2);
    for (let k = 0; k < n; k++) {
      hotspots.push({ x: (Math.random() - 0.5) * 0.9, y: (Math.random() - 0.5) * 0.9, s: 0.10 + Math.random() * 0.12 });
    }
  }
  for (let j = 0; j < RES; j++) {
    for (let i = 0; i < RES; i++) {
      const nx = (i / (RES - 1)) * 2 - 1;
      const ny = (j / (RES - 1)) * 2 - 1;
      const r = Math.sqrt((nx * nx) / (0.82 * 0.82) + (ny * ny) / (0.74 * 0.74));
      let h = 0.01 + Math.random() * 0.005;  // fondo frio limpio y casi plano (sin bailar)
      if (r <= 1) {
        const base_shape = Math.sqrt(1 - r * r);
        if (perfil === "verde") {
          const noise = (Math.random() - 0.5) * 0.01;
          h = (0.30 - 0.06 * r + noise) * base_shape;
        } else if (perfil === "maduro") {
          const noise = (Math.random() - 0.5) * 0.015;
          h = (0.62 * (1 - r * r) + 0.22 + noise) * base_shape;
        } else {
          const noise = (Math.random() - 0.5) * 0.02;
          let base_h = 0.55 * (1 - r * r) + 0.30 + noise;
          for (const hs of hotspots) {
            const d2 = ((nx - hs.x) ** 2 + (ny - hs.y) ** 2) / (2 * hs.s * hs.s);
            base_h += 0.75 * Math.exp(-d2);                                   // picos (hot-spots)
          }
          h = base_h * base_shape;
        }
      }
      // Quitamos el límite superior (clamp en 1) para que no se vea cortado arriba
      alturas[j * RES + i] = Math.max(0, h);
    }
  }
  return alturas;
}

/** Extrae un heightmap [0..1] de la luminancia de una imagen real (la importada por el usuario). */
function generarFirmaDesdeImagen(imagen) {
  const cv = document.createElement("canvas");
  cv.width = RES; cv.height = RES;
  const ctx = cv.getContext("2d");
  ctx.drawImage(imagen, 0, 0, RES, RES);
  const data = ctx.getImageData(0, 0, RES, RES).data;
  const alturas = new Float32Array(RES * RES);
  for (let p = 0; p < RES * RES; p++) {
    const r = data[p * 4], g = data[p * 4 + 1], b = data[p * 4 + 2];
    // Luminancia perceptual -> altura. En la paleta Ironbow, mas brillo = mas calor.
    alturas[p] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  }
  return alturas;
}


// ---------------------------------------------------------------------------
// 2. Utilidades de color
// ---------------------------------------------------------------------------
function colorIronbow(t) {
  t = Math.max(0, Math.min(1, t));
  for (let k = 0; k < IRONBOW.length - 1; k++) {
    const [v0, c0] = IRONBOW[k], [v1, c1] = IRONBOW[k + 1];
    if (t >= v0 && t <= v1) {
      const f = (t - v0) / (v1 - v0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return IRONBOW[IRONBOW.length - 1][1];
}


// ---------------------------------------------------------------------------
// 3. Construccion del objeto 3D segun el modo activo
// ---------------------------------------------------------------------------
function _posiciones(alturas, escalaZ) {
  const pos = new Float32Array(RES * RES * 3);
  for (let j = 0; j < RES; j++) {
    for (let i = 0; i < RES; i++) {
      const idx = j * RES + i;
      pos[idx * 3]     = (i / (RES - 1) - 0.5) * TAM;          // X
      pos[idx * 3 + 1] = alturas[idx] * escalaZ * (TAM * 0.28); // Y (elevacion = calor)
      pos[idx * 3 + 2] = (j / (RES - 1) - 0.5) * TAM;          // Z
    }
  }
  return pos;
}

function _indicesGrid() {
  const indices = [];
  for (let j = 0; j < RES - 1; j++) {
    for (let i = 0; i < RES - 1; i++) {
      const a = j * RES + i, b = a + 1, c = a + RES, d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  return indices;
}

function construirObjeto() {
  const alturas = estado3d.alturas;
  const pos = _posiciones(alturas, estado3d.escalaZ);
  const colorClase = new THREE.Color(ESTADOS[estado3d.claseActiva].color);

  if (estado3d.modo === "holografico") {
    // Nube de puntos teñida por la clase, con glow aditivo (holograma flotante).
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const colores = new Float32Array(RES * RES * 3);
    for (let p = 0; p < RES * RES; p++) {
      const brillo = 0.45 + Math.min(1.0, alturas[p]) * 0.55;
      colores[p * 3] = colorClase.r * brillo;
      colores[p * 3 + 1] = colorClase.g * brillo;
      colores[p * 3 + 2] = colorClase.b * brillo;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colores, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.12, vertexColors: true, transparent: true, opacity: 0.92,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    return new THREE.Points(geo, mat);
  }

  if (estado3d.modo === "ir") {
    // Malla solida con paleta Ironbow por altura (lectura cientifica de calor).
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(_indicesGrid());
    geo.computeVertexNormals();
    const colores = new Float32Array(RES * RES * 3);
    for (let p = 0; p < RES * RES; p++) {
      const [r, g, b] = colorIronbow(alturas[p]);
      colores[p * 3] = r; colores[p * 3 + 1] = g; colores[p * 3 + 2] = b;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colores, 3));
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.55, metalness: 0.15,
      flatShading: true, side: THREE.DoubleSide,
    });
    return new THREE.Mesh(geo, mat);
  }

  // diferencial -> topografia: superficie oscura + wireframe neon de la clase encima.
  const grupo = new THREE.Group();
  const geoBase = new THREE.BufferGeometry();
  geoBase.setAttribute("position", new THREE.BufferAttribute(pos.slice(), 3));
  geoBase.setIndex(_indicesGrid());
  geoBase.computeVertexNormals();
  grupo.add(new THREE.Mesh(geoBase, new THREE.MeshStandardMaterial({
    color: 0x0a1018, roughness: 0.9, metalness: 0.0, side: THREE.DoubleSide,
  })));
  const geoWire = new THREE.BufferGeometry();
  geoWire.setAttribute("position", new THREE.BufferAttribute(pos.slice(), 3));
  geoWire.setIndex(_indicesGrid());
  grupo.add(new THREE.LineSegments(
    new THREE.WireframeGeometry(geoWire),
    new THREE.LineBasicMaterial({ color: colorClase, transparent: true, opacity: 0.55 }),
  ));
  return grupo;
}

function reconstruirObjeto() {
  if (!estado3d.escena) return;
  if (estado3d.objeto) {
    estado3d.escena.remove(estado3d.objeto);
    estado3d.objeto.traverse?.((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    estado3d.objeto.geometry?.dispose?.();
    estado3d.objeto.material?.dispose?.();
  }
  estado3d.objeto = construirObjeto();
  estado3d.escena.add(estado3d.objeto);
}


// ---------------------------------------------------------------------------
// 4. Inicializacion de la escena y bucle de render
// ---------------------------------------------------------------------------
function iniciarEscena() {
  const cont = document.getElementById("holo-canvas-3d");
  if (!cont || estado3d.iniciado) return;
  estado3d.iniciado = true;

  const ancho = cont.clientWidth || 600;
  const alto = cont.clientHeight || 380;

  estado3d.escena = new THREE.Scene();
  estado3d.escena.fog = new THREE.FogExp2(0x05080f, 0.018);

  estado3d.camara = new THREE.PerspectiveCamera(45, ancho / alto, 0.1, 100);
  estado3d.camara.position.set(7, 10, 14);

  estado3d.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  estado3d.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  estado3d.renderer.setSize(ancho, alto);
  cont.appendChild(estado3d.renderer.domElement);

  estado3d.controles = new OrbitControls(estado3d.camara, estado3d.renderer.domElement);
  estado3d.controles.enableDamping = true;
  estado3d.controles.dampingFactor = 0.08;
  estado3d.controles.minDistance = 6;
  estado3d.controles.maxDistance = 30;
  estado3d.controles.target.set(0, 1, 0);

  estado3d.escena.add(new THREE.AmbientLight(0xffffff, 0.7));
  const luz = new THREE.DirectionalLight(0xffffff, 1.1);
  luz.position.set(6, 12, 8);
  estado3d.escena.add(luz);

  // Rejilla base estilo HUD
  const grid = new THREE.GridHelper(TAM * 1.6, 24, 0x1c3a52, 0x12222f);
  grid.position.y = -0.2;
  estado3d.escena.add(grid);

  estado3d.reloj = new THREE.Clock();
  estado3d.alturas = generarFirmaProcedural(ESTADOS[estado3d.claseActiva].perfil);
  reconstruirObjeto();

  const ro = new ResizeObserver(() => ajustarTamano());
  ro.observe(cont);

  bucle();
}

function ajustarTamano() {
  const cont = document.getElementById("holo-canvas-3d");
  if (!cont || !estado3d.renderer) return;
  const ancho = cont.clientWidth, alto = cont.clientHeight;
  if (ancho === 0 || alto === 0) return;
  estado3d.camara.aspect = ancho / alto;
  estado3d.camara.updateProjectionMatrix();
  estado3d.renderer.setSize(ancho, alto);
}

function bucle() {
  estado3d.animId = requestAnimationFrame(bucle);
  const dt = estado3d.reloj.getDelta();

  if (estado3d.autoRotar && estado3d.objeto) {
    estado3d.objeto.rotation.y += dt * 0.25;
  }
  // Jitter de "interferencia termica" desactivado para evitar que el modelo baile/salte
  if (estado3d.objeto) {
    estado3d.objeto.position.y = 0;
  }

  estado3d.controles.update();
  estado3d.renderer.render(estado3d.escena, estado3d.camara);
  actualizarHUD(dt);
}

function actualizarHUD(dt) {
  estado3d.fpsAcum += dt; estado3d.fpsContador++;
  if (estado3d.fpsAcum >= 0.5) {
    estado3d.fpsUltimo = Math.round(estado3d.fpsContador / estado3d.fpsAcum);
    estado3d.fpsAcum = 0; estado3d.fpsContador = 0;
    const hud = document.getElementById("holo-hud-fps");
    if (hud) hud.textContent = `${estado3d.fpsUltimo} FPS`;
  }
}


// ---------------------------------------------------------------------------
// 5. API global (consumida por main.js, que es script clasico)
// ---------------------------------------------------------------------------
function renderMangos3D() {
  iniciarEscena();
  sincronizarSelectorClase();
}

function cambiarModoVisualizacion(modo) {
  estado3d.modo = modo;
  document.querySelectorAll("#selector-modo button").forEach((b) => {
    b.classList.toggle("activo", b.dataset.modo === modo);
  });
  // Sincronizar el panel explicativo de modos de visualización
  document.querySelectorAll(".modo-desc-item").forEach((item) => {
    item.classList.toggle("activo", item.id === `desc-modo-${modo}`);
  });
  reconstruirObjeto();
}

function seleccionarClase3D(claseId) {
  if (!ESTADOS[claseId]) return;
  estado3d.claseActiva = claseId;
  estado3d.alturas = generarFirmaProcedural(ESTADOS[claseId].perfil);
  reconstruirObjeto();
  sincronizarSelectorClase();
}

/** Llamado tras /diagnostico: tiñe el holograma con la clase predicha. */
function actualizarClasificacionHolografica(diagnostico) {
  if (diagnostico && diagnostico.etiqueta) seleccionarClase3D(diagnostico.etiqueta);
}

/** Carga la imagen REAL importada por el usuario como mapa de calor 3D. */
function cargarImagenHolograma(url) {
  const img = new Image();
  img.onload = () => {
    estado3d.alturas = generarFirmaDesdeImagen(img);
    reconstruirObjeto();
  };
  img.src = url;
}

function setEscalaZ(valor) {
  estado3d.escalaZ = parseFloat(valor);
  reconstruirObjeto();
}

function setAutoRotar(activo) {
  estado3d.autoRotar = !!activo;
}

function sincronizarSelectorClase() {
  document.querySelectorAll("#holo-clases button").forEach((b) => {
    b.classList.toggle("activo", b.dataset.clase === estado3d.claseActiva);
  });
  const panel = document.getElementById("holo-info-estado");
  if (panel) {
    const e = ESTADOS[estado3d.claseActiva];
    const hex = "#" + e.color.toString(16).padStart(6, "0");
    panel.innerHTML = `
      <div class="holo-info-nombre" style="color:${hex}">${e.nombre}</div>
      <div class="holo-info-rango">Rango térmico: ${e.rango}</div>
      <p class="holo-info-desc">${e.desc}</p>`;
  }
}

// Exponer a main.js (script clasico cargado despues; este modulo defer ya
// ejecuto sus asignaciones para cuando dispare DOMContentLoaded).
window.renderMangos3D = renderMangos3D;
window.cambiarModoVisualizacion = cambiarModoVisualizacion;
window.actualizarClasificacionHolografica = actualizarClasificacionHolografica;
window.cargarImagenHolograma = cargarImagenHolograma;
window.seleccionarClase3D = seleccionarClase3D;
window.setEscalaZ3D = setEscalaZ;
window.setAutoRotar3D = setAutoRotar;
window.ajustarVisor3D = ajustarTamano;
