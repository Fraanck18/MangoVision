/**
 * Galería de imágenes térmicas. A diferencia de la Vista 3D / holográfica
 * (que reinterpreta la imagen como relieve), aquí se muestran los JPEG REALES
 * que el sensor FLIR aterriza en el Data Lake (HDFS), tal cual llegan. Sirve
 * para verificar de un vistazo que las imágenes están llegando y con qué
 * calidad, antes de que el modelo las clasifique.
 *
 * Todo viene del backend FastAPI (ver apiClient.js):
 *   GET /imagenes/lotes            -> lotes aterrizados (con conteo)
 *   GET /imagenes/lote/{nombre}    -> imágenes del lote + metadata
 *   GET /imagenes/archivo?ruta=... -> bytes del JPEG real
 */
const estadoGaleria = {
  lotes: [],
  loteSeleccionado: null,
  imagenes: [],
  filtro: "todos",
};

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("btn-gal-actualizar");
  if (btn) btn.addEventListener("click", () => renderGaleria());

  document.querySelectorAll("#gal-filtro-tipo button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#gal-filtro-tipo button").forEach((x) => x.classList.remove("activo"));
      b.classList.add("activo");
      estadoGaleria.filtro = b.dataset.filtro;
      renderGridImagenes();
    });
  });

  // Lightbox
  const lightbox = document.getElementById("gal-lightbox");
  const cerrar = document.getElementById("gal-lightbox-cerrar");
  if (cerrar) cerrar.addEventListener("click", cerrarLightbox);
  if (lightbox) {
    lightbox.addEventListener("click", (ev) => { if (ev.target === lightbox) cerrarLightbox(); });
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") cerrarLightbox();
  });
});

/** Punto de entrada llamado por main.js al navegar a la vista "galeria". */
async function renderGaleria() {
  const listaLotes = document.getElementById("gal-lista-lotes");
  const ultimaAct = document.getElementById("gal-ultima-act");
  if (!listaLotes) return;

  listaLotes.innerHTML = `<span class="entrenamiento-info">Cargando lotes…</span>`;

  try {
    const { lotes } = await consultarLotesImagenes();
    estadoGaleria.lotes = lotes;
    ultimaAct.textContent = `Última actualización: ${new Date().toLocaleTimeString()}`;

    if (lotes.length === 0) {
      listaLotes.innerHTML = `<span class="entrenamiento-info">Todavía no hay imágenes en el Data Lake. El sensor del camión aterriza un lote nuevo cada pocos minutos.</span>`;
      renderGridVacio("Sin lotes recibidos todavía.");
      return;
    }

    renderListaLotes();

    // Selecciona el primer lote (más reciente) si no hay uno válido elegido.
    const sigueValido = lotes.some((l) => l.nombre === estadoGaleria.loteSeleccionado);
    await seleccionarLote(sigueValido ? estadoGaleria.loteSeleccionado : lotes[0].nombre);
  } catch (error) {
    listaLotes.innerHTML = `<span class="entrenamiento-info">No se pudieron cargar los lotes: ${error.message}</span>`;
    renderGridVacio("Backend o Data Lake (HDFS) sin conexión.");
  }
}

function renderListaLotes() {
  const cont = document.getElementById("gal-lista-lotes");
  cont.innerHTML = estadoGaleria.lotes.map((lote) => {
    const activo = lote.nombre === estadoGaleria.loteSeleccionado ? "activo" : "";
    const estado = lote.procesado ? "completado" : "en_progreso";
    const estadoTxt = lote.procesado ? "procesado" : "pendiente";
    return `
      <button type="button" class="gal-lote ${activo}" data-lote="${lote.nombre}">
        <div class="gal-lote-cab">
          <span class="gal-lote-nombre">${formatearEtiquetaLote(lote.etiqueta)}</span>
          <span class="estado-pill ${estado}">${estadoTxt}</span>
        </div>
        <span class="gal-lote-conteo">${lote.total_imagenes} imagen(es)</span>
      </button>`;
  }).join("");

  cont.querySelectorAll(".gal-lote").forEach((b) => {
    b.addEventListener("click", () => seleccionarLote(b.dataset.lote));
  });
}

async function seleccionarLote(nombreLote) {
  estadoGaleria.loteSeleccionado = nombreLote;
  document.querySelectorAll("#gal-lista-lotes .gal-lote").forEach((b) => {
    b.classList.toggle("activo", b.dataset.lote === nombreLote);
  });

  const grid = document.getElementById("gal-grid");
  grid.innerHTML = `<span class="entrenamiento-info">Cargando imágenes…</span>`;

  try {
    const { imagenes } = await consultarImagenesDeLote(nombreLote);
    estadoGaleria.imagenes = imagenes;
    renderGridImagenes();
  } catch (error) {
    estadoGaleria.imagenes = [];
    renderGridVacio(`No se pudieron cargar las imágenes: ${error.message}`);
  }
}

function imagenesFiltradas() {
  const f = estadoGaleria.filtro;
  if (f === "todos") return estadoGaleria.imagenes;
  if (f === "degradadas") return estadoGaleria.imagenes.filter((im) => im.degradada);
  return estadoGaleria.imagenes.filter((im) => im.aumentado === f);
}

function renderGridImagenes() {
  const grid = document.getElementById("gal-grid");
  const titulo = document.getElementById("gal-titulo-grid");
  const conteo = document.getElementById("gal-conteo");
  if (!grid) return;

  if (estadoGaleria.loteSeleccionado) {
    const etiqueta = formatearEtiquetaLote(estadoGaleria.loteSeleccionado.split("=")[1] || "");
    titulo.textContent = `🌡 Imágenes · ${etiqueta}`;
  }

  const imagenes = imagenesFiltradas();
  conteo.textContent = `${imagenes.length} imagen(es)`;

  if (imagenes.length === 0) {
    renderGridVacio("No hay imágenes que coincidan con el filtro.");
    return;
  }

  grid.innerHTML = imagenes.map((im, i) => {
    const url = urlImagenLote(im.url);
    const tagDegradada = im.degradada ? `<span class="gal-tag gal-tag-degradada">degradada</span>` : "";
    const tagFlip = im.aumentado === "horizontal_flip" ? `<span class="gal-tag gal-tag-flip">espejada</span>` : "";
    const clase = im.clase_real_origen ? `<span class="gal-card-clase">${im.clase_real_origen.replace(/_/g, " ")}</span>` : "";
    return `
      <figure class="gal-card" data-indice="${i}">
        <div class="gal-card-img">
          <img src="${url}" alt="${im.nombre_archivo}" loading="lazy">
          <div class="gal-card-tags">${tagDegradada}${tagFlip}</div>
        </div>
        <figcaption class="gal-card-pie">
          ${clase}
          <span class="gal-card-sensor">${im.sensor_id || "sensor —"}</span>
        </figcaption>
      </figure>`;
  }).join("");

  grid.querySelectorAll(".gal-card").forEach((card) => {
    card.addEventListener("click", () => abrirLightbox(imagenes[Number(card.dataset.indice)]));
  });
}

function renderGridVacio(mensaje) {
  const grid = document.getElementById("gal-grid");
  const conteo = document.getElementById("gal-conteo");
  if (grid) grid.innerHTML = `<div class="gal-vacio">${mensaje}</div>`;
  if (conteo) conteo.textContent = "—";
}

function abrirLightbox(imagen) {
  const lightbox = document.getElementById("gal-lightbox");
  const img = document.getElementById("gal-lightbox-img");
  const caption = document.getElementById("gal-lightbox-caption");
  if (!lightbox || !imagen) return;

  img.src = urlImagenLote(imagen.url);
  const partes = [
    imagen.nombre_archivo,
    imagen.clase_real_origen ? `clase: ${imagen.clase_real_origen.replace(/_/g, " ")}` : null,
    imagen.aumentado === "horizontal_flip" ? "espejada (flip horizontal)" : "original",
    imagen.degradada ? "⚠ degradada (posible fallo de sensor)" : null,
    imagen.sensor_id || null,
  ].filter(Boolean);
  caption.textContent = partes.join(" · ");

  lightbox.classList.remove("oculto");
}

function cerrarLightbox() {
  const lightbox = document.getElementById("gal-lightbox");
  if (lightbox) lightbox.classList.add("oculto");
}

/** "20240630T142530" -> "2024-06-30 14:25:30"; deja intactas las fechas simples. */
function formatearEtiquetaLote(etiqueta) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$/.exec(etiqueta || "");
  if (!m) return etiqueta || "—";
  return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}`;
}
