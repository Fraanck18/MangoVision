/**
 * Utilidades de UX compartidas: animaciones FLIP, drawer lateral,
 * lista virtualizada y densidad de información. Sin dependencias externas
 * a proposito -- el proyecto es vanilla JS y no amerita traer una libreria
 * de animacion solo para esto.
 */

// ---------------------------------------------------------------------------
// FLIP (First, Last, Invert, Play): evita que las filas "salten" al filtrar.
// Los elementos a animar deben tener un atributo data-flip-key estable.
// ---------------------------------------------------------------------------
function flipRender(contenedor, actualizarFn) {
  const selector = "[data-flip-key]";
  const antes = new Map();
  contenedor.querySelectorAll(selector).forEach((el) => {
    antes.set(el.dataset.flipKey, el.getBoundingClientRect());
  });

  actualizarFn();

  contenedor.querySelectorAll(selector).forEach((el) => {
    const rectAntes = antes.get(el.dataset.flipKey);

    if (!rectAntes) {
      // Elemento nuevo: entra con fade + slide en vez de aparecer de golpe.
      el.style.opacity = "0";
      el.style.transform = "translateY(6px)";
      requestAnimationFrame(() => {
        el.style.transition = "opacity .25s ease, transform .25s ease";
        el.style.opacity = "1";
        el.style.transform = "translateY(0)";
      });
      return;
    }

    const rectDespues = el.getBoundingClientRect();
    const dx = rectAntes.left - rectDespues.left;
    const dy = rectAntes.top - rectDespues.top;
    if (dx || dy) {
      el.style.transition = "none";
      el.style.transform = `translate(${dx}px, ${dy}px)`;
      requestAnimationFrame(() => {
        el.style.transition = "transform .3s ease";
        el.style.transform = "";
      });
    }
  });
}

// ---------------------------------------------------------------------------
// Drawer lateral: detalle de un registro sin abandonar el dashboard.
// ---------------------------------------------------------------------------
function abrirDrawer(titulo, htmlContenido) {
  const drawer = document.getElementById("drawer");
  const overlay = document.getElementById("drawer-overlay");
  document.getElementById("drawer-titulo").textContent = titulo;
  document.getElementById("drawer-contenido").innerHTML = htmlContenido;

  drawer.classList.remove("oculto");
  overlay.classList.remove("oculto");
  requestAnimationFrame(() => {
    drawer.classList.add("abierto");
    overlay.classList.add("abierto");
  });
}

function cerrarDrawer() {
  const drawer = document.getElementById("drawer");
  const overlay = document.getElementById("drawer-overlay");
  if (!drawer || drawer.classList.contains("oculto")) return;
  drawer.classList.remove("abierto");
  overlay.classList.remove("abierto");
  setTimeout(() => {
    drawer.classList.add("oculto");
    overlay.classList.add("oculto");
  }, 220);
}

function inicializarDrawer() {
  document.getElementById("drawer-overlay")?.addEventListener("click", cerrarDrawer);
  document.getElementById("drawer-cerrar")?.addEventListener("click", cerrarDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") cerrarDrawer(); });
}

// ---------------------------------------------------------------------------
// Lista virtualizada: solo renderiza las filas visibles + un buffer.
// Pensada para listas que pueden crecer a miles de registros (historial,
// logs). Con pocos items simplemente renderiza todos sin costo extra.
// ---------------------------------------------------------------------------
function crearListaVirtual(contenedor, items, alturaItem, renderItem) {
  contenedor.innerHTML = "";
  contenedor.classList.add("lista-virtual");

  if (items.length === 0) return;

  const spacer = document.createElement("div");
  spacer.className = "lista-virtual-spacer";
  spacer.style.height = (items.length * alturaItem) + "px";
  contenedor.appendChild(spacer);

  const BUFFER = 4;
  function render() {
    const scrollTop = contenedor.scrollTop;
    const alturaVisible = contenedor.clientHeight || 260;
    const primero = Math.max(0, Math.floor(scrollTop / alturaItem) - BUFFER);
    const ultimo = Math.min(items.length, Math.ceil((scrollTop + alturaVisible) / alturaItem) + BUFFER);

    spacer.innerHTML = "";
    for (let i = primero; i < ultimo; i++) {
      const el = renderItem(items[i], i);
      el.style.position = "absolute";
      el.style.top = (i * alturaItem) + "px";
      el.style.left = "0";
      el.style.right = "0";
      spacer.appendChild(el);
    }
  }

  contenedor.addEventListener("scroll", render);
  render();
}

// ---------------------------------------------------------------------------
// Densidad de informacion: "Comoda" (default) vs "Compacta" (estilo Excel).
// ---------------------------------------------------------------------------
function aplicarDensidad(densidad) {
  document.body.dataset.densidad = densidad;
  const boton = document.getElementById("btn-densidad");
  if (boton) boton.textContent = densidad === "compacta" ? "▦ Vista cómoda" : "▤ Vista compacta";
}

function inicializarDensidad() {
  const boton = document.getElementById("btn-densidad");
  if (!boton) return;
  const guardada = localStorage.getItem("thermomango_densidad") || "comoda";
  aplicarDensidad(guardada);
  boton.addEventListener("click", () => {
    const siguiente = document.body.dataset.densidad === "compacta" ? "comoda" : "compacta";
    aplicarDensidad(siguiente);
    localStorage.setItem("thermomango_densidad", siguiente);
  });
}
