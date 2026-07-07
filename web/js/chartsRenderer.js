/**
 * Histograma de distribucion termica por estado, dibujado en <canvas> sin
 * librerias externas. [F10] La escala es ITR relativo 0-100 (NO °C): buckets y
 * colores vienen de la FUENTE UNICA `escala.js` (BUCKETS_ITR, colorPorITR), que
 * refleja MATURITY_INFO del backend. Antes usaba buckets °C [15..48] y un ITR de
 * 75-100 (defectuoso) se salia del ultimo bucket -> se descoloreaba.
 */
let conteosBuckets = new Array(BUCKETS_ITR.length - 1).fill(0);

function acumularMuestraGrafica(itr) {
  for (let i = 0; i < BUCKETS_ITR.length - 1; i += 1) {
    if (itr >= BUCKETS_ITR[i] && itr < BUCKETS_ITR[i + 1]) {
      conteosBuckets[i] += 1;
      break;
    }
  }
  // El ultimo bucket es cerrado por la derecha (incluye ITR = 100).
  if (itr >= BUCKETS_ITR[BUCKETS_ITR.length - 1]) conteosBuckets[conteosBuckets.length - 1] += 1;
  renderGraficaDistribucion();
}

function renderGraficaDistribucion() {
  const canvas = document.getElementById("grafica-distribucion");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const ancho = canvas.width;
  const alto = canvas.height;
  ctx.clearRect(0, 0, ancho, alto);

  const margenInferior = 28;
  const maxConteo = Math.max(1, ...conteosBuckets);
  const anchoBarra = (ancho - 40) / conteosBuckets.length;

  conteosBuckets.forEach((conteo, i) => {
    const x = 30 + i * anchoBarra;
    const alturaBarra = (conteo / maxConteo) * (alto - margenInferior - 14);
    const y = alto - margenInferior - alturaBarra;
    const itrMedio = (BUCKETS_ITR[i] + BUCKETS_ITR[i + 1]) / 2;

    ctx.fillStyle = conteo > 0 ? colorPorITR(itrMedio) : "#e3e9e6";
    ctx.beginPath();
    ctx.roundRect(x, y, anchoBarra - 8, Math.max(alturaBarra, 4), 4);
    ctx.fill();

    ctx.fillStyle = "#6b7d76";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${BUCKETS_ITR[i]}`, x + (anchoBarra - 8) / 2, alto - 12);
  });
}
