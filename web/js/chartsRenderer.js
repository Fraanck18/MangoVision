/**
 * Histograma de distribucion termica por estado, dibujado en <canvas> sin
 * librerias externas. Las barras se colorean segun el rango de temperatura
 * (azul=inmaduro, verde=optimo, rojo=sobre-maduro) y se recalculan cada vez
 * que llega un resultado real de /diagnostico (ver acumularMuestraGrafica).
 */
const BUCKETS_TEMP = [15, 17, 19, 24, 28, 31, 34, 37, 48];
let conteosBuckets = new Array(BUCKETS_TEMP.length - 1).fill(0);

function colorParaTemperatura(temp) {
  if (temp <= 21) return "#2f6fed";
  if (temp <= 32) return "#1f9d6b";
  return "#dc2626";
}

function acumularMuestraGrafica(temperatura) {
  for (let i = 0; i < BUCKETS_TEMP.length - 1; i += 1) {
    if (temperatura >= BUCKETS_TEMP[i] && temperatura < BUCKETS_TEMP[i + 1]) {
      conteosBuckets[i] += 1;
      break;
    }
  }
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
    const tempMedia = (BUCKETS_TEMP[i] + BUCKETS_TEMP[i + 1]) / 2;

    ctx.fillStyle = conteo > 0 ? colorParaTemperatura(tempMedia) : "#e3e9e6";
    ctx.beginPath();
    ctx.roundRect(x, y, anchoBarra - 8, Math.max(alturaBarra, 4), 4);
    ctx.fill();

    ctx.fillStyle = "#6b7d76";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${BUCKETS_TEMP[i]}°`, x + (anchoBarra - 8) / 2, alto - 12);
  });
}
