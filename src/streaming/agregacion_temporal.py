"""
Tecnica #3 — Agregacion TEMPORAL / multi-frame en el stream.

El sensor termico entrega MUCHOS frames por segundo del mismo fruto mientras pasa
por la cinta. Ese exceso de frames es una fuente de robustez que hoy se
desperdicia clasificando frame a frame. Aqui se explota de dos formas:

  1. FRAME AVERAGING (denoiser): promediar N frames consecutivos del mismo fruto
     reduce el ruido NETD del detector (~1/sqrt(N)) antes de clasificar. Sube el
     SNR de la entrada sin coste de entrenamiento.

  2. VOTING TEMPORAL / EMA de logits: en vez de creer a un unico frame (ruidoso),
     se agrega la prediccion sobre una ventana deslizante por fruto. Dos modos:
       * media/EMA de probabilidades -> decision = argmax de la media,
       * voto por mayoria de las clases predichas.
     Reduce la varianza de decision y evita parpadeos de clase entre frames.

Todo es puro post-proceso del stream: no reentrena el modelo. Encaja en la capa
Kafka/streaming (evaluador online) o entre el pipeline Spark y el resultado curado.
"""
from collections import defaultdict, deque

import numpy as np


def promediar_frames(frames):
    """
    FRAME AVERAGING: promedia una lista de imagenes (arrays HxWx3 o PIL) del MISMO
    fruto en un unico frame denoised (uint8 RGB). Requiere que esten alineadas
    (mismo encuadre, como en una cinta con encoder de posicion). Reduce el ruido
    del detector ~1/sqrt(N).
    """
    if not frames:
        raise ValueError("Se requiere al menos un frame para promediar.")
    arrs = [np.asarray(f, dtype=np.float32) for f in frames]
    forma = arrs[0].shape
    if any(a.shape != forma for a in arrs):
        raise ValueError("Todos los frames deben tener la misma forma para promediar.")
    return np.clip(np.mean(arrs, axis=0), 0, 255).astype(np.uint8)


class AgregadorTemporal:
    """
    Mantiene una ventana deslizante de PROBABILIDADES por fruto y entrega una
    decision agregada (mas estable que la de un frame suelto).

    ventana        -> nº de frames recientes por fruto que se agregan.
    modo           -> 'ema' (media exponencial), 'media' (media simple) o 'voto'
                      (voto por mayoria del argmax de cada frame).
    ema_alpha      -> factor de la EMA (solo modo 'ema'); mayor = mas reactivo.
    """

    def __init__(self, num_clases, ventana=8, modo="ema", ema_alpha=0.4):
        if modo not in ("ema", "media", "voto"):
            raise ValueError(f"modo desconocido: {modo!r}")
        self.num_clases = int(num_clases)
        self.ventana = int(ventana)
        self.modo = modo
        self.ema_alpha = float(ema_alpha)
        self._historial = defaultdict(lambda: deque(maxlen=self.ventana))
        self._ema = {}

    def actualizar(self, id_fruto, probabilidades):
        """
        Registra las probabilidades de un frame para `id_fruto` y devuelve la
        decision agregada actual: (clase_id, confianza_agregada, n_frames).
        `probabilidades` es un vector/array de longitud num_clases (suma 1).
        """
        probs = np.asarray(probabilidades, dtype=np.float64).ravel()
        if probs.size != self.num_clases:
            raise ValueError(f"Se esperaban {self.num_clases} probabilidades, "
                             f"llegaron {probs.size}.")
        self._historial[id_fruto].append(probs)

        if self.modo == "ema":
            previa = self._ema.get(id_fruto)
            actual = probs if previa is None else self.ema_alpha * probs + (1 - self.ema_alpha) * previa
            self._ema[id_fruto] = actual
            agregada = actual
        elif self.modo == "media":
            agregada = np.mean(self._historial[id_fruto], axis=0)
        else:  # voto por mayoria del argmax de cada frame
            votos = np.bincount([int(np.argmax(p)) for p in self._historial[id_fruto]],
                                minlength=self.num_clases)
            agregada = votos / votos.sum()

        clase_id = int(np.argmax(agregada))
        return clase_id, float(agregada[clase_id]), len(self._historial[id_fruto])

    def reiniciar(self, id_fruto=None):
        """Olvida el historial de un fruto (cuando termina de pasar) o de todos."""
        if id_fruto is None:
            self._historial.clear()
            self._ema.clear()
        else:
            self._historial.pop(id_fruto, None)
            self._ema.pop(id_fruto, None)
