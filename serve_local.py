"""
Lanzador para EXPONER el sistema completo por un tunel publico (Cloudflare/ngrok).

Reutiliza el backend REAL tal cual (src/api/app.py, con HDFS + Kafka + Spark +
scheduler + evaluador online) y solo le AÑADE:
  1) el servido del frontend estatico web/ en el mismo origen, y
  2) persistencia de la data de clasificacion en MongoDB Atlas (middleware +
     endpoint /clasificaciones).
No modifica la arquitectura ni el app.py original.

Uso:
    $env:MONGO_URI = "mongodb+srv://usuario:pass@cluster0.xxxx.mongodb.net/..."
    uvicorn serve_local:app --host 0.0.0.0 --port 8000
"""
import json
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.api.app import app  # backend completo (arranca scheduler/evaluador en startup)
import mongo_persistencia

WEB_DIR = Path(__file__).resolve().parent / "web"


class PersistenciaMongoMiddleware(BaseHTTPMiddleware):
    """
    Captura las respuestas de clasificacion y las guarda en MongoDB, sin que
    src/api/app.py se entere. Si Mongo falla, NUNCA rompe la peticion (la capa
    de persistencia degrada silenciosamente).
    """

    async def dispatch(self, request, call_next):
        respuesta = await call_next(request)

        ruta = request.url.path
        if ruta not in ("/diagnostico", "/hdfs/analizar") or respuesta.status_code != 200:
            return respuesta

        # Consumir el cuerpo para poder leerlo y luego reconstruir la respuesta.
        cuerpo = b""
        async for chunk in respuesta.body_iterator:
            cuerpo += chunk

        try:
            datos = json.loads(cuerpo)
            if ruta == "/diagnostico":
                await run_in_threadpool(mongo_persistencia.guardar_clasificacion, datos, "manual")
            else:  # /hdfs/analizar -> lista de resultados de lote
                for resultado in datos.get("resultados", []):
                    await run_in_threadpool(mongo_persistencia.guardar_clasificacion, resultado, "lote")
        except Exception:
            pass  # nunca romper la respuesta por un fallo de persistencia

        return Response(
            content=cuerpo,
            status_code=respuesta.status_code,
            headers=dict(respuesta.headers),
            media_type=respuesta.media_type,
        )


app.add_middleware(PersistenciaMongoMiddleware)


@app.get("/clasificaciones")
def listar_clasificaciones(limite: int = 50):
    """Ultimas clasificaciones persistidas en MongoDB (evidencia de la data guardada)."""
    return mongo_persistencia.ultimas_clasificaciones(limite)


# Se monta al final: las rutas de API (incluida /clasificaciones) ya estan
# registradas y tienen prioridad sobre este catch-all de archivos estaticos.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
