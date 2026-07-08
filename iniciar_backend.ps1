# Lanza el backend completo (HDFS/Kafka/Spark via src/api/app.py) + frontend +
# persistencia MongoDB. MONGO_URI se toma de la variable de usuario permanente
# (ya configurada con setx). Ejecutar en una terminal NUEVA.
Set-Location $PSScriptRoot
# Si la sesion no heredo la variable, la leo directo del entorno de USUARIO
# (la que se guardo con setx). Asi funciona aunque la terminal sea vieja.
if (-not $env:MONGO_URI) {
    $env:MONGO_URI = [System.Environment]::GetEnvironmentVariable("MONGO_URI", "User")
}
Write-Host "MONGO_URI detectada: $([bool]$env:MONGO_URI)" -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn serve_local:app --host 0.0.0.0 --port 8000
