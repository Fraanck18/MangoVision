# =============================================================================
#  Despliegue de MangoVision a Hugging Face Spaces (Fraanck18/Mango_Vision)
#  Ensambla una carpeta limpia (solo lo necesario para inferencia) y la sube.
#  Requisitos: git y git-lfs instalados.  Ejecutar desde cualquier carpeta.
# =============================================================================

$ErrorActionPreference = "Stop"

$Origen  = "C:\Mango_Vision\MangoVision"                 # tu proyecto
$Clone   = "C:\Mango_Vision\Mango_Vision_HF"             # carpeta del repo HF
$RepoUrl = "https://huggingface.co/spaces/Fraanck18/Mango_Vision"

# --- 1. git-lfs ---
git lfs install

# --- 2. Clonar el Space (si no existe ya) ---
if (-not (Test-Path $Clone)) {
    git clone $RepoUrl $Clone
}
Set-Location $Clone

# --- 3. Copiar archivos de despliegue (con los nombres que HF espera) ---
Copy-Item "$Origen\app.py"                              "$Clone\app.py"          -Force
Copy-Item "$Origen\Dockerfile.hf"                       "$Clone\Dockerfile"      -Force
Copy-Item "$Origen\requirements-deploy.txt"             "$Clone\requirements.txt" -Force
Copy-Item "$Origen\README-HF.md"                        "$Clone\README.md"       -Force
Copy-Item "$Origen\.gitattributes-hf"                   "$Clone\.gitattributes"  -Force

# Codigo fuente (sin caches) y frontend
if (Test-Path "$Clone\src") { Remove-Item "$Clone\src" -Recurse -Force }
if (Test-Path "$Clone\web") { Remove-Item "$Clone\web" -Recurse -Force }
Copy-Item "$Origen\src" "$Clone\src" -Recurse -Force
Copy-Item "$Origen\web" "$Clone\web" -Recurse -Force
Get-ChildItem "$Clone\src" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Modelo entrenado (92 MB -> ira por LFS gracias a .gitattributes)
Copy-Item "$Origen\dataset_mangos\best_model.pth" "$Clone\best_model.pth" -Force

# --- 4. Commit + push ---
git add .gitattributes
git add -A
git commit -m "Despliegue MangoVision: FastAPI (solo inferencia) + frontend"
Write-Host "`nSubiendo a Hugging Face... (usa tu TOKEN con permiso de escritura como contrasena)" -ForegroundColor Cyan
git push

Write-Host "`nListo. Abre: $RepoUrl" -ForegroundColor Green
