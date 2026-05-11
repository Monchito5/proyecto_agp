# Script para crear entorno virtual e instalar dependencias en Windows

$ENV_DIR = "splicing_env"

Write-Host "Creando entorno virtual en .\$ENV_DIR ..."
python -m venv $ENV_DIR

Write-Host "Activando entorno..."
& ".\$ENV_DIR\Scripts\Activate.ps1"

Write-Host "Actualizando pip..."
python -m pip install --upgrade pip

Write-Host "Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

Write-Host "Entorno listo"
Write-Host "Para activarlo en futuras sesiones, ejecuta: .\$ENV_DIR\Scripts\Activate.ps1"