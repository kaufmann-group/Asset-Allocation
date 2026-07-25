# check to see if python is installed! `python --version` if not install as `winget install -e --id Python.Python.3.11`
# run as `powershell -ExecutionPolicy Bypass -File .\setup.ps1`

$ErrorActionPreference = "Stop"

# Get the project root folder (one level up from this script)
$REPO = Resolve-Path "$PSScriptRoot\.."
Set-Location $REPO

python -m venv "$REPO\ocean"

& "$REPO\ocean\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install -r "$REPO\scripts\requirements.txt"

python -m ipykernel install --user --name=repo-env --display-name "Asset Allocation Repository Environment"

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate the environment from the project root with:"
Write-Host ".\ocean\Scripts\Activate.bat"
