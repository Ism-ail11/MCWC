param(
    [string]$venvPath = ".venv"
)

Write-Host "Creating virtual environment at $venvPath"
python -m venv $venvPath

Write-Host "Upgrading pip in the virtual environment"
& "$venvPath\Scripts\python.exe" -m pip install -U pip

Write-Host "Installing package in editable mode and developer requirements"
& "$venvPath\Scripts\python.exe" -m pip install -e .
if (Test-Path "dev-requirements.txt") {
    & "$venvPath\Scripts\python.exe" -m pip install -r dev-requirements.txt
}

Write-Host "Setup complete. To use the venv in PowerShell run: .\$venvPath\Scripts\Activate.ps1"