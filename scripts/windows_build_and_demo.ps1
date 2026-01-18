# Windows PowerShell helper for quick setup + smoke test
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\windows_build_and_demo.ps1

$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[hf,zstd]"

python .\scripts\smoke_test_roundtrip.py
Write-Host "[OK] Environment and smoke test completed." -ForegroundColor Green
