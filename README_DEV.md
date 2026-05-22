## Development setup

This file explains how to set up a development environment for the MCWC reference
implementation and run the smoke tests.

Windows (PowerShell):

```powershell
.
# create venv and install deps
.\scripts\setup_env.ps1

# activate
.\.venv\Scripts\Activate.ps1

# run smoke test (requires PyTorch)
python scripts/smoke_test_roundtrip.py

# run unit tests
python -m pytest -q
```

Linux / macOS:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
python scripts/smoke_test_roundtrip.py
python -m pytest -q
```

Notes:
- The repository requires heavy ML libraries for full functionality (PyTorch, Transformers, TorchVision, etc.).
- If these are not installed, importing the package is still possible; runtime operations will raise clear ImportError messages indicating the missing dependency.
