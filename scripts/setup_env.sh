#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=".venv"
echo "Creating virtual environment at ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"

echo "Upgrading pip in the virtual environment"
"${VENV_PATH}/bin/python" -m pip install -U pip

echo "Installing package in editable mode and developer requirements"
"${VENV_PATH}/bin/python" -m pip install -e .
if [ -f dev-requirements.txt ]; then
  "${VENV_PATH}/bin/python" -m pip install -r dev-requirements.txt
fi

echo "Setup complete. To activate the venv run: source ${VENV_PATH}/bin/activate"
