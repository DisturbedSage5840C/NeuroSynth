#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

START_SERVER=1
if [[ "${1:-}" == "--no-serve" ]]; then
  START_SERVER=0
fi

if [[ -x "$ROOT_DIR/.venv312/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv312/bin/python"
  PIP="$ROOT_DIR/.venv312/bin/pip"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
  PIP="$ROOT_DIR/.venv/bin/pip"
else
  PYTHON="python3"
  PIP="pip3"
fi

if [[ ! -f "$ROOT_DIR/oasis_longitudinal.csv" ]]; then
  echo "Missing oasis_longitudinal.csv in project root."
  exit 1
fi

echo "[1/4] Installing dependencies"
"$PIP" install -r requirements.txt

echo "[2/4] Exploring dataset"
"$PYTHON" explore.py

echo "[3/4] Training model"
"$PYTHON" train.py

echo "[4/4] Running smoke test against FastAPI app"
"$PYTHON" - <<'PY'
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
payload = {
    "age": 75,
    "educ": 12,
    "ses": 2,
    "mmse": 24,
    "cdr": 0.5,
    "etiv": 1450,
    "nwbv": 0.72,
    "asf": 1.2,
}
r = client.post("/predict", json=payload)
print("Status:", r.status_code)
print("Response:", r.json())
if r.status_code != 200:
    raise SystemExit(1)
PY

if [[ "$START_SERVER" -eq 1 ]]; then
  echo "Smoke test passed. Starting API server..."
  exec "$PYTHON" -m uvicorn main:app --reload
else
  echo "Done. Start the API with: $PYTHON -m uvicorn main:app --reload"
fi
