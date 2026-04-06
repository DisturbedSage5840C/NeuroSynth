# NeuroSynth Testing Guide (20 Cases)

## Quick Start

1. Start backend:

```bash
source .venv312/bin/activate
set -a
source .env
set +a
python -m backend.api
```

2. Start frontend:

```bash
npm --prefix frontend run dev -- --host 0.0.0.0 --port 5173
```

3. Open UI:

- http://localhost:5173

## What To Validate

1. Prediction label changes logically with MMSE/CDR shifts.
2. Probability and risk level move in same direction.
3. Trajectory is stable and monotonic for severe profiles.
4. Top risk factors include clinically plausible drivers.
5. Report endpoint returns structured section output.

## API Batch Test (All 20 Cases)

```bash
python - <<'PY'
import json
import requests

with open('test_payloads_20.json', 'r') as f:
    cases = json.load(f)

ok = 0
for case in cases:
    r = requests.post('http://127.0.0.1:8000/predict', json=case['payload'], timeout=30)
    if r.ok:
        out = r.json()
        print(f"{case['id']}: {out.get('prediction')} | p={out.get('probability')} | risk={out.get('risk_level')} | expected={case['expected_pattern']}")
        ok += 1
    else:
        print(f"{case['id']}: ERROR {r.status_code} {r.text[:120]}")

print(f"\nPassed requests: {ok}/{len(cases)}")
PY
```

## Can You Test Other Data?

Yes. You can test any additional payload as long as fields match:

- age
- educ
- ses
- mmse
- cdr
- etiv
- nwbv
- asf

Use either the UI sliders or POST to:

- `http://127.0.0.1:8000/predict`
- `http://127.0.0.1:8000/report`
- `http://127.0.0.1:8000/simulate`

## Interpretation Notes

1. Use outputs comparatively, not as absolute medical truth.
2. Larger CDR and lower MMSE should usually increase risk.
3. The causal map is directional model evidence, not definitive causality.
4. LLM report text is explanatory support; numeric prediction is the core signal.
5. Clinical use requires external validation and professional oversight.
