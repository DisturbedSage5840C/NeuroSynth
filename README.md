# NeuroSynth

NeuroSynth is a lightweight neurological risk prediction app trained on the OASIS longitudinal dataset.

It predicts `Demented` vs `Nondemented` using these features:

- `Age`
- `EDUC`
- `SES`
- `MMSE`
- `CDR`
- `eTIV`
- `nWBV`
- `ASF`

## Files Added For The Minimal Workflow

- `explore.py` inspects the dataset quickly
- `train.py` trains a `RandomForestClassifier`
- `main.py` serves predictions with FastAPI
- `index.html` provides a simple local frontend
- `app.py` provides a Gradio app for Hugging Face Spaces
- `requirements.txt` provides minimal dependencies

All other existing repository files remain untouched.

## Dataset

Place `oasis_longitudinal.csv` in the repository root.

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Explore data:

```bash
python explore.py
```

3. Train model:

```bash
python train.py
```

This creates:

- `model.pkl`
- `label_encoder.pkl`

4. Start API:

```bash
uvicorn main:app --reload
```

5. Test in browser:

- API docs: `http://localhost:8000/docs`
- Optional local UI: open `index.html` in a browser

## Deploy Free On Hugging Face Spaces

1. Create a new Space (Gradio).
2. Upload:

- `app.py`
- `model.pkl`
- `label_encoder.pkl`
- `requirements.txt`

3. Hugging Face will auto-build and host your app.
