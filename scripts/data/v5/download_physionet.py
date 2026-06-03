"""Download and process PhysioNet neurological datasets into the v5 schema.

Datasets (all open-access, no credentials required):
  - PADS: Parkinson's Disease Smartwatch Dataset
    physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
  - Non-EEG Physiological Signals of Neurological Status
    physionet.org/content/noneeg-neurological-status/1.0.0/
  - COVID-19 Patients with Neurological Comorbidities (includes MS)
    physionet.org/content/covid-neurological-comorbidities/1.0.0/

For Tier-2 MIMIC-IV Demo (requires free PhysioNet account):
    Place oasis-mimic_demo_processed.csv in data/raw/physionet/ and re-run.

Usage:
    python scripts/data/v5/download_physionet.py [--out-dir data/raw/physionet]
    # Requires: pip install wfdb requests
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scripts.data.v5.schema import (
    ALL_FEATURES,
    DISEASE_GENOMIC_PRIORS,
    META_COLS,
    POP_DEFAULTS,
)

_PHYSIONET_BASE = "https://physionet.org/files"

# PhysioNet dataset registry: name → (db_path, version)
_DATASETS = {
    "pads": ("parkinsons-disease-smartwatch", "1.0.0"),
    "noneeg": ("noneeg-neurological-status", "1.0.0"),
}

_RATE_LIMIT_DELAY = 1.5  # seconds between requests (PhysioNet rate-limits aggressively)


def _scaffold(n: int, disease_type: str, data_source: str) -> pd.DataFrame:
    df = pd.DataFrame({col: [POP_DEFAULTS.get(col, np.nan)] * n for col in ALL_FEATURES})
    genomic = DISEASE_GENOMIC_PRIORS.get(disease_type, DISEASE_GENOMIC_PRIORS["Parkinson's Disease"])
    for col, val in genomic.items():
        df[col] = val
    df["DiseaseType"] = disease_type
    df["data_source"] = data_source
    return df


def _fetch_file(url: str, timeout: int = 60) -> bytes:
    time.sleep(_RATE_LIMIT_DELAY)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _list_physionet_files(db: str, version: str) -> list[str]:
    """List files in a PhysioNet database using the REST API."""
    url = f"https://physionet.org/rest/files/{db}/{version}/"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Returns list of {name, url, type} objects
        return [f["name"] for f in data if f.get("type") == "file"]
    except Exception:
        return []


# ─── PADS: Parkinson's Disease Smartwatch Dataset ────────────────────────────

def download_pads(out_dir: Path) -> pd.DataFrame | None:
    """
    PADS contains per-subject time-series from smartwatch sensors.
    We aggregate to per-subject rows extracting wearable feature summaries.
    """
    db, ver = _DATASETS["pads"]
    base_url = f"{_PHYSIONET_BASE}/{db}/{ver}"
    pads_dir = out_dir / "pads"
    pads_dir.mkdir(parents=True, exist_ok=True)

    # Try to get the subject metadata first
    meta_candidates = [
        f"{base_url}/participants.tsv",
        f"{base_url}/RECORDS",
        f"{base_url}/demographics.csv",
        f"{base_url}/subjects.csv",
    ]

    meta_df = None
    for url in meta_candidates:
        try:
            content = _fetch_file(url)
            fname = url.split("/")[-1]
            (pads_dir / fname).write_bytes(content)
            if fname.endswith(".tsv"):
                meta_df = pd.read_csv(io.BytesIO(content), sep="\t")
            elif fname.endswith(".csv"):
                meta_df = pd.read_csv(io.BytesIO(content))
            if meta_df is not None:
                print(f"[pads] metadata: {fname} — {len(meta_df)} subjects")
                break
        except Exception:
            continue

    # Try to fetch summary CSV files (some PhysioNet datasets include pre-computed features)
    summary_candidates = [
        f"{base_url}/features.csv",
        f"{base_url}/summary.csv",
        f"{base_url}/clinical_data.csv",
    ]
    summary_df = None
    for url in summary_candidates:
        try:
            content = _fetch_file(url)
            summary_df = pd.read_csv(io.BytesIO(content))
            (pads_dir / url.split("/")[-1]).write_bytes(content)
            print(f"[pads] feature summary: {len(summary_df)} rows")
            break
        except Exception:
            continue

    if summary_df is None and meta_df is None:
        print("[pads] could not download PADS automatically — PhysioNet may require account.")
        print("       Manual: download from https://physionet.org/content/parkinsons-disease-smartwatch/")
        print("       Place CSV files in data/raw/physionet/pads/ and re-run.")
        return _load_local_pads(pads_dir)

    # Process whichever we got
    raw = summary_df if summary_df is not None else meta_df
    return _process_pads(raw)


def _load_local_pads(pads_dir: Path) -> pd.DataFrame | None:
    """Process any CSV/TSV already placed in data/raw/physionet/pads/."""
    csvs = list(pads_dir.glob("*.csv")) + list(pads_dir.glob("*.tsv"))
    if not csvs:
        return None
    raw = pd.concat(
        [pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ",") for f in csvs],
        ignore_index=True
    )
    print(f"[pads] loaded {len(raw)} rows from {len(csvs)} local files")
    return _process_pads(raw)


def _process_pads(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    if n == 0:
        return None

    col = {c.strip().lower().replace(" ", "_").replace("-", "_"): c for c in raw.columns}

    # Determine PD vs control label
    label_col = None
    for candidate in ("group", "diagnosis", "label", "condition", "pd", "status", "subject_group"):
        if candidate in col:
            label_col = col[candidate]
            break

    df = _scaffold(n, "Parkinson's Disease", "physionet_pads")

    if label_col:
        labels = raw[label_col].astype(str).str.lower()
        is_pd = labels.isin(["pd", "parkinson", "parkinson's", "1", "yes", "patient"])
        df["risk_label"] = is_pd.astype(int)
        df.loc[~is_pd, "DiseaseType"] = "Healthy"
        df.loc[~is_pd, "risk_label"] = 0
    else:
        df["risk_label"] = 1  # all subjects in PADS are PD patients

    # Age / sex
    for src, dst in [("age", "Age"), ("sex", "Gender"), ("gender", "Gender")]:
        if src in col:
            df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")

    # Wearable features (map from common PADS column names)
    wearable_map = {
        "tremor": "tremor_amplitude", "tremor_amplitude": "tremor_amplitude",
        "tremor_rms": "tremor_amplitude", "acceleration_rms": "tremor_amplitude",
        "gait_speed": "gait_velocity", "gait_velocity": "gait_velocity",
        "step_velocity": "gait_velocity", "walking_speed": "gait_velocity",
        "step_asymmetry": "step_asymmetry", "gait_asymmetry": "step_asymmetry",
        "activity_index": "actigraphy_activity_index",
        "activity": "actigraphy_activity_index",
        "hr": "HR_variability", "heart_rate": "HR_variability",
        "hr_variability": "HR_variability", "hrv": "HR_variability",
        "spo2": "SpO2_mean", "spo2_mean": "SpO2_mean",
    }
    for src_key, dst in wearable_map.items():
        if src_key in col:
            df[dst] = pd.to_numeric(raw[col[src_key]], errors="coerce")

    # UPDRS if present
    for src_key, dst in [("updrs_motor", "UPDRS_motor"), ("motor_updrs", "UPDRS_motor"),
                          ("updrs_total", "UPDRS_total"), ("total_updrs", "UPDRS_total"),
                          ("updrs", "UPDRS_total")]:
        if src_key in col:
            df[dst] = pd.to_numeric(raw[col[src_key]], errors="coerce")

    print(f"[pads] processed {n} rows, PD: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── Non-EEG Neurological Status ─────────────────────────────────────────────

def download_noneeg(out_dir: Path) -> pd.DataFrame | None:
    db, ver = _DATASETS["noneeg"]
    base_url = f"{_PHYSIONET_BASE}/{db}/{ver}"
    noneeg_dir = out_dir / "noneeg"
    noneeg_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        f"{base_url}/clinical_data.csv",
        f"{base_url}/demographics.csv",
        f"{base_url}/participants.tsv",
        f"{base_url}/summary.csv",
        f"{base_url}/data.csv",
    ]
    raw_df = None
    for url in candidates:
        try:
            content = _fetch_file(url)
            fname = url.split("/")[-1]
            (noneeg_dir / fname).write_bytes(content)
            raw_df = pd.read_csv(
                io.BytesIO(content),
                sep="\t" if fname.endswith(".tsv") else ","
            )
            print(f"[noneeg] found {fname}: {len(raw_df)} rows")
            break
        except Exception:
            continue

    if raw_df is None:
        print("[noneeg] could not download — check manually at:")
        print("         https://physionet.org/content/noneeg-neurological-status/")
        print("         Place CSV files in data/raw/physionet/noneeg/ and re-run.")
        return _load_local_noneeg(noneeg_dir)

    return _process_noneeg(raw_df)


def _load_local_noneeg(noneeg_dir: Path) -> pd.DataFrame | None:
    csvs = list(noneeg_dir.glob("*.csv")) + list(noneeg_dir.glob("*.tsv"))
    if not csvs:
        return None
    raw = pd.concat(
        [pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ",") for f in csvs],
        ignore_index=True
    )
    return _process_noneeg(raw)


def _process_noneeg(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    if n == 0:
        return None

    col = {c.strip().lower().replace(" ", "_").replace("-", "_"): c for c in raw.columns}
    df = _scaffold(n, "Healthy", "physionet_noneeg")

    # Label — neurological status
    label_col = None
    for candidate in ("neurological_status", "status", "group", "diagnosis", "label", "condition"):
        if candidate in col:
            label_col = col[candidate]
            break

    if label_col:
        labels = raw[label_col].astype(str).str.lower()
        is_neuro = ~labels.isin(["healthy", "normal", "control", "0", "no"])
        df["risk_label"] = is_neuro.astype(int)
        # Assign disease type from label if informative
        for idx, label_val in enumerate(labels):
            for key, disease in {
                "parkinson": "Parkinson's Disease", "ms": "Multiple Sclerosis",
                "sclerosis": "Multiple Sclerosis", "epilep": "Epilepsy",
                "alzheimer": "Alzheimer's Disease",
            }.items():
                if key in label_val:
                    df.loc[idx, "DiseaseType"] = disease
                    break
        df.loc[~is_neuro, "DiseaseType"] = "Healthy"
    else:
        df["risk_label"] = 0

    # Demographics
    for src, dst in [("age", "Age"), ("sex", "Gender"), ("gender", "Gender")]:
        if src in col:
            df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")

    # Wearable physiological signals
    wearable_map = {
        "hr": "HR_variability", "heart_rate": "HR_variability", "hrv": "HR_variability",
        "spo2": "SpO2_mean", "oxygen_saturation": "SpO2_mean",
        "temperature": "actigraphy_activity_index",  # proxy; no direct mapping
        "acceleration": "tremor_amplitude", "acc_rms": "tremor_amplitude",
        "eda": "actigraphy_activity_index",
        "gait": "gait_velocity",
    }
    for src_key, dst in wearable_map.items():
        if src_key in col:
            df[dst] = pd.to_numeric(raw[col[src_key]], errors="coerce")

    print(f"[noneeg] processed {n} rows, neuro fraction: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── Local fallback: any CSVs manually placed in physionet/ root ──────────────

def _scan_local_csvs(out_dir: Path) -> list[pd.DataFrame]:
    """Process any CSVs manually downloaded and placed in out_dir."""
    frames = []
    for csv_path in out_dir.glob("*.csv"):
        try:
            raw = pd.read_csv(csv_path)
            print(f"[local] {csv_path.name}: {len(raw)} rows, cols: {list(raw.columns[:8])}")
            frames.append((csv_path.stem, raw))
        except Exception as exc:
            print(f"[local] failed to read {csv_path.name}: {exc}")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Download PhysioNet neurological datasets")
    ap.add_argument("--out-dir", default="data/raw/physionet")
    ap.add_argument("--skip-download", action="store_true",
                    help="Only process already-downloaded files in out-dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    # PADS
    print("\n=== PADS: Parkinson's Disease Smartwatch ===")
    try:
        df_pads = download_pads(out_dir) if not args.skip_download else _load_local_pads(out_dir / "pads")
        if df_pads is not None and len(df_pads) > 0:
            p = out_dir / "pads_v5.parquet"
            df_pads.to_parquet(p, index=False)
            print(f"Saved {len(df_pads)} rows → {p}")
            frames.append(df_pads)
    except Exception as exc:
        print(f"PADS failed: {exc}")

    # Non-EEG
    print("\n=== Non-EEG Neurological Status ===")
    try:
        df_noneeg = download_noneeg(out_dir) if not args.skip_download else _load_local_noneeg(out_dir / "noneeg")
        if df_noneeg is not None and len(df_noneeg) > 0:
            p = out_dir / "noneeg_v5.parquet"
            df_noneeg.to_parquet(p, index=False)
            print(f"Saved {len(df_noneeg)} rows → {p}")
            frames.append(df_noneeg)
    except Exception as exc:
        print(f"Non-EEG failed: {exc}")

    # Any other CSVs manually placed
    print("\n=== Scanning for manually downloaded files ===")
    for stem, raw in _scan_local_csvs(out_dir):
        # Minimal processing: just check for known disease/risk columns
        col = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}
        n = len(raw)
        disease = "Parkinson's Disease"
        df = _scaffold(n, disease, f"physionet_{stem}")
        for src, dst in [("age", "Age"), ("sex", "Gender"), ("mmse", "MMSE")]:
            if src in col:
                df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")
        df["risk_label"] = 0
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        p = out_dir / "physionet_combined_v5.parquet"
        combined.to_parquet(p, index=False)
        dist = combined["DiseaseType"].value_counts().to_dict()
        print(f"\nPhysioNet combined: {len(combined)} rows → {p}")
        print(f"Disease distribution: {dist}")
    else:
        print("\nNo PhysioNet data processed.")
        print("\nManual download instructions:")
        print("  PADS: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/")
        print("  Non-EEG: https://physionet.org/content/noneeg-neurological-status/1.0.0/")
        print("  MS/COVID: https://physionet.org/content/covid-neurological-comorbidities/")
        print("  Place downloaded CSVs in data/raw/physionet/ and re-run with --skip-download")


if __name__ == "__main__":
    main()
