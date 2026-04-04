from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def run_cmd(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSynth training orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Print planned training steps without running heavy jobs")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    steps = [
        ["python", "-m", "neurosynth.connectome.train", "--help"],
        ["python", "-m", "neurosynth.genomic.train", "--help"],
        ["python", "-m", "neurosynth.temporal_tft.train", "--help"],
    ]

    summary = {"dry_run": args.dry_run, "steps": []}
    for step in steps:
        if args.dry_run:
            summary["steps"].append({"cmd": step, "status": "planned"})
            continue
        rc = run_cmd(step)
        summary["steps"].append({"cmd": step, "status": "ok" if rc == 0 else "failed", "rc": rc})
        if rc != 0:
            break

    (ARTIFACTS / "train_orchestrator_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(s["status"] in {"planned", "ok"} for s in summary["steps"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
