from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(model_dir: Path) -> dict:
    files = []
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "path": str(p.relative_to(model_dir)),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish model artifact manifest")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", default=Path("artifacts/model_artifacts_manifest.json"), type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect(args.model_dir)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.out), "n_files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
