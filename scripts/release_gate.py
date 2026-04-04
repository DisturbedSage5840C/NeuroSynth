from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERNS = [
    r"123456789012",
    r"REPLACE_IN_SECRET_MANAGER",
    r"example\.org",
    r"\bstub\b",
    r"\bplaceholder\b",
]

SCAN_GLOBS = ["src/**/*.py", "terraform/**/*.tf", "helm/**/*.yaml", "helm/**/*.yml", ".github/workflows/*.yml"]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def scan_placeholders() -> list[str]:
    issues: list[str] = []
    regexes = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]
    for glob in SCAN_GLOBS:
        for path in ROOT.glob(glob):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for rx in regexes:
                if rx.search(text):
                    issues.append(f"{path.relative_to(ROOT)} matched {rx.pattern}")
    return sorted(set(issues))


def main() -> int:
    report: dict[str, object] = {"checks": {}}

    code, out = run([sys.executable, "-m", "compileall", "src"])
    report["checks"]["compile_src"] = {"ok": code == 0, "output": out[-2000:]}

    placeholders = scan_placeholders()
    report["checks"]["placeholder_scan"] = {"ok": len(placeholders) == 0, "issues": placeholders}

    required_env = [
        "NEURO_JWKS_URL",
        "NEURO_ALLOWED_ORIGINS",
        "NEURO_REDIS_URL",
    ]
    missing = [v for v in required_env if not os.getenv(v)]
    report["checks"]["required_env"] = {"ok": len(missing) == 0, "missing": missing}

    ok = all(v.get("ok", False) for v in report["checks"].values())
    report["release_ready"] = ok

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
