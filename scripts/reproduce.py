"""Recompute the repository's archived statistics and figures.

This entry point starts from archived forecast-center coordinates. It does not
download ERA5, run the three weather models, or reread regional GRIB files.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(relative_path: str) -> None:
    command = [sys.executable, str(ROOT / relative_path)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    for folder in (ROOT / "figures", ROOT / "tables", ROOT / "audit",
                   ROOT / "supplement_13" / "figures",
                   ROOT / "supplement_13" / "tables",
                   ROOT / "supplement_13" / "audit"):
        folder.mkdir(parents=True, exist_ok=True)

    stages = [
        "code/derive_rmse.py",
        "code/check_jma_dateline.py",
        "code/analyze_chapter4.py",
        "code/verify_chapter4.py",
        "code/plot_chapter4.py",
        "code/extend_lead_analysis.py",
        "supplement_13/code/analyze_short.py",
        "supplement_13/code/verify_short.py",
        "supplement_13/code/plot_short.py",
        "code/plot_chapter5_focused.py",
    ]
    for stage in stages:
        run(stage)
    print("PASS: all archived-coordinate analyses and figures were reproduced")


if __name__ == "__main__":
    main()
