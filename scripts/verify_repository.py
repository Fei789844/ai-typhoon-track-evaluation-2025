"""Check repository structure, local Markdown links and core sample counts."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def check_markdown_links() -> list[str]:
    missing: list[str] = []
    pattern = re.compile(r"\]\(([^)]+)\)")
    for document in ROOT.rglob("*.md"):
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            path = (document.parent / target).resolve()
            if not path.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    return missing


def main() -> None:
    required = [
        ROOT / "data" / "overall_by_lead.csv",
        ROOT / "data" / "paired_by_lead.csv",
        ROOT / "data" / "IMD_correlations.csv",
        ROOT / "inputs" / "track_points.csv",
        ROOT / "tables" / "analysis_records.csv",
        ROOT / "manuscript" / "AI台风路径论文.md",
    ]
    absent = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not absent, f"Missing required files: {absent}"

    records = pd.read_csv(ROOT / "tables" / "analysis_records.csv")
    assert len(records) == 4440
    assert records["TC_ID"].nunique() == 14
    assert records["case_id"].nunique() == 74
    assert set(records["lead_hours"]) == set(range(6, 121, 6))
    assert set(records["model"]) == {"graphcast", "pangu", "fourcastnet"}

    missing = check_markdown_links()
    assert not missing, "Broken local Markdown links:\n" + "\n".join(missing)

    oversized = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size > 50 * 1024 * 1024]
    assert not oversized, f"Files larger than 50 MiB: {oversized}"
    print("PASS: required files, 4,440-record main sample, Markdown links and file sizes")


if __name__ == "__main__":
    main()
