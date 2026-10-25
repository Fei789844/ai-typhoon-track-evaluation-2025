"""Independent checks of source integrity, pairing, stage contrasts and outputs.

This program reads the archived CSVs directly; it does not import the analysis
program or its helper functions. Run after analyze_chapter4.py.
"""
from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from build_2025_jma_case_catalog import parse_jma_zip


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"
METRICS = ("DPE", "absATE", "absCTE")


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv", dtype={"TC_ID": str})


def close(left, right, label: str, tol: float = 1e-7) -> None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    assert a.shape == b.shape and np.allclose(a, b, rtol=0, atol=tol), label


def haversine(lat1, lon1, lat2, lon2):
    a, b, c, d = np.deg2rad([lat1, lon1, lat2, lon2])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def main() -> None:
    checks = []
    manifest = json.loads((AUDIT / "input_manifest.json").read_text())
    for item in manifest["files"]:
        actual = hashlib.sha256((INPUTS / item["file"]).read_bytes()).hexdigest()
        assert actual == item["sha256"], item["file"]
    checks.append(f"SHA-256: {len(manifest['files'])} input files match")

    source_metadata = json.loads((INPUTS / "jma_source_metadata.json").read_text())
    assert source_metadata["source_sha256"] == hashlib.sha256(
        (INPUTS / "jma_original_bst_all_20260723.zip").read_bytes()).hexdigest()
    tracks, _ = parse_jma_zip(INPUTS / "jma_original_bst_all_20260723.zip")
    archived_jma = pd.read_csv(INPUTS / "jma_track_points.csv", dtype={"TC_ID": str})
    lookup = {(tc, point.time.strftime("%Y-%m-%d %H:%M UTC")): point
              for tc, points in tracks.items() for point in points}
    for row in archived_jma.itertuples():
        source = lookup[(row.TC_ID, row.time_utc)]
        assert (source.lat, source.lon, source.grade, source.source_line) == (
            row.lat_deg_n, row.lon_deg_e, row.jma_grade_code, row.source_line)
    checks.append(f"JMA source ZIP: SHA-256 and {len(archived_jma):,} parsed positions match")

    d = table("analysis_records")
    d["init_time"] = pd.to_datetime(d["init_time"], utc=True)
    d["valid_time"] = pd.to_datetime(d["valid_time"], utc=True)
    assert len(d) == 4440 and d.TC_ID.nunique() == 14 and d.case_id.nunique() == 74
    assert not d.duplicated(["case_id", "model", "lead_hours"]).any()
    keys = [set(map(tuple, x[["case_id", "lead_hours"]].to_numpy())) for _, x in d.groupby("model")]
    assert keys[0] == keys[1] == keys[2] and len(keys[0]) == 1480
    assert ((d.valid_time - d.init_time).dt.total_seconds() / 3600 == d.lead_hours).all()
    checks.append("Paired data: 14 storms, 74 cases, 20 leads, 4,440 forecast records")

    recomputed = haversine(d.jma_lat_deg_n, d.jma_lon_deg_e, d.pred_lat_deg_n, d.pred_lon_deg_e)
    discrepancy = float(np.max(np.abs(recomputed - d.DPE)))
    assert discrepancy < 0.00501
    valid = d.ATE.notna() & d.CTE.notna()
    close(d.loc[valid, "ATE"] ** 2 + d.loc[valid, "CTE"] ** 2,
          d.loc[valid, "DPE"] ** 2, "ATE²+CTE²=DPE²", tol=2e-5)
    assert int((~valid).sum()) == 24
    checks.append(f"Geometry: DPE discrepancy <= {discrepancy:.6f} km; 24 direction records excluded")

    classes = table("turn_classification")
    previous = pd.read_csv(INPUTS / "previous_classes.csv", dtype={"TC_ID": str})
    merged = classes.merge(previous, on="TC_ID", suffixes=("", "_old"), validate="one_to_one")
    assert (merged.recurving == merged.recurving_old).all()
    main_storms = set(d.TC_ID)
    assert len(classes) == 27 and classes[classes.TC_ID.isin(main_storms)].recurving.sum() == 5
    checks.append("Turn detection: exact archived class agreement across 27 storms; 5/14 in main analysis")

    stage = d[d.recurving & (d.init_relative_h < 0) & d.turn_relative_h.between(-24, 24)].copy()
    stage["phase"] = np.select([stage.turn_relative_h <= -12, stage.turn_relative_h >= 12],
                                ["pre", "post"], default="near")
    direct_cells = stage.groupby(["TC_ID", "model", "lead_hours", "phase"], as_index=False)[list(METRICS)].mean()
    saved_cells = table("event_stage_cells")
    check = saved_cells.merge(direct_cells, on=["TC_ID", "model", "lead_hours", "phase"], suffixes=("", "_direct"), validate="one_to_one")
    assert len(check) == len(saved_cells)
    for metric in METRICS:
        close(check[metric], check[f"{metric}_direct"], f"stage mean {metric}")
    checks.append("Stage cells recomputed independently from record-level times and errors")

    before = direct_cells[direct_cells.phase == "pre"]
    after = direct_cells[direct_cells.phase == "post"]
    pairs = before.merge(after, on=["TC_ID", "model", "lead_hours"], suffixes=("_pre", "_post"), validate="one_to_one")
    assert pairs.TC_ID.nunique() == 5 and len(pairs) == 72  # 24 storm×lead cells, times 3 models
    saved_effects = table("event_storm_effects")
    for metric in METRICS:
        direct = pairs.assign(delta=pairs[f"{metric}_post"] - pairs[f"{metric}_pre"])
        direct = direct.groupby(["TC_ID", "model"], as_index=False).delta.mean()
        test = saved_effects[saved_effects.metric == metric].merge(direct, on=["TC_ID", "model"], suffixes=("", "_direct"), validate="one_to_one")
        close(test.delta, test.delta_direct, f"per-storm {metric} effect")
    checks.append("Matched post-minus-pre: 24 storm×lead cells, equal-lead pairing and storm effects agree")

    inference = table("event_primary_inference")
    for row in inference.itertuples():
        values = saved_effects[(saved_effects.model == row.model) & (saved_effects.metric == row.metric)].sort_values("TC_ID").delta.to_numpy()
        close([values.mean()], [row.mean], f"mean {row.model}/{row.metric}")
        close([values.std(ddof=1) / math.sqrt(5)], [row.SE], f"SE {row.model}/{row.metric}")
        # Enumerating all 5^5 resamples removes Monte Carlo variation.
        boot = np.array([values[list(idx)].mean() for idx in product(range(5), repeat=5)])
        close(np.quantile(boot, [.025, .975]), [row.low, row.high], f"exact bootstrap {row.model}/{row.metric}")
    checks.append("All 9 stage means, SEs and exact 3,125-resample percentile intervals agree")

    group = table("group_means")
    subgroup = d[valid].groupby(["TC_ID", "model", "lead_hours", "recurving"], as_index=False)[list(METRICS)].mean()
    by_group = subgroup.groupby(["model", "lead_hours", "recurving"], as_index=False)[list(METRICS)].mean()
    long = by_group.melt(id_vars=["model", "lead_hours", "recurving"], var_name="metric", value_name="direct")
    test = group[group.metric.isin(METRICS)].merge(long, on=["model", "lead_hours", "recurving", "metric"], validate="one_to_one")
    assert len(test) == 360
    close(test["mean"], test.direct, "group means")
    checks.append("All 360 lead×model×group×metric means agree with record-level recalculation")

    output = {"status": "passed", "checks": checks, "maximum_distance_discrepancy_km": discrepancy,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (AUDIT / "verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print("\n".join(checks))


if __name__ == "__main__":
    main()
