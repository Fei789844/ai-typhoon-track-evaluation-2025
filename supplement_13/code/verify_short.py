"""Independent record-level checks for the 13-storm supplementary analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
INPUTS = ROOT.parent / "inputs"


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv", dtype={"TC_ID": str})


def same(a, b, name, tolerance=1e-7):
    assert np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tolerance, rtol=0), name


def main():
    raw = pd.read_csv(INPUTS / "track_points.csv", dtype={"TC_ID": str})
    case = pd.read_csv(INPUTS / "case_catalog.csv", dtype={"TC_ID": str})
    short_ids = set(case.loc[case.case_type == "SHORT", "case_id"])
    original = raw[raw.case_id.isin(short_ids) & raw.lead_hours.gt(0)].copy()
    saved = read("short_records")
    assert len(original) == len(saved) == 456
    assert set(original.case_id) == set(saved.case_id) and saved.TC_ID.nunique() == 13
    assert not saved.duplicated(["model", "case_id", "lead_hours"]).any()
    q = saved.merge(original, on=["case_id", "model", "lead_hours"], suffixes=("", "_source"), validate="one_to_one")
    same(q.DPE, q.track_error_km_source, "archived DPE")
    common = saved[saved.lead_hours.between(6, 36)]
    assert len(common) == 13*3*6 and common[["DPE", "ATE", "CTE"]].notna().all().all()
    same(common.ATE**2+common.CTE**2, common.DPE**2, "directional decomposition", tolerance=2e-5)
    mean = read("cohort_means_common6_36")
    short_mean = mean[mean.cohort == "short13"]
    direct = common.groupby(["model", "lead_hours"], as_index=False)[["DPE", "absATE", "absCTE"]].mean()
    q = short_mean.merge(direct, on=["model", "lead_hours"], suffixes=("", "_check"), validate="one_to_one")
    for metric in ("DPE", "absATE", "absCTE"):
        same(q[metric], q[f"{metric}_check"], f"short mean {metric}")
    pairs = read("short_pair_by_storm")
    direct_pair = common.pivot(index=["TC_ID", "lead_hours"], columns="model", values="DPE")
    for pair, group in pairs.groupby("pair"):
        a,b = pair.split("-")
        d = (direct_pair[a]-direct_pair[b]).rename("check").reset_index()
        q = group.merge(d, on=["TC_ID", "lead_hours"], validate="one_to_one")
        same(q.delta, q.check, f"model pair {pair}")
    turn = read("short_turn_case_support")
    assert len(turn) == 2 and turn.forecast_init_before_turn.sum() == 1
    assert turn.within_storm_same_lead_pre_post_pairs.sum() == 0
    snap = saved[(saved.recurving) & (saved.model == "graphcast") & (saved.lead_hours == 36)]
    assert sorted(snap.r_hours.tolist()) == [0, 36]
    report = {"status": "passed", "n_short_records": 456, "n_common_records": len(common),
              "n_turn_storms": 2, "same_lead_pre_post_pairs": 0,
              "checks": ["source DPE", "direction identity", "cohort means", "paired contrasts", "turn support"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "audit/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
