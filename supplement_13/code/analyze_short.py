"""Evaluate the 13 truncated storms against the prespecified common-lead window.

Reads only archived input CSVs and the saved full-sample record table. No model
field or tracking rerun is performed. Output tables can be regenerated locally.
"""
from __future__ import annotations

import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent
INPUTS = MAIN / "inputs"
OUT = ROOT / "tables"
AUDIT = ROOT / "audit"
LEADS = list(range(6, 37, 6))
MODELS = ("graphcast", "pangu", "fourcastnet")
METRICS = ("DPE", "absATE", "absCTE")
SEED = 20261018
B = 20_000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(data: pd.DataFrame, name: str) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True, parents=True)
    data.to_csv(OUT / f"{name}.csv", index=False, float_format="%.12g")
    return data


def dpe(lat1, lon1, lat2, lon2):
    a, b, c, d = np.deg2rad([lat1, lon1, lat2, lon2])
    h = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    a, b = np.deg2rad([lat1, lat2])
    dl = np.deg2rad(lon2-lon1)
    return np.arctan2(np.sin(dl)*np.cos(b), np.cos(a)*np.sin(b)-np.sin(a)*np.cos(b)*np.cos(dl))


def load_short() -> pd.DataFrame:
    manifest = json.loads((MAIN / "audit/input_manifest.json").read_text())
    for item in manifest["files"]:
        assert digest(INPUTS / item["file"]) == item["sha256"], item["file"]
    raw = pd.read_csv(INPUTS / "track_points.csv", dtype={"TC_ID": str})
    cat = pd.read_csv(INPUTS / "case_catalog.csv", dtype={"TC_ID": str})
    raw = raw.merge(cat[["case_id", "case_type", "init_utc", "forecast_hours"]], on="case_id", validate="many_to_one")
    z = raw[(raw.case_type == "SHORT") & (raw.lead_hours > 0)].copy()
    assert len(z) == 456 and z.TC_ID.nunique() == 13 and z.case_id.nunique() == 13
    assert not z.duplicated(["model", "case_id", "lead_hours"]).any()
    assert set(z.model) == set(MODELS)
    paired = [set(map(tuple, x[["case_id", "lead_hours"]].to_numpy())) for _, x in z.groupby("model")]
    assert paired[0] == paired[1] == paired[2]
    z["valid_time"] = pd.to_datetime(z.valid_utc.str.replace(" UTC", ""), utc=True)
    z["init_time"] = pd.to_datetime(z.init_utc.str.replace(" UTC", ""), utc=True)
    assert ((z.valid_time-z.init_time).dt.total_seconds()/3600 == z.lead_hours).all()
    truth = pd.read_csv(INPUTS / "jma_track_points.csv", dtype={"TC_ID": str})
    truth["ts"] = pd.to_datetime(truth.time_utc.str.replace(" UTC", ""), utc=True)
    lookup = {(r.TC_ID, r.ts): (r.lat_deg_n, r.lon_deg_e) for r in truth.itertuples()}
    assert all(lookup[(r.TC_ID, r.valid_time)] == (r.jma_lat_deg_n, r.jma_lon_deg_e) for r in z.itertuples())
    calc = dpe(z.pred_lat_deg_n, z.pred_lon_deg_e, z.jma_lat_deg_n, z.jma_lon_deg_e)
    discrepancy = float(np.abs(calc-z.track_error_km).max())
    assert discrepancy < .00501
    z["DPE"] = z.track_error_km
    angles = []
    for r in z[["TC_ID", "valid_time", "jma_lat_deg_n", "jma_lon_deg_e"]].drop_duplicates().itertuples():
        previous = lookup.get((r.TC_ID, r.valid_time-pd.Timedelta(hours=12)))
        displacement = float(dpe(*previous, r.jma_lat_deg_n, r.jma_lon_deg_e)) if previous else np.nan
        alpha = float(bearing(r.jma_lat_deg_n, r.jma_lon_deg_e, *previous)+np.pi) if previous and displacement >= 10 else np.nan
        angles.append((r.TC_ID, r.valid_time, alpha, displacement))
    angles = pd.DataFrame(angles, columns=["TC_ID", "valid_time", "alpha", "back12_km"])
    z = z.merge(angles, on=["TC_ID", "valid_time"], validate="many_to_one")
    beta = bearing(z.jma_lat_deg_n, z.jma_lon_deg_e, z.pred_lat_deg_n, z.pred_lon_deg_e)
    z["ATE"] = z.DPE*np.cos(beta-z.alpha)
    z["CTE"] = z.DPE*np.sin(beta-z.alpha)
    z["absATE"] = z.ATE.abs()
    z["absCTE"] = z.CTE.abs()
    valid = z.ATE.notna()
    assert np.allclose(z.loc[valid, "ATE"]**2+z.loc[valid, "CTE"]**2, z.loc[valid, "DPE"]**2, rtol=1e-10, atol=2e-5)
    assert z[z.lead_hours.isin(LEADS)][list(METRICS)].notna().all().all()
    kinds = pd.read_csv(MAIN / "tables/turn_classification.csv", dtype={"TC_ID": str})
    z = z.merge(kinds[["TC_ID", "recurving", "turn_time"]], on="TC_ID", validate="many_to_one")
    z["turn_time"] = pd.to_datetime(z.turn_time, utc=True)
    z["r_hours"] = (z.valid_time-z.turn_time).dt.total_seconds()/3600
    save(z, "short_records")
    audit = {"status": "passed", "n_short_storms": 13, "n_short_cases": 13,
             "n_records": len(z), "n_turn_storms": int(z[z.recurving].TC_ID.nunique()),
             "common_leads": LEADS, "direction_valid_common_rows": int(z[z.lead_hours.isin(LEADS)].ATE.notna().sum()),
             "dpe_max_recalculation_discrepancy_km": discrepancy}
    return z, audit


def boot_matrix(matrix: pd.DataFrame, key: str, simultaneous: bool = False) -> pd.DataFrame:
    assert matrix.notna().all().all()
    x = matrix.to_numpy(dtype=float)
    n = len(x)
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{key}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, n, size=(B, n))
    means = x[sample].mean(axis=1)
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1)/np.sqrt(n)
    low, high = np.quantile(means, [.025, .975], axis=0)
    result = matrix.columns.to_frame(index=False)
    result["mean"] = mean
    result["low"] = low
    result["high"] = high
    result["SE"] = se
    result["n_storms"] = n
    if simultaneous:
        good = se > 1e-12
        q = float(np.quantile(np.abs((means[:, good]-mean[good])/se[good]).max(axis=1), .95))
        result["sim_low"] = mean-q*se
        result["sim_high"] = mean+q*se
        result["family_size"] = len(mean)
        result["q95"] = q
    return result


def storm_values(records: pd.DataFrame, leads=LEADS) -> pd.DataFrame:
    z = records[records.lead_hours.isin(leads)]
    return z.groupby(["TC_ID", "model", "lead_hours"], as_index=False)[list(METRICS)].mean()


def pair_matrix(storm: pd.DataFrame) -> pd.DataFrame:
    wide = storm.pivot(index=["TC_ID", "lead_hours"], columns="model", values="DPE")
    rows = []
    for a, b in combinations(MODELS, 2):
        t = (wide[a]-wide[b]).rename("delta").reset_index()
        t["pair"] = f"{a}-{b}"
        rows.append(t)
    all_pairs = pd.concat(rows, ignore_index=True)
    return all_pairs.pivot(index="TC_ID", columns=["pair", "lead_hours"], values="delta").sort_index(axis=1)


def main() -> None:
    z, audit = load_short()
    full = pd.read_csv(MAIN / "tables/analysis_records.csv", dtype={"TC_ID": str})
    first_init = full[["TC_ID", "case_id", "init_time"]].drop_duplicates().sort_values("init_time").drop_duplicates("TC_ID")
    first = full[full.case_id.isin(first_init.case_id)]
    cohorts = {"short13": z, "full14_all_cases": full, "full14_first_case": first}
    means = []
    mean_intervals = []
    pair_stats = []
    pair_growth = []
    growth = []
    for label, records in cohorts.items():
        storm = storm_values(records)
        storm["cohort"] = label
        means.append(storm)
        dpe_matrix = storm.pivot(index="TC_ID", columns=["model", "lead_hours"], values="DPE").sort_index(axis=1)
        one = boot_matrix(dpe_matrix, f"dpe_mean_{label}")
        one["cohort"] = label
        mean_intervals.append(one)
        pair = pair_matrix(storm)
        p = boot_matrix(pair, f"pair_{label}", simultaneous=True)
        p["cohort"] = label
        pair_stats.append(p)
        pair_changes = {}
        for pair_name in pair.columns.get_level_values("pair").unique():
            pair_changes[pair_name] = pair[(pair_name, 36)]-pair[(pair_name, 6)]
        pair_change_matrix = pd.DataFrame(pair_changes)
        pair_change_matrix.columns.name = "pair"
        pg = boot_matrix(pair_change_matrix, f"pair_growth_{label}", simultaneous=True)
        pg["cohort"] = label
        pair_growth.append(pg)
        wide = storm.pivot(index="TC_ID", columns=["model", "lead_hours"], values=list(METRICS))
        changes = {}
        for m in MODELS:
            for metric in METRICS:
                changes[(m, metric)] = wide[(metric, m, 36)]-wide[(metric, m, 6)]
        matrix = pd.DataFrame(changes).sort_index(axis=1)
        matrix.columns.names = ["model", "metric"]
        g = boot_matrix(matrix, f"growth_{label}", simultaneous=True)
        g["cohort"] = label
        growth.append(g)
    storm_all = save(pd.concat(means, ignore_index=True), "storm_means_common6_36")
    save(pd.concat(mean_intervals, ignore_index=True), "cohort_DPE_CI_common6_36")
    pair_all = save(pd.concat(pair_stats, ignore_index=True), "paired_DPE_common6_36")
    save(pd.concat(pair_growth, ignore_index=True), "pair_gap_change6_to36")
    save(pd.concat(growth, ignore_index=True), "growth6_to36")
    avg = storm_all.groupby(["cohort", "model", "lead_hours"], as_index=False)[list(METRICS)].mean()
    avg["n_storms"] = avg.cohort.map({"short13": 13, "full14_all_cases": 14, "full14_first_case": 14})
    save(avg, "cohort_means_common6_36")

    short_storm = storm_all[storm_all.cohort == "short13"]
    direction = short_storm.assign(delta=short_storm.absATE-short_storm.absCTE).pivot(
        index="TC_ID", columns=["model", "lead_hours"], values="delta").sort_index(axis=1)
    save(boot_matrix(direction, "direction_short13", simultaneous=True), "short_direction_contrast")
    pair_by_cohort = {k: pair_matrix(storm_all[storm_all.cohort == k]) for k in cohorts}
    pair_storm = pair_by_cohort["short13"].stack(["pair", "lead_hours"], future_stack=True).rename("delta").reset_index()
    save(pair_storm, "short_pair_by_storm")
    loco = []
    for omitted in pair_by_cohort["short13"].index:
        for (pair, lead), val in pair_by_cohort["short13"].drop(index=omitted).mean().items():
            loco.append({"omitted_TC_ID": omitted, "pair": pair, "lead_hours": lead, "mean": val})
    save(pd.DataFrame(loco), "short_pair_LOCO")
    save(short_storm.groupby(["model", "lead_hours"], as_index=False)[list(METRICS)].median(), "short_model_medians")
    cross = []
    short_x = pair_by_cohort["short13"].to_numpy()
    for comparison in ["full14_all_cases", "full14_first_case"]:
        full_x = pair_by_cohort[comparison].to_numpy()
        seed = int.from_bytes(hashlib.sha256(f"{SEED}|cross_{comparison}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        b1 = short_x[rng.integers(0, 13, size=(B, 13))].mean(axis=1)
        b0 = full_x[rng.integers(0, 14, size=(B, 14))].mean(axis=1)
        boot = b1-b0
        mu = short_x.mean(0)-full_x.mean(0)
        low, high = np.quantile(boot, [.025, .975], axis=0)
        se = np.sqrt(short_x.var(axis=0, ddof=1)/13 + full_x.var(axis=0, ddof=1)/14)
        q = float(np.quantile(np.abs((boot-mu)/se).max(axis=1), .95))
        t = pair_by_cohort["short13"].columns.to_frame(index=False)
        t["short_minus_full"] = mu
        t["low"] = low
        t["high"] = high
        t["sim_low"] = mu-q*se
        t["sim_high"] = mu+q*se
        t["family_size"] = len(mu)
        t["q95"] = q
        t["full_comparison"] = comparison
        cross.append(t)
    save(pd.concat(cross, ignore_index=True), "cross_cohort_pair_difference")

    counts = z.groupby("lead_hours").agg(n_storms=("TC_ID", "nunique"), n_cases=("case_id", "nunique")).reset_index()
    save(counts, "short_support_by_lead")
    second = storm_values(z, list(range(42, 79, 6)))
    second = second.groupby(["model", "lead_hours"], as_index=False).DPE.mean().merge(
        counts[["lead_hours", "n_storms"]], on="lead_hours", validate="many_to_one")
    save(second, "secondary_DPE_42_to78")

    # The turn stage check tests identifiability, never replaces same-lead pairing with raw time averages.
    turn_cases = z[z.recurving][["TC_ID", "name", "case_id", "init_time", "turn_time", "forecast_hours"]].drop_duplicates()
    turn_cases["forecast_init_before_turn"] = turn_cases.init_time < turn_cases.turn_time
    turn_cases["turn_within_window"] = turn_cases.turn_time.between(
        turn_cases.init_time, turn_cases.init_time+pd.to_timedelta(turn_cases.forecast_hours, unit="h"))
    # A case has one verification time for any given lead, so it cannot supply both stages at the same lead.
    turn_cases["within_storm_same_lead_pre_post_pairs"] = 0
    save(turn_cases, "short_turn_case_support")
    t = storm_all[(storm_all.cohort == "short13") & (storm_all.lead_hours.isin(LEADS))].copy()
    t["recurving"] = t.TC_ID.isin(turn_cases.TC_ID)
    save(t, "short_turn_individual_common6_36")
    save(t.groupby(["recurving", "model", "lead_hours"], as_index=False)[list(METRICS)].mean(), "short_turn_group_means_common6_36")

    audit.update({"seed": SEED, "B": B, "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(),
                  "protocol_sha256": digest(ROOT / "分析方案.md"), "code_sha256": digest(Path(__file__)),
                  "n_common_short_storms": short_storm.TC_ID.nunique(), "n_turn_storms": turn_cases.TC_ID.nunique(),
                  "n_turn_storms_init_before_turn": int(turn_cases.forecast_init_before_turn.sum()),
                  "n_same_lead_event_pairs": int(turn_cases.within_storm_same_lead_pre_post_pairs.sum())})
    AUDIT.mkdir(exist_ok=True, parents=True)
    (AUDIT / "analysis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
