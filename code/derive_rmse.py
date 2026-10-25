"""由归档案例DPE计算台风等权距离RMSE；不运行天气模型或追踪器。

运行：python code/derive_rmse.py
依赖：numpy, pandas；输入在本目录上一级 data/。
算法：先求台风内DPE平方均值，再对台风等权平均，最后开平方。
所有模型和时效共用同一台风bootstrap索引；区间仍为逐坐标百分位区间。
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SOURCE = DATA / 'error_components_cases.csv'
BASELINE = DATA / 'overall_by_lead.csv'
SEED = 20261018
REPEATS = 10000

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    all_records = pd.read_csv(SOURCE)
    df = all_records.loc[all_records.case_type.eq('FULL_120H')].copy()
    assert len(df) == 4440
    assert df.TC_ID.nunique() == 14 and df.case_id.nunique() == 74
    assert not df.duplicated(['model', 'case_id', 'lead_hours']).any()
    assert set(df.lead_hours) == set(range(6, 121, 6))
    assert df.groupby(['model', 'lead_hours']).size().eq(74).all()
    assert np.isfinite(df.track_error_km).all() and df.track_error_km.ge(0).all()
    keysets = [set(zip(x.case_id, x.lead_hours)) for _, x in df.groupby('model')]
    assert all(s == keysets[0] for s in keysets)
    models = ['graphcast', 'pangu', 'fourcastnet']
    storms = sorted(df.TC_ID.unique())
    n = len(storms)
    indices = np.random.default_rng(SEED).integers(0, n, (REPEATS, n))
    baseline = pd.read_csv(BASELINE).query("analysis_set == 'primary_full120'")
    rows = []
    max_mean_diff = 0.0
    for lead in range(6, 121, 6):
        for model in models:
            x = df.loc[df.lead_hours.eq(lead) & df.model.eq(model)]
            means = x.groupby('TC_ID').track_error_km.mean().reindex(storms)
            mse = x.assign(squared=x.track_error_km**2).groupby('TC_ID').squared.mean().reindex(storms).to_numpy()
            original = baseline.loc[baseline.lead_hours.eq(lead) & baseline.model.eq(model), 'mean_error_km_storm_weighted'].item()
            max_mean_diff = max(max_mean_diff, abs(means.mean() - original))
            assert np.isclose(means.mean(), original, atol=1e-9, rtol=0)
            estimate = float(np.sqrt(mse.mean()))
            assert estimate + 1e-9 >= means.mean()
            boot = np.sqrt(mse[indices].mean(axis=1))
            low, high = np.quantile(boot, [0.025, 0.975], method='linear')
            rows.append(dict(model=model, lead_hours=lead, n_storms=n,
                             n_cases=len(x), rmse_km=estimate,
                             ci95_low_km=float(low), ci95_high_km=float(high)))
    result = pd.DataFrame(rows)
    result.to_csv(DATA / 'rmse_by_lead.csv', index=False)
    audit = dict(seed=SEED, repeats=REPEATS, numpy=np.__version__,
                 pandas=pd.__version__, python=platform.python_version(),
                 n_records=len(df), n_storms=n, n_cases=df.case_id.nunique(),
                 maximum_mean_DPE_difference_from_archive_km=max_mean_diff,
                 input_sha256={p.name: sha256(p) for p in [SOURCE, BASELINE]},
                 script_sha256=sha256(Path(__file__)),
                 output_sha256=sha256(DATA / 'rmse_by_lead.csv'),
                 method='sqrt(mean_across_storms(mean_within_storm(DPE**2)))',
                 interval='pointwise percentile cluster bootstrap; 14 storm draws; fixed shared indices')
    (DATA / 'rmse_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result.loc[result.lead_hours.isin([24,48,72,96,120])].to_string(index=False))

if __name__ == '__main__':
    main()
