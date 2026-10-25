"""Experiment 1: validate DPE, storm-weighted means, bootstrap and plots.

Run: python experiment1.py --project-root /path/to/project --output-dir /path/to/output
Dependencies: numpy, pandas, matplotlib. Does not rerun weather models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ['fourcastnet', 'pangu', 'graphcast']
LABELS = ['FourCastNet', 'Pangu-Weather', 'GraphCast']
COLORS = ['#0072B2', '#D55E00', '#009E73']
SETS = ['primary_full120', 'supplement_all_available']


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; angles are supplied in degrees."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad((np.asarray(lon2) - np.asarray(lon1) + 180) % 360 - 180)
    a = np.sin((p2 - p1) / 2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_data(root):
    root=Path(root).resolve()
    paths = sorted((root / 'results/tracks/by_job').glob('*/track_points.csv'))
    cases_path = root / 'aliyun_2025_reproduction/config/case_catalog.csv'
    cases = pd.read_csv(cases_path, dtype={'TC_ID': str})
    verified=Path(__file__).resolve().parents[1]/'data/verified_track_points.csv'
    if not verified.exists():raise FileNotFoundError('Run retrack_verified.py first: '+str(verified))
    track=pd.read_csv(verified,dtype={'TC_ID':str})
    assert len(paths) == 71 and len(cases) == 87
    assert cases.case_id.is_unique and set(track.model) == set(MODELS)
    assert set(track.case_id) == set(cases.case_id)
    assert not track.duplicated(['case_id', 'lead_hours', 'model']).any()
    assert np.isfinite(track.track_error_km).all()
    assert (track.track_error_km >= 0).all()
    cols = ['case_id', 'case_type', 'forecast_hours', 'init_utc']
    data = track.merge(cases[cols], on='case_id', validate='many_to_one')
    for case_id, g in data.groupby('case_id'):
        hours = int(g.forecast_hours.iloc[0])
        for model in MODELS:
            assert set(g.loc[g.model == model, 'lead_hours']) == set(range(0, hours+1, 6))
        assert g.TC_ID.nunique() == 1
        assert g.TC_ID.iloc[0] == cases.set_index('case_id').loc[case_id, 'TC_ID']
    for _, g in data.groupby(['case_id', 'lead_hours']):
        assert g.valid_utc.nunique() == 1
        assert g.jma_lat_deg_n.nunique() == g.jma_lon_deg_e.nunique() == 1
        valid = pd.to_datetime(g.valid_utc.iloc[0].replace(' UTC', ''), utc=True)
        init = pd.to_datetime(g.init_utc.iloc[0].replace(' UTC', ''), utc=True)
        assert valid == init + pd.Timedelta(hours=int(g.lead_hours.iloc[0]))
    distance = haversine(data.pred_lat_deg_n, data.pred_lon_deg_e,
                         data.jma_lat_deg_n, data.jma_lon_deg_e)
    delta = np.abs(distance - data.track_error_km)
    assert np.isfinite(distance).all() and delta.max() <= 0.02
    # Preserve the production CSV's stored precision after checking DPE.
    data = data.loc[data.lead_hours > 0].copy()
    full = data.loc[data.case_type == 'FULL_120H']
    assert full.case_id.nunique() == 74 and full.TC_ID.nunique() == 14
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [cases_path, *paths,verified]}
    return data, {'max_dpe_recalculation_difference_km': float(delta.max()),
                  'forecast_rows': len(data), 'status_counts': data.status.value_counts().to_dict(),
                  'input_sha256': manifest}


def bootstrap(values, key, repeats=10000, seed=20250831):
    """Pointwise percentile interval; each value is one cyclone mean."""
    raw = '|'.join(map(str, (seed, 'overall', *key))).encode()
    derived = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little')
    rng = np.random.default_rng(derived)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975])


def summarize(data):
    rows, storms = [], []
    for name in SETS:
        d = data if name == SETS[1] else data.loc[data.case_type == 'FULL_120H']
        s = d.groupby(['TC_ID', 'lead_hours', 'model']).track_error_km.mean().reset_index()
        s['analysis_set'] = name
        storms.append(s)
        for (lead, model), g in s.groupby(['lead_hours', 'model']):
            v = g.sort_values('TC_ID').track_error_km.to_numpy()
            low, high = bootstrap(v, (name, lead, model))
            n = d.loc[(d.lead_hours == lead) & (d.model == model)].case_id.nunique()
            rows.append([name, lead, model, len(v), n, v.mean(), low, high])
    cols = ['analysis_set', 'lead_hours', 'model', 'n_storms', 'n_cases',
            'mean_error_km_storm_weighted', 'bootstrap_ci95_low_km', 'bootstrap_ci95_high_km']
    return pd.DataFrame(rows, columns=cols), pd.concat(storms)


def plot_summary(summary, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    for name in SETS:
        d = summary.loc[summary.analysis_set == name]
        fig, (ax, an) = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True,
            gridspec_kw={'height_ratios': [4, 1.1], 'hspace': .12})
        for m, label, color in zip(MODELS, LABELS, COLORS):
            g = d.loc[d.model == m].sort_values('lead_hours')
            x = g.lead_hours.to_numpy()
            ax.plot(x, g.mean_error_km_storm_weighted, color=color, label=label,
                    marker='o', markersize=3, linewidth=2)
            ax.fill_between(x, g.bootstrap_ci95_low_km, g.bootstrap_ci95_high_km,
                            color=color, alpha=.13)
        ax.set(ylabel='Mean DPE (km)', ylim=(0, 950))
        ax.legend(loc='upper left', frameon=False)
        ax.grid(alpha=.2)
        n = d.loc[d.model == MODELS[0]].sort_values('lead_hours')
        an.plot(n.lead_hours, n.n_cases, 'o-', color='#333333', markersize=3, label='Cases')
        an.plot(n.lead_hours, n.n_storms, 's--', color='#888888', markersize=3, label='Cyclones')
        an.set(ylabel='Sample size', xlabel='Forecast lead time (h)', ylim=(0, 100),
               xticks=[6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
        an.legend(loc='center right', ncol=2, frameon=False, fontsize=9)
        an.grid(alpha=.2)
        fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.98)
        stem = 'Figure_1_primary' if name == SETS[0] else 'Figure_S1_all_available'
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{stem}.{extension}', dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    for folder in ['data', 'figures']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    data, audit = load_data(args.project_root)
    summary, storms = summarize(data)
    summary.to_csv(out / 'data/overall_by_lead.csv', index=False, encoding='utf-8-sig')
    storms.to_csv(out / 'data/storm_level.csv', index=False, encoding='utf-8-sig')
    key = summary.loc[summary.lead_hours.isin([24, 48, 72, 96, 120])]
    key.to_csv(out / 'data/key_leads.csv', index=False, encoding='utf-8-sig')
    plot_summary(summary, out / 'figures')
    audit.update({'bootstrap_replicates': 10000, 'seed': 20250831,
                  'ci_type': 'pointwise percentile; cyclone resampling',
                  'numpy': np.__version__, 'pandas': pd.__version__})
    (out / 'data/run_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
