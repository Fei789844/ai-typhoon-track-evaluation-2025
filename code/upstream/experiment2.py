"""Paired cyclone bootstrap. Run with --project-root and --output-dir.
Dependencies: numpy pandas matplotlib. Experiment1.py must be alongside this file.
Simultaneous bands are approximate, centered, fixed-SE max-deviation bands.
All model pairs and all 20 leads form one family (60 coordinates).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment1 import load_data, summarize, plot_summary
PAIRS = [('graphcast','pangu'), ('graphcast','fourcastnet'), ('pangu','fourcastnet')]
NAMES = ['GraphCast - Pangu-Weather','GraphCast - FourCastNet','Pangu-Weather - FourCastNet']
LEADS = np.arange(6,121,6)
B, SEED = 100000, 20261003

def analyze(data, out):
    # Pair BEFORE cyclone averaging; every difference uses the identical case.
    wide = data.pivot(index=['TC_ID','case_id','case_type','lead_hours'],columns='model',values='track_error_km')
    assert wide.notna().all().all()
    diffs = pd.DataFrame({a+'-'+b:wide[a]-wide[b] for a,b in PAIRS}).reset_index()
    diffs.to_csv(out/'data/case_paired_differences.csv',index=False)
    rows=[]; summary={}; loco=[]
    for name in ['primary_full120','supplement_all_available']:
        d=diffs if name.startswith('supplement') else diffs[diffs.case_type=='FULL_120H']
        ids=sorted(d.TC_ID.unique()); n=len(ids)
        s=d.groupby(['TC_ID','lead_hours'])[[a+'-'+b for a,b in PAIRS]].mean()
        x=np.stack([s[p].unstack('lead_hours').reindex(index=ids,columns=LEADS).to_numpy() for p in s.columns],axis=2)
        assert x.shape==(n,20,3)
        rng=np.random.default_rng(SEED+(0 if name.startswith('primary') else 1))
        # Reuse each sampled cyclone vector across every lead and model pair.
        idx=rng.integers(n,size=(B,n))
        weights=np.zeros((B,n),dtype=np.int32)
        np.add.at(weights,(np.repeat(np.arange(B),n),idx.ravel()),1)
        flat=x.reshape(n,-1)
        valid=weights@np.isfinite(flat).astype(float)
        numer=weights@np.nan_to_num(flat)
        boot=np.divide(numer,valid,out=np.full_like(numer,np.nan),where=valid>0).reshape(B,20,3)
        mu=np.nanmean(x,axis=0); lo,hi=np.nanquantile(boot,[.025,.975],axis=0)
        bandlo=np.full_like(mu,np.nan); bandhi=bandlo.copy()
        if name=='primary_full120':
            assert np.isfinite(x).all()
            se=x.std(axis=0,ddof=1)/np.sqrt(n)
            assert (se>0).all()
            maxima=np.max(np.abs((boot-mu)/se),axis=(1,2))
            q=float(np.quantile(maxima,.95)); bandlo=mu-q*se; bandhi=mu+q*se
            summary['simultaneous_critical_value']=q
            for k,tc in enumerate(ids):
                v=np.delete(x,k,axis=0).mean(axis=0)
                for j,t in enumerate(LEADS):
                    loco.append({'omitted_cyclone':tc,'lead_hours':int(t),'GC_minus_Pangu':float(v[j,0])})
        for j,t in enumerate(LEADS):
            for k,(a,b) in enumerate(PAIRS):
                rows.append(dict(analysis_set=name,pair=a+'-'+b,lead_hours=int(t),n_storms=int(np.isfinite(x[:,j,k]).sum()),n_cases=int(d[d.lead_hours==t].case_id.nunique()),mean=mu[j,k],point_low=lo[j,k],point_high=hi[j,k],sim_low=bandlo[j,k],sim_high=bandhi[j,k]))
        summary[name]={}
        for k,pair in enumerate(NAMES):
            summary[name][pair]={'negative_point_CI_leads':LEADS[hi[:,k]<0].tolist(),'positive_point_CI_leads':LEADS[lo[:,k]>0].tolist(),'negative_sim_band_leads':LEADS[bandhi[:,k]<0].tolist(),'positive_sim_band_leads':LEADS[bandlo[:,k]>0].tolist()}
    result=pd.DataFrame(rows); result.to_csv(out/'data/paired_by_lead.csv',index=False)
    ld=pd.DataFrame(loco); ld.to_csv(out/'data/loco_GC_Pangu.csv',index=False)
    # No fitted change point: report first positive lead that remains positive.
    crossing=[]
    for tc,g in ld.groupby('omitted_cyclone'):
        v=g.sort_values('lead_hours').GC_minus_Pangu.to_numpy()
        cross=next((int(LEADS[j]) for j in range(1,20) if v[j-1]<0 and (v[j:]>0).all()),None)
        crossing.append({'omitted':tc,'first_sustained_positive_lead':cross})
    summary['loco_crossing']=crossing
    summary['loco120_range']=[float(ld[ld.lead_hours==120].GC_minus_Pangu.min()),float(ld[ld.lead_hours==120].GC_minus_Pangu.max())]
    plot(result,ld,out)
    return result,summary

def plot(r,ld,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for name,stem in [('primary_full120','Figure_2_paired'),('supplement_all_available','Figure_S2_paired')]:
        fig,axs=plt.subplots(3,1,figsize=(8.5,8.8),sharex=True)
        for k,((a,b),title,ax) in enumerate(zip(PAIRS,NAMES,axs)):
            g=r[(r.analysis_set==name)&(r.pair==a+'-'+b)]
            ax.axhline(0,color='black',lw=.9); ax.fill_between(g.lead_hours,g.point_low,g.point_high,color='#27699d',alpha=.2,label='Pointwise 95% CI')
            ax.plot(g.lead_hours,g['mean'],'o-',color='#27699d',ms=3,label='Paired mean')
            if name=='primary_full120':
                ax.plot(g.lead_hours,g.sim_low,'--',color='#9d453b',lw=1.2,label='Simultaneous 95% band (60 comparisons)')
                ax.plot(g.lead_hours,g.sim_high,'--',color='#9d453b',lw=1.2)
            ax.set_title(title,loc='left'); ax.set_ylabel('DPE difference (km)'); ax.grid(alpha=.2)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=1,fontsize=9,frameon=False)
        fig.subplots_adjust(left=.12,right=.98,bottom=.08,top=.865,hspace=.32)
        axs[-1].set_xlabel('Forecast lead time (h)'); axs[-1].set_xticks([6,24,48,72,96,120])
        for ext in ['png','pdf','svg']: fig.savefig(out/f'figures/{stem}.{ext}',dpi=250)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
    for tc,g in ld.groupby('omitted_cyclone'): ax.plot(g.lead_hours,g.GC_minus_Pangu,color='gray',alpha=.45,lw=1)
    g=r[(r.analysis_set=='primary_full120')&(r.pair=='graphcast-pangu')]
    ax.plot(g.lead_hours,g['mean'],color='#27699d',lw=2,label='All 14 cyclones');ax.axhline(0,color='black',lw=.8)
    ax.set(xlabel='Forecast lead time (h)',ylabel='GraphCast - Pangu DPE (km)',title='Leave-one-cyclone-out sensitivity of paired mean');ax.legend();ax.grid(alpha=.2)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'figures/Figure_S3_loco.{ext}',dpi=250)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    for f in ['data','figures']:(args.output_dir/f).mkdir(parents=True,exist_ok=True)
    data,audit=load_data(args.project_root);s,storms=summarize(data)
    s.to_csv(args.output_dir/'data/overall_by_lead.csv',index=False);storms.to_csv(args.output_dir/'data/storm_level.csv',index=False);plot_summary(s,args.output_dir/'figures')
    result,summary=analyze(data,args.output_dir)
    # Independent identity check: paired means = difference of overall means.
    for _,row in result.iterrows():
        a,b=row['pair'].split('-');g=s[(s.analysis_set==row.analysis_set)&(s.lead_hours==row.lead_hours)].set_index('model')
        assert abs(row['mean']-(g.loc[a,'mean_error_km_storm_weighted']-g.loc[b,'mean_error_km_storm_weighted']))<1e-9
    audit.update({'experiment2_repeats':B,'experiment2_seed':SEED,'simultaneous_family':'3 pairs x 20 leads, primary only','method':'centered pairs bootstrap max absolute deviation / original standard error','summary':summary})
    (args.output_dir/'data/audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    print(result[(result.analysis_set=='primary_full120')&(result.pair=='graphcast-pangu')].to_string(index=False))
if __name__=='__main__': main()
