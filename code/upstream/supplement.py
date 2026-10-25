"""Additional conditional contrasts and sensitivity; deterministic derived outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments3to6 import ci,wcorr,base_weights,wquantile,MODELS,LABELS,COLORS,KEY,rng_for,score_metrics,geometry
from experiment1 import load_data
W=Path(__file__).resolve().parents[1]

def main():
 d=pd.read_csv(W/'data/error_components_cases.csv',dtype={'TC_ID':str});d=d[d.case_type=='FULL_120H'];cl=pd.read_csv(W/'data/cyclone_path_types.csv',dtype={'TC_ID':str});z=pd.read_csv(W/'data/disagreement_cases.csv',dtype={'TC_ID':str})
 x=np.array([1,2,2,4.]);y=np.array([4,2,3,1.]);assert np.allclose(wcorr(x,y,np.ones((1,4)))[0],np.corrcoef(pd.Series(x).rank(),pd.Series(y).rank())[0,1])
 d=d.merge(cl[['TC_ID','recurving','turn_time']],on='TC_ID',validate='many_to_one')
 d['turn_time']=pd.to_datetime(d.turn_time,utc=True);d['valid_time']=pd.to_datetime(d.valid_utc.str.replace(' UTC',''),utc=True);d['init_time']=pd.to_datetime(d.init_utc.str.replace(' UTC',''),utc=True)
 d['contains_turn']=(d.turn_time>=d.init_time)&(d.turn_time<=d.init_time+pd.Timedelta(hours=120))
 d['near_turn']=abs((d.valid_time-d.turn_time).dt.total_seconds()/3600)<=12
 cols=['case_id','TC_ID','recurving','turn_time','contains_turn']
 d[cols].drop_duplicates().to_csv(W/'data/case_path_classes.csv',index=False)
 # Conditional component summaries requested in the research outline.
 conditional=[]
 for (kind,m,lead),v in d[d.ATE.notna()].groupby(['recurving','model','lead_hours']):
  s=v.groupby('TC_ID')[['ATE','CTE','absATE','absCTE']].mean()
  for metric in s.columns:
   lo,hi=ci(s[metric],f'group_component{kind}{m}{lead}{metric}')
   conditional.append(dict(recurving=kind,model=m,lead_hours=lead,metric=metric,mean=s[metric].mean(),low=lo,high=hi,n_storms=len(s),n_cases=len(v)))
 conditional=pd.DataFrame(conditional);conditional.to_csv(W/'data/path_type_components.csv',index=False)
 fig,axs=plt.subplots(2,2,figsize=(9,7),sharex=True,sharey='row')
 for row,metric in enumerate(['absATE','absCTE']):
  for col,kind in enumerate([False,True]):
   ax=axs[row,col]
   for m,l,c in zip(MODELS,LABELS,COLORS):
    v=conditional[(conditional.recurving==kind)&(conditional.model==m)&(conditional.metric==metric)]
    ax.plot(v.lead_hours,v['mean'],color=c,label=l);ax.fill_between(v.lead_hours,v.low,v.high,color=c,alpha=.1)
   ax.set_title(('Recurving' if kind else 'No recurvature detected')+' / '+metric);ax.grid(alpha=.2)
   if row==1:ax.set_xlabel('Lead time (h)')
   if col==0:ax.set_ylabel('km')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(W/'figures/Figure_S7_group_components.png',dpi=220);plt.close(fig)
 rows=[];direction=[];group=[];near=[]
 raw,_=load_data(W.parents[1]);truth=pd.read_csv(W.parents[1]/'outputs/2025_jma_cases_v2/data/jma_2025_track_points.csv',dtype={'TC_ID':str})
 low,_=geometry(raw,truth,min_distance=5);low=low[low.case_type=='FULL_120H']
 for omit in sorted(d.TC_ID.unique()):
  g=d[(d.TC_ID!=omit)&(d.lead_hours==120)]
  for m,v in g.groupby('model'):
   s=v.groupby('TC_ID')[['ATE2','CTE2']].mean();rows.append(dict(omitted=omit,metric='along_squared_fraction',model=m,lead_hours=120,value=s.ATE2.mean()/(s.ATE2.mean()+s.CTE2.mean())))
   s=v.groupby(['recurving','TC_ID']).track_error_km.mean();rows.append(dict(omitted=omit,metric='recurve_minus_other',model=m,lead_hours=120,value=s.loc[True].mean()-s.loc[False].mean()))
  for lead,v in z[z.TC_ID!=omit].groupby('lead_hours'):
   rows.append(dict(omitted=omit,metric='IMD_rho',model='consensus',lead_hours=lead,value=wcorr(v.IMD.to_numpy(),v.consensus_error.to_numpy(),base_weights(v)[None,:])[0]))
 pd.DataFrame(rows).to_csv(W/'data/LOCO_secondary.csv',index=False)
 for (m,lead),v in d.groupby(['model','lead_hours']):
  matched=v[v.ATE_back12.notna()&v.ATE_center12.notna()]
  for kind in ['back12','center12']:
   s=matched.groupby('TC_ID')[[f'ATE_{kind}',f'CTE_{kind}']].agg(lambda x:(x*x).mean());a,c=s.mean()
   direction.append(dict(model=m,lead_hours=lead,kind=kind,matched_cases=len(matched),n_storms=len(s),along_fraction=a/(a+c)))
  for threshold in [5,10,20]:
   good=low[(low.model==m)&(low.lead_hours==lead)&(low.back12_distance>=threshold)];s=good.groupby('TC_ID')[['ATE2','CTE2']].mean();a,c=s.mean();direction.append(dict(model=m,lead_hours=lead,kind=f'displacement_ge{threshold}',matched_cases=len(good),n_storms=len(s),along_fraction=a/(a+c)))
  s=v.groupby(['recurving','TC_ID']).track_error_km.mean();aa=s.loc[True].to_numpy();bb=s.loc[False].to_numpy();rng=rng_for(f'group_contrast{m}{lead}');boot=aa[rng.integers(len(aa),size=(10000,len(aa)))].mean(1)-bb[rng.integers(len(bb),size=(10000,len(bb)))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);group.append(dict(model=m,lead_hours=lead,mean=aa.mean()-bb.mean(),low=lo,high=hi,n_recurve=len(aa),n_other=len(bb)))
  threshold=wquantile(v.track_error_km,base_weights(v),.75)
  for kind in [False,True]:
   vv=v[v.near_turn==kind];s=vv.groupby('TC_ID').track_error_km.mean();evt=vv.assign(event=vv.track_error_km>threshold).groupby('TC_ID').event.mean()
   near.append(dict(model=m,lead_hours=lead,near_turn=kind,n_storms=len(s),n_cases=len(vv),mean=s.mean(),large_error_fraction=evt.mean(),large_error_threshold=threshold))
 pd.DataFrame(direction).to_csv(W/'data/direction_matched.csv',index=False);pd.DataFrame(group).to_csv(W/'data/group_contrasts.csv',index=False);pd.DataFrame(near).to_csv(W/'data/near_turn_risk.csv',index=False)
 # Group interaction: paired model difference in recurving minus non-recurving storms.
 interactions=[]
 ss=d.groupby(['TC_ID','recurving','lead_hours','model']).track_error_km.mean().unstack('model')
 for lead in KEY:
  v=ss.xs(lead,level='lead_hours')
  for a,b in [('graphcast','pangu'),('graphcast','fourcastnet'),('pangu','fourcastnet')]:
   diff=v[a]-v[b];aa=diff.xs(True,level='recurving').to_numpy();bb=diff.xs(False,level='recurving').to_numpy();rng=rng_for(f'interaction{lead}{a}{b}');boot=aa[rng.integers(len(aa),size=(10000,len(aa)))].mean(1)-bb[rng.integers(len(bb),size=(10000,len(bb)))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);interactions.append(dict(lead_hours=lead,pair=a+'-'+b,mean=aa.mean()-bb.mean(),low=lo,high=hi))
 pd.DataFrame(interactions).to_csv(W/'data/group_interactions.csv',index=False)
 # Case-balance-preserving trend uses identical cyclone/case sets at both endpoints.
 fraction=[]
 for (lead,m),v in d.groupby(['lead_hours','model']):
  s=v.groupby('TC_ID')[['ATE2','CTE2']].mean();rng=rng_for(f'fraction{lead}{m}');a=s.ATE2.to_numpy();c=s.CTE2.to_numpy();idx=rng.integers(len(s),size=(10000,len(s)));b=a[idx].mean(1)/(a[idx].mean(1)+c[idx].mean(1));lo,hi=np.quantile(b,[.025,.975]);fraction.append(dict(lead_hours=lead,model=m,along_fraction=a.mean()/(a.mean()+c.mean()),low=lo,high=hi))
 pd.DataFrame(fraction).to_csv(W/'data/component_fraction_ci.csv',index=False)
 # Verify Monte Carlo stability of the original 60-coordinate band.
 pair_order=[('graphcast','pangu'),('graphcast','fourcastnet'),('pangu','fourcastnet')]
 s=d.groupby(['TC_ID','lead_hours','model']).track_error_km.mean().unstack('model')
 xx=np.stack([(s[a]-s[b]).unstack('lead_hours').to_numpy() for a,b in pair_order],axis=2);mu=xx.mean(0);se=xx.std(0,ddof=1)/np.sqrt(len(xx));seedcheck=[]
 for seed in [20261003,20261004,20261005]:
  rng=np.random.default_rng(seed);idx=rng.integers(len(xx),size=(100000,len(xx)));ww=np.zeros((100000,len(xx)),dtype=np.int32);np.add.at(ww,(np.repeat(np.arange(100000),len(xx)),idx.ravel()),1)
  boot=(ww@xx.reshape(len(xx),-1)/len(xx)).reshape(100000,20,3);q=float(np.quantile(np.max(abs((boot-mu)/se),axis=(1,2)),.95))
  seedcheck.append(dict(seed=seed,repeats=100000,q=q,GP120_mean=mu[-1,0],GP120_low=mu[-1,0]-q*se[-1,0],GP120_high=mu[-1,0]+q*se[-1,0]))
 pd.DataFrame(seedcheck).to_csv(W/'data/MC_seed_sensitivity.csv',index=False)
 if not (W/'data/retracking_audit.json').exists():
  print('Statistical supplements complete; awaiting complete GRIB audit');return
 # Retracking: matched original case definitions and same equal-cyclone weights.
 rt=pd.read_csv(W/'data/retracking_points.csv',dtype={'TC_ID':str});caseids=set(d.case_id);rt=rt[rt.case_id.isin(caseids)]
 assert rt.status.eq('TRACKED').all()
 s=rt.groupby(['variant','model','lead_hours','TC_ID']).DPE.mean().reset_index();s.to_csv(W/'data/retracking_storm_means.csv',index=False)
 summary=s.groupby(['variant','model','lead_hours']).DPE.agg(['mean','count']).reset_index();summary.to_csv(W/'data/retracking_summary.csv',index=False)
 pivot=s.pivot(index=['variant','lead_hours','TC_ID'],columns='model',values='DPE');pairs=[]
 for (variant,lead),v in pivot.groupby(['variant','lead_hours']):
  for a,b in [('graphcast','pangu'),('graphcast','fourcastnet'),('pangu','fourcastnet')]:
   diff=v[a]-v[b];lo,hi=ci(diff,f'retrack{variant}{lead}{a}{b}');pairs.append(dict(variant=variant,lead_hours=lead,pair=a+'-'+b,mean=diff.mean(),low=lo,high=hi))
 pd.DataFrame(pairs).to_csv(W/'data/retracking_pairs.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(10,3.5),sharey=True)
 for ax,m,l in zip(axs,MODELS,LABELS):
  for variant,g in summary[summary.model==m].groupby('variant'):ax.plot(g.lead_hours,g['mean'],label=variant,lw=2 if variant=='baseline' else 1)
  ax.set(title=l,xlabel='Lead time (h)');ax.grid(alpha=.2)
 axs[0].set_ylabel('Equal-cyclone DPE (km)');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=6,fontsize=8);fig.tight_layout(rect=(0,0,1,.86));fig.savefig(W/'figures/Figure_7_retracking.png',dpi=220);plt.close(fig)
 # High-error case map: diagnostic only, no physical identification claim.
 worst=d[d.lead_hours==120].sort_values('track_error_km',ascending=False).drop_duplicates('case_id').head(4)
 fig,axs=plt.subplots(2,2,figsize=(9,7))
 for ax,r in zip(axs.flat,worst.itertuples()):
  v=d[d.case_id==r.case_id];truth=v[v.model=='graphcast'].sort_values('lead_hours');ax.plot(truth.jma_lon_deg_e,truth.jma_lat_deg_n,'k.-',label='JMA')
  for m,l,c in zip(MODELS,LABELS,COLORS):
   u=v[v.model==m].sort_values('lead_hours');ax.plot(u.pred_lon_deg_e,u.pred_lat_deg_n,'.-',color=c,label=l,ms=3)
  ax.set(title=r.case_id, xlabel='Longitude (E)',ylabel='Latitude (N)');ax.grid(alpha=.2)
 fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper center',ncol=4);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(W/'figures/Figure_S5_high_error.png',dpi=200);plt.close(fig)
 d[d.track_error_km>1000][['case_id','TC_ID','model','lead_hours','track_error_km']].to_csv(W/'data/high_error_records.csv',index=False)
 print('Supplement complete',flush=True)
if __name__=='__main__':main()
