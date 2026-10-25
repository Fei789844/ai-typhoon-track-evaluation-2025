"""All-lead extension; existing matching and complete-case group definitions retained.
Figures: matched-cell heterogeneity; group differences; deletion sensitivity.
No weather models rerun. No new significance tests on descriptive time bands.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tables';F=ROOT/'figures';A=ROOT/'audit'
M=['graphcast','pangu','fourcastnet'];names=['GraphCast','Pangu-Weather','FourCastNet'];K=['DPE','absATE','absCTE'];labels=['DPE','Absolute ATE','Absolute CTE'];leads=list(range(6,121,6));storms=[2509,2519,2522,2523,2526]
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=A/(name+'.alignment.json'),strict=True)
 fig.savefig(F/(name+'.png'),dpi=600)
 fig.savefig(F/(name+'.pdf'))
 fig.savefig(F/(name+'.svg'))
 plt.close(fig)
def read(n):return pd.read_csv(T/(n+'.csv'))
def write(d,n):d.to_csv(T/(n+'.csv'),index=False,float_format='%.12g')
pairs=read('event_matched_lead_differences');pairs=pairs[pairs.metric.isin(K)]
assert len(pairs)==216 and not pairs.duplicated(['TC_ID','model','metric','lead_hours']).any()
summary=pairs.groupby(['model','metric']).agg(positive_cells=('delta',lambda x:(x>0).sum()),negative_cells=('delta',lambda x:(x<0).sum()),n_cells=('delta','size')).reset_index();write(summary,'event_cell_sign_counts')
fig,axs=plt.subplots(3,3,figsize=(10,6.7));fig.subplots_adjust(left=.12,right=.97,top=.91,bottom=.18,hspace=.43,wspace=.19)
lim=np.ceil(pairs.delta.abs().max()/100)*100
for i,m in enumerate(M):
 for j,k in enumerate(K):
  ax=axs[i,j];z=pairs[(pairs.model==m)&(pairs.metric==k)].pivot(index='TC_ID',columns='lead_hours',values='delta').reindex(index=storms,columns=leads)
  cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
  im=ax.imshow(z,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
  ax.set_yticks(range(5),[str(x) for x in storms] if j==0 else [])
  ax.set_xticks([0,3,7,11,15,19],[6,24,48,72,96,120]);ax.set_title(f'{chr(97+i*3+j)}  {labels[j]}')
  if j==0:ax.set_ylabel(names[i],labelpad=7)
  if i==2:ax.set_xlabel('Forecast lead time (h)')
cax=fig.add_axes([.33,.065,.40,.018]);fig.colorbar(im,cax=cax,orientation='horizontal',label='Post-turn minus pre-turn error (km)')
# Colorbar is not a comparable data panel.
cax.set_label('<colorbar>')
save(fig,'Figure_4_3b_event_leads')
raw=read('analysis_records')
d=raw.dropna(subset=K)  # Same joint-valid records as the archived group analysis.
record_counts={'before':len(raw),'after':len(d),'excluded':len(raw)-len(d),'rule':'DPE, absATE and absCTE all finite; matches original group_analysis'}
long=d.melt(id_vars=['TC_ID','model','lead_hours','recurving'],value_vars=K,var_name='metric',value_name='value')
s=long.groupby(['TC_ID','recurving','model','lead_hours','metric'],as_index=False).value.mean();write(s,'group_storm_lead_means')
bas=s.groupby(['recurving','model','lead_hours','metric']).value.mean().unstack('recurving');base=(bas[True]-bas[False]).rename('mean').reset_index()
old=read('group_difference');chk=base.merge(old,on=['model','metric','lead_hours'],suffixes=('_new','_old'));assert np.allclose(chk.mean_new,chk.mean_old)
loco=[]
for tc in sorted(s.TC_ID.unique()):
 z=s[s.TC_ID!=tc].groupby(['recurving','model','lead_hours','metric']).value.mean().unstack('recurving');z=(z[True]-z[False]).rename('mean').reset_index();z['omitted']=tc;loco.append(z)
loco=pd.concat(loco,ignore_index=True);assert len(loco)==2520;write(loco,'group_loco_all_leads')
check=loco[loco.lead_hours==120].merge(read('group_loco120'),on=['omitted','model','metric'],suffixes=('_new','_old'));assert len(check)==126 and np.allclose(check.mean_new,check.mean_old)
env=loco.groupby(['model','metric','lead_hours']).agg(minimum=('mean','min'),maximum=('mean','max'),positive_deletions=('mean',lambda x:(x>0).sum())).reset_index();write(env,'group_loco_envelope')
bands=[]
for name,lo,hi in [('6–48 h',6,48),('54–96 h',54,96),('102–120 h',102,120)]:
 z=s[s.lead_hours.between(lo,hi)].groupby(['TC_ID','recurving','model','metric'],as_index=False).value.mean();a=z.groupby(['recurving','model','metric']).value.mean().unstack('recurving');a=a.rename(columns={True:'turn',False:'other'});a['difference']=a.turn-a.other;a=a.reset_index();a['band']=name
 zz=z[z.TC_ID!=2522].groupby(['recurving','model','metric']).value.mean().unstack('recurving');a=a.merge((zz[True]-zz[False]).rename('without_HALONG').reset_index(),on=['model','metric']);bands.append(a)
write(pd.concat(bands),'group_time_band_summary')
for sensitivity in [False,True]:
 fig,axs=plt.subplots(3,3,figsize=(9,7.2),sharex=True);fig.subplots_adjust(left=.10,right=.97,bottom=.09,top=.90,hspace=.43,wspace=.32)
 for i,m in enumerate(M):
  for j,k in enumerate(K):
   ax=axs[i,j];g=old[(old.model==m)&(old.metric==k)].sort_values('lead_hours');x=g.lead_hours.to_numpy()
   ax.axhline(0,color='#444444',lw=.7)
   if sensitivity:
    z=env[(env.model==m)&(env.metric==k)].sort_values('lead_hours');ax.fill_between(x,z.minimum,z.maximum,color='#dce3e8',label='Range of 14 deletions')
    h=loco[(loco.model==m)&(loco.metric==k)&(loco.omitted==2522)].sort_values('lead_hours');ax.plot(x,h['mean'],color='#d07830',ls='--',lw=1.3,label='Without HALONG')
   else:
    ax.fill_between(x,g.low,g.high,color='#dce9f0',label='Pointwise 95% CI');ax.plot(x,g.sim_low,color='#d07830',ls='--',lw=1,label='Simultaneous 95% band');ax.plot(x,g.sim_high,color='#d07830',ls='--',lw=1)
   ax.plot(x,g['mean'],color='#12688c',lw=1.5,label='All storms')
   ax.set_title(f'{chr(97+i*3+j)}  {names[i]} | {labels[j]}',loc='left',fontsize=8)
   if j==0:ax.set_ylabel('Turning − non-turning (km)')
   if i==2:ax.set_xlabel('Forecast lead time (h)')
   ax.set_xticks([6,24,48,72,96,120]);ax.set_xlim(6,120)
 handles,ll=axs[0,0].get_legend_handles_labels();fig.legend(handles,ll,loc='upper center',ncol=3,frameon=False,fontsize=8)
 save(fig,'Figure_4_4c_group_loco' if sensitivity else 'Figure_4_4b_group_differences')
audit={'inputs':{n:hashlib.sha256((T/(n+'.csv')).read_bytes()).hexdigest() for n in ['analysis_records','event_matched_lead_differences','group_difference']},'matched_cells_per_model_metric':24,'loco_rows':len(loco),'group_baseline_reproduced':True,'loco120_reproduced':True,'band_summary':'storm first; descriptive; no significance tests','figure_contract':'Matched heatmaps show within-storm signs; group curves show all-lead contrasts and uncertainty; deletion envelope shows sensitivity, not a confidence interval.'}
audit['record_counts']=record_counts
audit['layout']='Thesis screen-readable panels: 254 mm heatmap; 228.6 mm group grids. Not a journal submission layout.'
(A/'all_lead_extension_20261024.json').write_text(json.dumps(audit,indent=2))
print(summary.to_string(index=False));print(pd.concat(bands).query("metric=='DPE'").to_string(index=False))
for m in M:
 for k in K:
  g=old[(old.model==m)&(old.metric==k)];h=loco[(loco.model==m)&(loco.metric==k)&(loco.omitted==2522)];z=env[(env.model==m)&(env.metric==k)]
  print(m,k,'positive',g[g['mean']>0].lead_hours.tolist(),'noHALpositive',h[h['mean']>0].lead_hours.tolist(),'all_deletions_positive',z[z.minimum>0].lead_hours.tolist(),'sim_exclude0',g[(g.sim_low>0)|(g.sim_high<0)].lead_hours.tolist())
