"""MSLP maps for reproducible inspection of high-influence HALONG cases."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tracker_source as tr
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1]
def main():
 d=pd.read_csv(W/'data/error_components_cases.csv',dtype={'TC_ID':str})
 cases=d[(d.TC_ID=='2522')&(d.model=='pangu')&(d.lead_hours==120)].sort_values('track_error_km',ascending=False).head(2)
 fig,axs=plt.subplots(2,3,figsize=(12,7.5));sources=[]
 for row,case in enumerate(cases.itertuples()):
  for col,(m,label) in enumerate(zip(['graphcast','pangu','fourcastnet'],['GraphCast','Pangu-Weather','FourCastNet'])):
   r=d[(d.case_id==case.case_id)&(d.model==m)&(d.lead_hours==120)].iloc[0]
   file=ROOT/'forecasts'/m/'by_init'/r.forecast_job_id/(('fcn' if m=='fourcastnet' else m)+'-120.grib');f=tr.read_grib_fields(file);ax=axs[row,col]
   contour=ax.contour(f.lon,f.lat,f.msl_hpa,levels=np.arange(940,1041,4),colors='#8d9aa5',linewidths=.5);ax.clabel(contour,fontsize=6,inline=True)
   ax.scatter(r.jma_lon_deg_e,r.jma_lat_deg_n,c='black',s=45,marker='*',label='JMA');ax.scatter(r.pred_lon_deg_e,r.pred_lat_deg_n,c='#D55E00',s=30,marker='o',label='Tracked center')
   ax.plot([r.jma_lon_deg_e,r.pred_lon_deg_e],[r.jma_lat_deg_n,r.pred_lat_deg_n],'--',c='#D55E00',lw=.8)
   minlon=min(r.jma_lon_deg_e,r.pred_lon_deg_e)-6;maxlon=max(r.jma_lon_deg_e,r.pred_lon_deg_e)+6;minlat=min(r.jma_lat_deg_n,r.pred_lat_deg_n)-6;maxlat=max(r.jma_lat_deg_n,r.pred_lat_deg_n)+6
   ax.set(xlim=(max(f.lon.min(),minlon),min(f.lon.max(),maxlon)),ylim=(max(f.lat.min(),minlat),min(f.lat.max(),maxlat)),xlabel='Longitude (E)',ylabel='Latitude (N)',title=f'{label}: DPE {r.track_error_km:.0f} km\nInit {r.init_utc[:16]} / +120h');ax.grid(alpha=.15)
   sources.append(dict(case_id=r.case_id,model=m,lead_hours=120,path=str(file.relative_to(ROOT))))
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(W/'figures/Figure_S6_HALONG_fields.png',dpi=220);plt.close(fig)
 pd.DataFrame(sources).to_csv(W/'provenance/field_diagnostic_sources.csv',index=False)
if __name__=='__main__':main()
