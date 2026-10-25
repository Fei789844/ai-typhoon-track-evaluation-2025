"""Independent 3-D tangent-vector check of bearing-based ATE/CTE."""
from pathlib import Path
import json
import numpy as np,pandas as pd
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1]
def vec(lat,lon):
 a,b=np.deg2rad([lat,lon]);return np.array([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)])
def main():
 d=pd.read_csv(W/'data/error_components_cases.csv',dtype={'TC_ID':str});d=d[d.ATE.notna()]
 t=pd.read_csv(ROOT/'outputs/2025_jma_cases_v2/data/jma_2025_track_points.csv',dtype={'TC_ID':str});t['time']=pd.to_datetime(t.time_utc.str.replace(' UTC',''),utc=True);lookup=t.set_index(['TC_ID','time'])
 err=[]
 for r in d.itertuples():
  time=pd.to_datetime(r.valid_utc.replace(' UTC',''),utc=True);p=lookup.loc[(r.TC_ID,time-pd.Timedelta(hours=12))]
  now=vec(r.jma_lat_deg_n,r.jma_lon_deg_e);old=vec(p.lat_deg_n,p.lon_deg_e);pred=vec(r.pred_lat_deg_n,r.pred_lon_deg_e)
  tangent=np.dot(old,now)*now-old;tangent/=np.linalg.norm(tangent);right=np.cross(tangent,now)
  error=pred-np.dot(pred,now)*now;norm=np.linalg.norm(error)
  if norm<1e-12:
   assert r.track_error_km<.01;continue
  error=error/norm*r.track_error_km
  err.append([abs(error@tangent-r.ATE),abs(error@right-r.CTE)])
 maximum=np.max(err,axis=0);assert max(maximum)<1e-7
 result={'records_checked':len(err),'method':'independent Cartesian tangent vectors','max_ATE_difference_km':maximum[0],'max_CTE_difference_km':maximum[1]}
 (W/'provenance/geometry_verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
