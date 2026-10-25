"""Deterministic GRIB tracking; retain historical differences and original inputs."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
import tracker_deterministic as tr
VARIANTS={'baseline':{},'radius500':{'base_search_radius_km':500.},'radius900':{'base_search_radius_km':900.},'motion0':{'motion_extrapolation_factor':0.},'motion1':{'motion_extrapolation_factor':1.},'wind0':{'wind_weight':0.}}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def job(args):
 root,path,variants=args;root=Path(root);path=Path(path);cfg=json.loads((root/'aliyun_2025_reproduction/config/tracker_config.json').read_text())['tracking'];old=pd.read_csv(path,dtype={'TC_ID':str});jid=path.parent.name;rows=[];man=[]
 assert json.loads((path.parent/'run_metadata.json').read_text())['config']['tracking']==cfg
 for m,z in old.groupby('model'):
  hist={(cid,v):[(g.iloc[0].jma_lat_deg_n,g.iloc[0].jma_lon_deg_e)] for cid,g in z[z.lead_hours==0].groupby('case_id') for v in variants}
  folder=root/'forecasts'/m/'by_init'/jid;expected={l.split()[1]:l.split()[0] for l in (folder/'SHA256SUMS.txt').read_text().splitlines() if l.strip()}
  for lead,g in z[z.lead_hours>0].groupby('lead_hours',sort=True):
   name=('fcn' if m=='fourcastnet' else m)+f'-{lead}.grib';p=folder/name;actual=sha(p);assert actual==expected[name]
   man.append(dict(path=str(p.relative_to(root)),sha256=actual,bytes=p.stat().st_size,original_checksum_match=True));fields=tr.read_grib_fields(p)
   elat=np.r_[fields.lat[0,:],fields.lat[-1,:],fields.lat[:,0],fields.lat[:,-1]];elon=np.r_[fields.lon[0,:],fields.lon[-1,:],fields.lon[:,0],fields.lon[:,-1]]
   for r in g.itertuples():
    for v,patch in variants.items():
     config={**cfg,**patch};key=(r.case_id,v);anchor=tr.motion_anchor(hist[key],config['motion_extrapolation_factor']);radius=min(config['maximum_search_radius_km'],config['base_search_radius_km']+(lead//6-1)*config['search_radius_growth_km_per_step'])
     cand=tr.find_candidates(fields,*anchor,radius,config);assert cand;best=cand[0];hist[key].append((best.lat,best.lon));distance=float(tr.haversine_km(best.lat,best.lon,r.jma_lat_deg_n,r.jma_lon_deg_e));ae=float(np.min(tr.haversine_km(*anchor,elat,elon)));ce=float(np.min(tr.haversine_km(best.lat,best.lon,elat,elon)))
     rows.append(dict(case_id=r.case_id,TC_ID=r.TC_ID,model=m,lead_hours=lead,variant=v,job_id=jid,status='TRACKED',pred_lat=best.lat,pred_lon=best.lon,DPE=distance,original_DPE=r.track_error_km,position_change_km=float(tr.haversine_km(best.lat,best.lon,r.pred_lat_deg_n,r.pred_lon_deg_e)),anchor_edge_km=ae,search_radius_km=radius,search_may_clip=ae<radius,center_edge_km=ce,wind_may_clip=ce<config['wind_radius_km'],wind_available=best.max_wind_ms is not None,source_file=str(p.relative_to(root))))
 return rows,man
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--baseline-only',action='store_true');a=ap.parse_args();a.root=a.root.resolve()
 for sub in ['data','provenance']:(a.out/sub).mkdir(parents=True,exist_ok=True)
 variants={'baseline':{}} if a.baseline_only else VARIANTS;paths=sorted((a.root/'results/tracks/by_job').glob('*/track_points.csv'));rows=[];man=[];start=time.time()
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,(rr,mm) in enumerate(pool.map(job,[(str(a.root),str(p),variants) for p in paths])):
   rows.extend(rr);man.extend(mm);print(f'Deterministic jobs {i+1}/{len(paths)}, files {len(man)}, seconds {time.time()-start:.0f}',flush=True)
 d=pd.DataFrame(rows).sort_values(['case_id','model','lead_hours','variant']);d.to_csv(a.out/'data/retracking_points.csv',index=False);pd.DataFrame(man).to_csv(a.out/'provenance/grib_manifest.csv',index=False)
 base=d[d.variant=='baseline'];assert len(base)==4896;changed=base[base.position_change_km>.02];changed.to_csv(a.out/'data/historical_track_differences.csv',index=False)
 originals=pd.concat([pd.read_csv(p,dtype={'TC_ID':str}) for p in paths]);new=originals.merge(base[['case_id','model','lead_hours','pred_lat','pred_lon','DPE']],on=['case_id','model','lead_hours'],how='left',validate='one_to_one');good=new.lead_hours>0
 new.loc[good,'pred_lat_deg_n']=new.loc[good,'pred_lat'];new.loc[good,'pred_lon_deg_e']=new.loc[good,'pred_lon'];new.loc[good,'track_error_km']=new.loc[good,'DPE'].round(2)
 columns=['model','case_id','TC_ID','name','forecast_job_id','lead_hours','valid_utc','pred_lat_deg_n','pred_lon_deg_e','jma_lat_deg_n','jma_lon_deg_e','track_error_km','status','source_file']
 new[columns].to_csv(a.out/'data/verified_track_points.csv',index=False)
 sources=[*paths,*[p.parent/'run_metadata.json' for p in paths],a.root/'aliyun_2025_reproduction/config/case_catalog.csv',a.root/'outputs/2025_jma_cases_v2/data/jma_2025_track_points.csv',a.root/'aliyun_2025_reproduction/config/case_truth_6h.csv',*list((a.root/'local_backup/weights').rglob('*'))];sources=[p for p in sources if p.is_file()]
 pd.DataFrame([dict(path=str(p.relative_to(a.root)),sha256=sha(p),bytes=p.stat().st_size) for p in sources]).to_csv(a.out/'provenance/source_manifest.csv',index=False)
 audit=dict(jobs=len(paths),grib_files=len(man),grib_sha256_all_match=True,baseline_rows=len(base),historical_changed_rows=len(changed),historical_max_position_change_km=float(base.position_change_km.max()),historical_max_DPE_change_km=float(abs(base.DPE-base.original_DPE).max()),search_may_clip_rows=int(base.search_may_clip.sum()),wind_may_clip_rows=int(base.wind_may_clip.sum()),variants=variants,tie_rule='ascending pressure, then ascending native flat grid index',seconds=time.time()-start)
 (a.out/'data/retracking_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
