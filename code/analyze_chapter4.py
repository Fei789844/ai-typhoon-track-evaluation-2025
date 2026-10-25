"""Chapter 4: reproducible descriptive/cluster inference; never reruns weather models.
Run from any directory: python /path/to/code/analyze_chapter4.py
All inputs are the immutable CSV copies under ../inputs.
"""
from pathlib import Path
from itertools import product, combinations
import hashlib, json, platform, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'inputs'; OUT=ROOT/'tables'
MODELS=['graphcast','pangu','fourcastnet']; METRICS=['DPE','absATE','absCTE','ATE','CTE']
PRIMARY=METRICS[:3]; LEADS=list(range(6,121,6)); B=20000; SEED=20261018
AUDIT={}; FAMILIES=[]

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rng(key): return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{SEED}|{key}'.encode()).digest()[:8],'little'))
def weights(n,key,exact=False):
    ix=np.array(list(product(range(n),repeat=n))) if exact else rng(key).integers(n,size=(B,n))
    w=np.zeros((len(ix),n))
    for k in range(n): w[:,k]=(ix==k).sum(axis=1)/n
    return w
def write(df,name):
    df.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
    return df
def xyz(lat,lon):
    p,l=np.deg2rad(lat),np.deg2rad(lon)
    return np.stack([np.cos(p)*np.cos(l),np.cos(p)*np.sin(l),np.sin(p)],axis=-1)
def basis(lat,lon):
    p,l=np.deg2rad(lat),np.deg2rad(lon)
    return np.array([-np.sin(l),np.cos(l),0]), np.array([-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)])
def distance(lat1,lon1,lat2,lon2):
    a,b=np.deg2rad(lat1),np.deg2rad(lat2); dl=np.deg2rad(np.asarray(lon2)-lon1)
    h=np.sin((b-a)/2)**2+np.cos(a)*np.cos(b)*np.sin(dl/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(np.clip(h,0,1)))
def bearing(lat1,lon1,lat2,lon2):
    a,b=np.deg2rad(lat1),np.deg2rad(lat2); dl=np.deg2rad(np.asarray(lon2)-lon1)
    return np.arctan2(np.sin(dl)*np.cos(b),np.cos(a)*np.sin(b)-np.sin(a)*np.cos(b)*np.cos(dl))

def intervals(mat,boot,family,sim=True):
    """mat: independent-cluster rows; joint mean boot draws in matching columns."""
    a=mat.to_numpy(); mu=a.mean(0); se=a.std(0,ddof=1)/np.sqrt(len(a))
    return interval_arrays(mat.columns,mu,se,boot,family,sim)
def interval_arrays(cols,mu,se,boot,family,sim=True):
    assert np.isfinite(mu).all() and np.isfinite(boot).all()
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    d=cols.to_frame(index=False);d['mean']=mu;d['low']=lo;d['high']=hi;d['SE']=se
    if sim:
        good=se>1e-12
        maxima=np.abs((boot[:,good]-mu[good])/se[good]).max(1)
        q=float(np.quantile(maxima,.95));d['sim_low']=mu-q*se;d['sim_high']=mu+q*se
        d['family']=family;d['family_size']=int(good.sum());d['critical_value']=q
        FAMILIES.append({'family':family,'coordinates':len(cols),'nonzero_SE':int(good.sum()),'B':len(boot),'q95':q})
    return d

def classify(truth):
    found=[]; candidates=[]
    for tc,g in truth.groupby('TC_ID',sort=True):
        g=g[g.jma_grade_code.isin([2,3,4,5])].sort_values('ts'); look=g.set_index('ts'); valid=[]
        for x in g.itertuples():
            if x.jma_grade_code<3:continue
            times=[x.ts+pd.Timedelta(hours=h) for h in range(-24,25,6)]
            if not all(t in look.index for t in times):continue
            z=look.loc[times[::2]];lon=np.rad2deg(np.unwrap(np.deg2rad(z.lon_deg_e.to_numpy())));lat=z.lat_deg_n.to_numpy()
            west=lon[0]-lon[2];east=lon[4]-lon[2];north1=lat[2]-lat[0];north2=lat[4]-lat[2]
            continuity=bool((np.diff(lon[:3])<=0).all() and (np.diff(lon[2:])>=0).all())
            passed=bool(west>=1 and east>=1 and north1>0 and north2>0 and continuity)
            candidates.append(dict(TC_ID=tc,time=x.ts,west24_deg=west,east24_deg=east,north_pre_deg=north1,north_post_deg=north2,direction_consistent=continuity,passed=passed))
            if passed:valid.append((lon[2],x.ts))
        turn=min(valid,key=lambda a:(a[0],a[1]))[1] if valid else pd.NaT
        found.append(dict(TC_ID=tc,name=g.name.iloc[0],recurving=bool(valid),turn_time=turn,threshold_deg=1.0))
    return pd.DataFrame(found),pd.DataFrame(candidates)

def load_and_geometry():
    raw=pd.read_csv(IN/'track_points.csv',dtype={'TC_ID':str});cat=pd.read_csv(IN/'case_catalog.csv',dtype={'TC_ID':str})
    truth=pd.read_csv(IN/'jma_track_points.csv',dtype={'TC_ID':str})
    truth['ts']=pd.to_datetime(truth.time_utc.str.replace(' UTC',''),utc=True)
    assert not truth.duplicated(['TC_ID','ts']).any()
    truth=truth[truth.ts.dt.hour.mod(6).eq(0)]
    cl,candidates=classify(truth);write(cl,'turn_classification');write(candidates,'turn_candidates')
    old=pd.read_csv(IN/'previous_classes.csv',dtype={'TC_ID':str}); merged=cl.merge(old,on='TC_ID',suffixes=('','_old'))
    assert (merged.recurving==merged.recurving_old).all()
    assert (pd.to_datetime(merged.turn_time,utc=True).fillna(pd.Timestamp('1900',tz='UTC'))==pd.to_datetime(merged.turn_time_old,utc=True).fillna(pd.Timestamp('1900',tz='UTC'))).all()
    d=raw.merge(cat[['case_id','case_type','init_utc']],on='case_id',validate='many_to_one')
    d=d[d.lead_hours.gt(0)&d.case_type.eq('FULL_120H')].copy()
    assert len(d)==4440 and d.case_id.nunique()==74 and d.TC_ID.nunique()==14
    assert not d.duplicated(['model','case_id','lead_hours']).any()
    keys=[set(zip(g.case_id,g.lead_hours)) for _,g in d.groupby('model')];assert all(k==keys[0] for k in keys)
    d['valid_time']=pd.to_datetime(d.valid_utc.str.replace(' UTC',''),utc=True); d['init_time']=pd.to_datetime(d.init_utc.str.replace(' UTC',''),utc=True)
    assert ((d.valid_time-d.init_time).dt.total_seconds()/3600==d.lead_hours).all()
    look={(r.TC_ID,r.ts):(r.lat_deg_n,r.lon_deg_e) for r in truth.itertuples()}
    assert all(look[(r.TC_ID,r.valid_time)]==(r.jma_lat_deg_n,r.jma_lon_deg_e) for r in d.itertuples())
    recalc=distance(d.pred_lat_deg_n,d.pred_lon_deg_e,d.jma_lat_deg_n,d.jma_lon_deg_e)
    diff=np.abs(recalc-d.track_error_km); assert diff.max()<.00501
    d['DPE']=d.track_error_km
    geometry=[]
    turns=cl.set_index('TC_ID').turn_time.to_dict()
    for r in d[['TC_ID','valid_time','jma_lat_deg_n','jma_lon_deg_e']].drop_duplicates().itertuples():
        now=(r.jma_lat_deg_n,r.jma_lon_deg_e); prior=look.get((r.TC_ID,r.valid_time-pd.Timedelta(hours=12)))
        backdist=float(distance(*prior,*now)) if prior is not None else np.nan
        alpha=float(bearing(*now,*prior)+np.pi) if prior is not None and backdist>=10 else np.nan
        p=look.get((r.TC_ID,r.valid_time-pd.Timedelta(hours=6)));q=look.get((r.TC_ID,r.valid_time+pd.Timedelta(hours=6)))
        central=np.nan
        e,n=basis(*now)
        if p is not None and q is not None and distance(*p,*q)>=10:
            v=xyz(*q)-xyz(*p); central=np.arctan2(v@e,v@n)
        fixed=np.nan;turn=turns[r.TC_ID]
        if pd.notna(turn):
            p0=look[(r.TC_ID,turn)];prev0=look.get((r.TC_ID,turn-pd.Timedelta(hours=12)))
            if prev0 is not None and distance(*prev0,*p0)>=10:
                a0=bearing(*p0,*prev0)+np.pi;e0,n0=basis(*p0);v0=np.sin(a0)*e0+np.cos(a0)*n0
                u0,u=xyz(*p0),xyz(*now);axis=np.cross(u0,u);s=np.linalg.norm(axis);c=np.clip(u0@u,-1,1)
                if s<1e-12: transported=v0
                else:
                    axis/=s;transported=v0*c+np.cross(axis,v0)*s+axis*(axis@v0)*(1-c)
                assert abs(transported@u)<1e-9 and abs(np.linalg.norm(transported)-1)<1e-9
                fixed=np.arctan2(transported@e,transported@n)
        geometry.append(dict(TC_ID=r.TC_ID,valid_time=r.valid_time,alpha=alpha,back12_distance=backdist,alpha_center=central,alpha_fixed=fixed))
    d=d.merge(pd.DataFrame(geometry),on=['TC_ID','valid_time'],validate='many_to_one')
    beta=bearing(d.jma_lat_deg_n,d.jma_lon_deg_e,d.pred_lat_deg_n,d.pred_lon_deg_e)
    for suffix,angle in [('',d.alpha),('_center',d.alpha_center),('_fixed',d.alpha_fixed)]:
        d['ATE'+suffix]=d.DPE*np.cos(beta-angle);d['CTE'+suffix]=d.DPE*np.sin(beta-angle)
        d['absATE'+suffix]=d['ATE'+suffix].abs();d['absCTE'+suffix]=d['CTE'+suffix].abs()
        ok=d['ATE'+suffix].notna();assert np.allclose(d.loc[ok,'ATE'+suffix]**2+d.loc[ok,'CTE'+suffix]**2,d.loc[ok,'DPE']**2,atol=1e-6,rtol=1e-10)
    d=d.merge(cl[['TC_ID','recurving','turn_time']],on='TC_ID',validate='many_to_one')
    d['turn_relative_h']=(d.valid_time-d.turn_time).dt.total_seconds()/3600
    d['init_relative_h']=(d.init_time-d.turn_time).dt.total_seconds()/3600
    write(d,'analysis_records');write(d[d.ATE.isna()],'excluded_direction_records')
    storms=d[['TC_ID','name','recurving','turn_time']].drop_duplicates()
    c=d[['TC_ID','case_id','init_time','turn_time']].drop_duplicates();c['contains_turn']=c.turn_time.between(c.init_time,c.init_time+pd.Timedelta(hours=120))
    counts=c.groupby('TC_ID').agg(n_cases=('case_id','size'),n_cases_containing_turn=('contains_turn','sum')).reset_index()
    write(storms.merge(counts,on='TC_ID'),'storm_catalog')
    AUDIT.update(n_records=len(d),n_storms=d.TC_ID.nunique(),n_cases=d.case_id.nunique(),n_turn_storms=storms.recurving.sum().item(),distance_max_discrepancy_km=float(diff.max()),excluded_direction_rows=int(d.ATE.isna().sum()),classification_matches_archive=True)
    return d,truth

def matrix(d,metrics=METRICS):
    long=d.melt(id_vars=['TC_ID','model','lead_hours'],value_vars=metrics,var_name='metric',value_name='value')
    return long.groupby(['TC_ID','model','lead_hours','metric']).value.mean().unstack(['model','lead_hours','metric']).sort_index(axis=1)

def overall(d):
    mat=matrix(d);w=weights(len(mat),'overall');boot=w@mat.to_numpy()
    stats=intervals(mat,boot,'overall_marginals',sim=False)
    counts=d.melt(id_vars=['TC_ID','model','lead_hours'],value_vars=METRICS,var_name='metric',value_name='value').dropna().groupby(['model','lead_hours','metric']).agg(n_cases=('value','size'),n_storms=('TC_ID','nunique')).reset_index()
    write(stats.merge(counts,on=['model','lead_hours','metric']),'overall_by_lead')
    old=pd.read_csv(IN/'previous_components.csv').query("setting=='primary' and metric in ['ATE','CTE','absATE','absCTE']")
    check=stats.merge(old,on=['model','lead_hours','metric'],suffixes=('','_old'));err=np.abs(check['mean']-check.mean_old).max();assert err<1e-8
    AUDIT['component_mean_max_discrepancy_from_archive_km']=float(err)
    aa=mat.xs('absATE',level='metric',axis=1);cc=mat.xs('absCTE',level='metric',axis=1);delta=aa-cc
    res=intervals(delta,w@delta.to_numpy(),'overall_absATE_minus_absCTE_60');write(res,'direction_difference')
    sq=d.assign(a2=d.ATE**2,c2=d.CTE**2).groupby(['TC_ID','model','lead_hours'])[['a2','c2']].mean()
    fractions=[]
    for (m,h),g in sq.groupby(['model','lead_hours']):
        g=g.droplevel(['model','lead_hours']).reindex(mat.index); a,c=g.a2.to_numpy(),g.c2.to_numpy();b=(w@a)/(w@(a+c));lo,hi=np.quantile(b,[.025,.975])
        fractions.append(dict(model=m,lead_hours=h,along_fraction=a.mean()/(a+c).mean(),low=lo,high=hi))
    write(pd.DataFrame(fractions),'along_squared_fraction')
    growth=[]
    for (tc,m),g in d.groupby(['TC_ID','model']):
        g=g[g.lead_hours.isin([24,120])].dropna(subset=PRIMARY)
        a=g.pivot(index='case_id',columns='lead_hours',values=PRIMARY).dropna()
        for metric in PRIMARY:
            v=(a[(metric,120)]-a[(metric,24)]).mean()
            growth.append(dict(TC_ID=tc,model=m,metric=metric,change=v,n_matched_cases=len(a)))
    g=pd.DataFrame(growth);write(g,'endpoint_growth_storms');a=g.pivot(index='TC_ID',columns=['model','metric'],values='change').sort_index(axis=1)
    r=intervals(a,w@a.to_numpy(),'growth_24_to_120_9');r['mean_per_hour']=r['mean']/96;write(r,'endpoint_growth')

def event_cells(d,window=24,before=True,kind='',common=False):
    z=d[d.recurving & d.turn_relative_h.between(-window,window)].copy()
    if before:z=z[z.init_relative_h<0]
    if common:z=z.dropna(subset=['ATE','CTE','ATE_center','CTE_center','ATE_fixed','CTE_fixed'])
    if kind:
        for k in ['ATE','CTE','absATE','absCTE']:z[k]=z[k+kind]
    z=z.dropna(subset=METRICS)
    z['phase']=np.select([z.turn_relative_h<=-12,z.turn_relative_h>=12],['pre','post'],default='near')
    cells=z.groupby(['TC_ID','name','model','lead_hours','phase'])[METRICS].mean().reset_index()
    cnt=z.groupby(['TC_ID','model','lead_hours','phase']).agg(n_cases=('case_id','size'),min_rel=('turn_relative_h','min'),max_rel=('turn_relative_h','max')).reset_index()
    return z,cells.merge(cnt,on=['TC_ID','model','lead_hours','phase'])

def matched(cells,left='pre',right='post'):
    idx=['TC_ID','name','model','lead_hours']
    a=cells[cells.phase==left];b=cells[cells.phase==right]
    joined=a.merge(b,on=idx,suffixes=('_pre','_post'),validate='one_to_one')
    out=[]
    for k in METRICS:
        z=joined[idx].copy();z['metric']=k;z['before']=joined[k+'_pre'];z['after']=joined[k+'_post'];z['delta']=z.after-z.before
        z['n_pre']=joined.n_cases_pre;z['n_post']=joined.n_cases_post;out.append(z)
    pairs=pd.concat(out,ignore_index=True)
    storm=pairs.groupby(['TC_ID','name','model','metric']).agg(delta=('delta','mean'),before=('before','mean'),after=('after','mean'),n_matched_leads=('lead_hours','size'),min_lead=('lead_hours','min'),max_lead=('lead_hours','max')).reset_index()
    return pairs,storm

def event_analysis(d):
    records,cells=event_cells(d);write(records,'event_records');write(cells,'event_stage_cells')
    pairs,storm=matched(cells);write(pairs,'event_matched_lead_differences');write(storm,'event_storm_effects')
    # Three-phase view uses identical storm/lead support in all three phases.
    key=['TC_ID','model','lead_hours']
    triple_keys=cells.groupby(key).phase.nunique().loc[lambda x:x==3].reset_index()[key]
    triple=cells.merge(triple_keys,on=key,validate='many_to_one')
    triple_storm=triple.groupby(['TC_ID','name','model','phase'])[METRICS].mean().reset_index()
    write(triple,'event_three_phase_cells');write(triple_storm,'event_three_phase_storms')
    phase_long=triple_storm.melt(id_vars=['TC_ID','name','model','phase'],value_vars=METRICS,var_name='metric',value_name='value')
    phase_mat=phase_long.pivot(index='TC_ID',columns=['model','phase','metric'],values='value').sort_index(axis=1)
    assert len(phase_mat)==5 and phase_mat.notna().all().all()
    write(intervals(phase_mat,weights(5,'phase',exact=True)@phase_mat.to_numpy(),'three_phase_descriptive',sim=False),'event_three_phase_means')
    support=pairs[(pairs.model=='graphcast')&(pairs.metric=='DPE')][['TC_ID','name','lead_hours','n_pre','n_post']];write(support,'event_support')
    write(support.groupby('lead_hours').agg(n_storms=('TC_ID','nunique')).reset_index(),'event_support_by_lead')
    a=storm[storm.metric.isin(PRIMARY)].pivot(index='TC_ID',columns=['model','metric'],values='delta').sort_index(axis=1)
    assert len(a)==5 and a.notna().all().all();w=weights(5,'event',exact=True)
    res=intervals(a,w@a.to_numpy(),'event_post_minus_pre_9');res['n_storms']=5
    signs=np.array(list(product([-1,1],repeat=5)));permuted=signs@a.to_numpy()/5
    mu=a.mean().to_numpy();se=a.std(ddof=1).to_numpy()/np.sqrt(5)
    res['signflip_p']=(np.abs(permuted)>=np.abs(mu)[None,:]-1e-10).mean(0)
    maxima=np.abs(permuted/se).max(1);res['signflip_p_maxT']=(maxima[:,None]>=np.abs(mu/se)[None,:]-1e-10).mean(0)
    res['n_positive_storms']=(a.to_numpy()>0).sum(0);write(res,'event_primary_inference')
    write(pd.DataFrame({'sign_pattern':[' '.join(map(str,s)) for s in signs],'max_abs_t':maxima}),'event_signflip_distribution')
    other=storm[~storm.metric.isin(PRIMARY)].pivot(index='TC_ID',columns=['model','metric'],values='delta').sort_index(axis=1)
    write(intervals(other,w@other.to_numpy(),'event_signed_descriptive',sim=False),'event_signed_changes')
    # Raw phase summaries retain forecast-lead distribution as an explicit confounding check.
    raw=[]
    for (phase,m),g in records.groupby(['phase','model']):
        for k in METRICS:
            x=g.groupby('TC_ID')[k].mean();raw.append(dict(phase=phase,model=m,metric=k,mean=x.mean(),n_storms=len(x),n_records=len(g),lead_min=g.lead_hours.min(),lead_median=g.lead_hours.median(),lead_max=g.lead_hours.max()))
    write(pd.DataFrame(raw),'event_unmatched_phase_descriptive')
    trans=[]
    for left,right in [('pre','near'),('near','post')]:
        p,s=matched(cells,left,right);s['contrast']=right+'-'+left;trans.append(s)
    write(pd.concat(trans),'event_near_transitions_storms')
    # All prespecified sensitivities, not selected by sign or interval.
    configs=[('baseline',24,True,'',False),('window48',48,True,'',False),('all_initializations',24,False,'',False),('common_back12',24,True,'',True),('common_center12',24,True,'_center',True),('common_fixed_turn_basis',24,True,'_fixed',True)]
    sensitivity=[];detail=[]
    for label,win,bef,kind,common in configs:
        r,c=event_cells(d,win,bef,kind,common);p,s=matched(c);s['setting']=label;detail.append(s)
        for (m,k),g in s.groupby(['model','metric']):
            sensitivity.append(dict(setting=label,model=m,metric=k,mean=g.delta.mean(),n_storms=len(g),minimum_storm_change=g.delta.min(),maximum_storm_change=g.delta.max()))
        write(p,f'event_matching_{label}')
    write(pd.DataFrame(sensitivity),'event_sensitivity');write(pd.concat(detail),'event_sensitivity_storms')
    loco=[]
    for tc in a.index:
        for (m,k),v in a.drop(tc).mean().items():loco.append(dict(omitted=tc,model=m,metric=k,mean=v))
    write(pd.DataFrame(loco),'event_loco')
    AUDIT.update(event_n_storms=5,event_bootstrap_sequences=len(w),event_max_storms_at_one_lead=int(support.groupby('lead_hours').TC_ID.nunique().max()),event_matched_storm_lead_cells=len(support),event_primary_records=len(records))

def group_analysis(d):
    z=d.dropna(subset=PRIMARY);a=matrix(z,PRIMARY);cl=d[['TC_ID','recurving']].drop_duplicates().set_index('TC_ID').recurving
    t=a.loc[cl[cl].index];u=a.loc[cl[~cl].index];w1=weights(len(t),'groups_turn');w0=weights(len(u),'groups_other')
    bt=w1@t.to_numpy();bu=w0@u.to_numpy()
    r1=intervals(t,bt,'group_means',sim=False);r1['recurving']=True;r0=intervals(u,bu,'group_means',sim=False);r0['recurving']=False
    counts=z.groupby(['recurving','model','lead_hours']).agg(n_cases=('case_id','size'),n_storms=('TC_ID','nunique')).reset_index()
    write(pd.concat([r1,r0]).merge(counts,on=['recurving','model','lead_hours']),'group_means')
    se=np.sqrt(t.var(ddof=1)/len(t)+u.var(ddof=1)/len(u)).to_numpy()
    diff=interval_arrays(a.columns,(t.mean()-u.mean()).to_numpy(),se,bt-bu,'path_group_difference_180');write(diff,'group_difference')
    # Preserve exactly the same group bootstrap draws across model pairs and all leads.
    means=[];ses=[];boots=[];meta=[];interaction_mu=[];interaction_se=[];interaction_boot=[];imeta=[]
    for ma,mb in combinations(MODELS,2):
        for h in LEADS:
            for k in PRIMARY:
                i=a.columns.get_loc((ma,h,k));j=a.columns.get_loc((mb,h,k));td=t.iloc[:,i]-t.iloc[:,j];ud=u.iloc[:,i]-u.iloc[:,j]
                for group,v,b in [('turn',td,bt[:,i]-bt[:,j]),('other',ud,bu[:,i]-bu[:,j])]:
                    meta.append((group,f'{ma}-{mb}',h,k));means.append(v.mean());ses.append(v.std(ddof=1)/np.sqrt(len(v)));boots.append(b)
                imeta.append((f'{ma}-{mb}',h,k));interaction_mu.append(td.mean()-ud.mean());interaction_se.append(np.sqrt(td.var(ddof=1)/len(td)+ud.var(ddof=1)/len(ud)));interaction_boot.append((bt[:,i]-bt[:,j])-(bu[:,i]-bu[:,j]))
    cols=pd.MultiIndex.from_tuples(meta,names=['group','pair','lead_hours','metric'])
    write(interval_arrays(cols,np.array(means),np.array(ses),np.array(boots).T,'within_group_model_pairs_360'),'within_group_model_pairs')
    cols=pd.MultiIndex.from_tuples(imeta,names=['pair','lead_hours','metric'])
    write(interval_arrays(cols,np.array(interaction_mu),np.array(interaction_se),np.array(interaction_boot).T,'model_by_path_interaction_180'),'group_interactions')
    loco=[]
    for tc in a.index:
        t1=t.drop(tc,errors='ignore');u1=u.drop(tc,errors='ignore')
        for (m,h,k),v in (t1.mean()-u1.mean()).items():
            if h==120:loco.append(dict(omitted=tc,model=m,metric=k,mean=v,n_turn=len(t1),n_other=len(u1)))
    write(pd.DataFrame(loco),'group_loco120')
    base=pd.read_csv(IN/'previous_group_contrasts.csv');now=diff.query("metric=='DPE' and lead_hours==120").merge(base.query('lead_hours==120'),on=['model','lead_hours'],suffixes=('','_old'))
    assert np.allclose(now['mean'],now.mean_old,atol=1e-8,rtol=0)
    AUDIT['group120_matches_archive']=True

def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    # Validate source files before all analysis; the manifest contains exact copied input hashes.
    manifest=json.loads((ROOT/'audit/input_manifest.json').read_text())
    for item in manifest['files']:assert digest(IN/item['file'])==item['sha256']
    d,truth=load_and_geometry();overall(d);event_analysis(d);group_analysis(d)
    AUDIT.update(status='passed',seed=SEED,B=B,seconds=time.time()-start,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'docs/第四章分析方案.md'),inference_families=FAMILIES)
    (ROOT/'audit/analysis_audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2))
    print(json.dumps(AUDIT,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
