"""Reproducible exploratory cyclone analyses 3-6. See protocol.txt.
Run: python experiments3to6.py --project-root PROJECT --output-dir OUTPUT
Requires experiment1.py, numpy, pandas, matplotlib. No weather-model reruns.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment1 import load_data, haversine
MODELS=['graphcast','pangu','fourcastnet']
LABELS=['GraphCast','Pangu-Weather','FourCastNet']
COLORS=['#009E73','#D55E00','#0072B2']
LEADS=np.arange(6,121,6); KEY=[24,48,72,96,120]
B=10000;SEED=20261004

def rng_for(key):
    seed=int.from_bytes(hashlib.sha256(f'{SEED}|{key}'.encode()).digest()[:8],'little')
    return np.random.default_rng(seed)

def ci(v,key):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    if len(v)<2:return [np.nan,np.nan]
    idx=rng_for(key).integers(len(v),size=(B,len(v)))
    return np.quantile(v[idx].mean(1),[.025,.975])

def xyz(lat,lon):
    lat,lon=np.deg2rad(lat),np.deg2rad(lon)
    return np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],axis=-1)

def bearing(lat1,lon1,lat2,lon2):
    p1,p2=np.deg2rad(lat1),np.deg2rad(lat2);dl=np.deg2rad(np.asarray(lon2)-lon1)
    return np.arctan2(np.sin(dl)*np.cos(p2),np.cos(p1)*np.sin(p2)-np.sin(p1)*np.cos(p2)*np.cos(dl))

def geometry(data,truth,min_distance=10):
    truth=truth.copy();truth['ts']=pd.to_datetime(truth.time_utc.str.replace(' UTC',''),utc=True)
    truth=truth[truth.ts.dt.hour.mod(6).eq(0)].drop_duplicates(['TC_ID','ts'])
    lookup={(r.TC_ID,r.ts):(r.lat_deg_n,r.lon_deg_e) for r in truth.itertuples()}
    bearings=[]
    for r in data[['TC_ID','valid_utc','jma_lat_deg_n','jma_lon_deg_e']].drop_duplicates().itertuples():
        t=pd.to_datetime(r.valid_utc.replace(' UTC',''),utc=True);now=(r.jma_lat_deg_n,r.jma_lon_deg_e)
        vals={}
        for kind in ['back12','center12']:
            a=lookup.get((r.TC_ID,t-pd.Timedelta(hours=12 if kind=='back12' else 6)))
            b=now if kind=='back12' else lookup.get((r.TC_ID,t+pd.Timedelta(hours=6)))
            heading=np.nan;distance=np.nan
            if a is not None and b is not None:
                distance=float(haversine(*a,*b))
                if distance>=min_distance:
                    if kind=='back12':heading=float(bearing(*now,*a)+np.pi)
                    else:
                        v=xyz(*b)-xyz(*a);la,lo=np.deg2rad(now)
                        east=np.array([-np.sin(lo),np.cos(lo),0]);north=np.array([-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)])
                        heading=float(np.arctan2(v@east,v@north))
            vals[kind]=heading;vals[kind+'_distance']=distance
        bearings.append(dict(TC_ID=r.TC_ID,valid_utc=r.valid_utc,**vals))
    d=data.merge(pd.DataFrame(bearings),on=['TC_ID','valid_utc'],validate='many_to_one')
    err=bearing(d.jma_lat_deg_n,d.jma_lon_deg_e,d.pred_lat_deg_n,d.pred_lon_deg_e)
    for k in ['back12','center12']:
        theta=err-d[k];d['ATE_'+k]=d.track_error_km*np.cos(theta);d['CTE_'+k]=d.track_error_km*np.sin(theta)
        good=d[k].notna();assert np.allclose(d.loc[good,'ATE_'+k]**2+d.loc[good,'CTE_'+k]**2,d.loc[good,'track_error_km']**2,rtol=1e-10,atol=1e-6)
    d['ATE']=d.ATE_back12;d['CTE']=d.CTE_back12;d['absATE']=d.ATE.abs();d['absCTE']=d.CTE.abs();d['ATE2']=d.ATE**2;d['CTE2']=d.CTE**2
    return d,truth

def components(d,out):
    rows=[]
    for setting in ['primary','all_available']:
        z=d[d.case_type=='FULL_120H'] if setting=='primary' else d
        z=z[z.ATE.notna()]
        for (lead,m),g in z.groupby(['lead_hours','model']):
            s=g.groupby('TC_ID')[['ATE','CTE','absATE','absCTE','ATE2','CTE2']].mean()
            for metric in ['ATE','CTE','absATE','absCTE']:
                low,high=ci(s[metric],f'comp|{setting}|{lead}|{m}|{metric}')
                rows.append(dict(setting=setting,lead_hours=lead,model=m,metric=metric,mean=s[metric].mean(),low=low,high=high,n_storms=len(s),n_cases=len(g)))
            # Ratio of equal-cyclone mean squared components, not ratio of means.
            frac=s.CTE2.mean()/(s.ATE2.mean()+s.CTE2.mean())
            rows.append(dict(setting=setting,lead_hours=lead,model=m,metric='CTE_squared_fraction',mean=frac,low=np.nan,high=np.nan,n_storms=len(s),n_cases=len(g)))
    r=pd.DataFrame(rows);r.to_csv(out/'data/components.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(9,7.6),sharex=True)
    for ax,met,title in zip(axs.flat,['absATE','absCTE','ATE','CTE'],['Mean absolute along-track error','Mean absolute cross-track error','Signed along-track bias (+ ahead)','Signed cross-track bias (+ right)']):
        for m,l,c in zip(MODELS,LABELS,COLORS):
            g=r[(r.setting=='primary')&(r.model==m)&(r.metric==met)]
            ax.plot(g.lead_hours,g['mean'],color=c,label=l);ax.fill_between(g.lead_hours,g.low,g.high,color=c,alpha=.12)
        ax.set_title(title,fontsize=10);ax.set_ylabel('km');ax.grid(alpha=.2);ax.axhline(0,color='black',lw=.5)
    for ax in axs[-1]:ax.set_xlabel('Lead time (h)');ax.set_xticks([24,48,72,96,120])
    outside_legend(fig,axs[0,0]);save(fig,out,'Figure_3_components')
    return r

def classify(truth,threshold=1.):
    rows=[]
    for tc,g in truth.groupby('TC_ID'):
        g=g.sort_values('ts');g=g[g.jma_grade_code.isin([2,3,4,5])];lookup=g.set_index('ts')
        eligible=[]
        for r in g.itertuples():
            if r.jma_grade_code<3:continue
            times=[r.ts+pd.Timedelta(hours=h) for h in [-24,-12,0,12,24]]
            continuous=[r.ts+pd.Timedelta(hours=h) for h in range(-24,25,6)]
            if not all(t in lookup.index for t in continuous):continue
            a=lookup.loc[times];lon=np.rad2deg(np.unwrap(np.deg2rad(a.lon_deg_e.to_numpy())));lat=a.lat_deg_n.to_numpy()
            if (lon[2]-lon[0]<=-threshold and lon[4]-lon[2]>=threshold and lat[2]>lat[0] and lat[4]>lat[2] and np.all(np.diff(lon[:3])<=0) and np.all(np.diff(lon[2:])>=0)):
                eligible.append((lon[2],r.ts))
        turn=min(eligible,key=lambda x:(x[0],x[1]))[1] if eligible else pd.NaT
        rows.append(dict(TC_ID=tc,name=g.name.iloc[0],recurving=bool(eligible),turn_time=turn,threshold_deg=threshold))
    return pd.DataFrame(rows)

def stratified(d,truth,out):
    cl=classify(truth);cl.to_csv(out/'data/cyclone_path_types.csv',index=False)
    full=d[d.case_type=='FULL_120H'].merge(cl[['TC_ID','recurving','turn_time']],on='TC_ID',validate='many_to_one')
    full['turn_relative_h']=(pd.to_datetime(full.valid_utc.str.replace(' UTC',''),utc=True)-pd.to_datetime(full.turn_time,utc=True)).dt.total_seconds()/3600
    rows=[];pairs=[];growth=[]
    for (kind,lead,m),g in full.groupby(['recurving','lead_hours','model']):
        s=g.groupby('TC_ID').track_error_km.mean();lo,hi=ci(s,f'group{kind}{lead}{m}');rows.append(dict(recurving=kind,lead_hours=lead,model=m,mean=s.mean(),low=lo,high=hi,n_storms=len(s),n_cases=len(g)))
    for kind,g in full.groupby('recurving'):
        ss=g.groupby(['TC_ID','lead_hours','model']).track_error_km.mean().unstack('model')
        for lead in KEY:
            v=ss.xs(lead,level='lead_hours');diff=v.graphcast-v.pangu;lo,hi=ci(diff,f'group_pair{kind}{lead}')
            pairs.append(dict(recurving=kind,lead_hours=lead,mean=diff.mean(),low=lo,high=hi,n_storms=len(diff)))
        for m in MODELS:
            v=ss[m].unstack('lead_hours');rate=(v[120]-v[24])/96;lo,hi=ci(rate,f'growth{kind}{m}')
            growth.append(dict(recurving=kind,model=m,mean=rate.mean(),low=lo,high=hi,n_storms=len(rate)))
    r=pd.DataFrame(rows);r.to_csv(out/'data/path_type_errors.csv',index=False);pd.DataFrame(pairs).to_csv(out/'data/path_type_GC_Pangu.csv',index=False);pd.DataFrame(growth).to_csv(out/'data/path_type_growth.csv',index=False)
    # Within each lead, compare near-turn and other cases descriptively.
    near=[]
    for (lead,m),g in full.groupby(['lead_hours','model']):
        for kind,z in g.groupby(g.turn_relative_h.abs().le(12)):
            v=z.groupby('TC_ID').track_error_km.mean()
            near.append(dict(lead_hours=lead,model=m,near_turn=kind,n_cases=len(z),n_storms=len(v),mean=v.mean()))
    pd.DataFrame(near).to_csv(out/'data/near_turn_errors.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,4.6),sharey=True)
    for ax,kind,title in zip(axs,[False,True],['No recurvature detected','Recurving cyclones']):
        for m,l,c in zip(MODELS,LABELS,COLORS):
            g=r[(r.recurving==kind)&(r.model==m)];ax.plot(g.lead_hours,g['mean'],color=c,label=l);ax.fill_between(g.lead_hours,g.low,g.high,color=c,alpha=.12)
        info=r[(r.recurving==kind)&(r.lead_hours==24)].iloc[0]
        ax.set_title(f'{title}\n{int(info.n_storms)} cyclones, {int(info.n_cases)} cases',fontsize=11);ax.set_xlabel('Lead time (h)');ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean DPE (km)');outside_legend(fig,axs[0]);save(fig,out,'Figure_4_path_types')
    # Track map is a schematic in lon-lat, no invented coastlines.
    fig,ax=plt.subplots(figsize=(9,5.5))
    fullids=set(full.TC_ID)
    for tc,g in truth[truth.TC_ID.isin(fullids)&truth.jma_grade_code.isin([2,3,4,5])].groupby('TC_ID'):
        row=cl[cl.TC_ID==tc].iloc[0];col='#D55E00' if row.recurving else '#0072B2';ax.plot(g.lon_deg_e,g.lat_deg_n,color=col,alpha=.65);ax.text(g.lon_deg_e.iloc[-1],g.lat_deg_n.iloc[-1],str(tc),fontsize=7)
        if row.recurving:
            v=g[g.ts==row.turn_time].iloc[0];ax.plot(v.lon_deg_e,v.lat_deg_n,'o',color=col,ms=5)
    ax.set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',title='Observed tracks: orange = recurving; dots = diagnosed turn');ax.grid(alpha=.2);fig.tight_layout();save(fig,out,'Figure_S4_track_types')
    return full,cl

def wquantile(x,w,q):
    order=np.argsort(x);x=np.asarray(x)[order];w=np.asarray(w)[order];return float(x[np.searchsorted(np.cumsum(w)/sum(w),q,side='left')])

def wranks(x,w):
    # Weighted mid-distribution ranks. w may contain bootstrap multiplicities.
    order=np.argsort(x,kind='stable');x=np.asarray(x)[order];sw=w[:,order];starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1];ends=np.r_[starts[1:],len(x)]
    gw=np.add.reduceat(sw,starts,axis=1);gr=np.cumsum(gw,axis=1)-gw/2
    ranks=np.repeat(gr,ends-starts,axis=1);return ranks[:,np.argsort(order)]

def wcorr(x,y,w):
    x=wranks(x,w);y=wranks(y,w);total=w.sum(1)[:,None];mx=(w*x).sum(1)[:,None]/total;my=(w*y).sum(1)[:,None]/total
    xc=x-mx;yc=y-my;den=np.sqrt((w*xc**2).sum(1)*(w*yc**2).sum(1));return np.divide((w*xc*yc).sum(1),den,out=np.full(len(w),np.nan),where=den>0)

def base_weights(g):
    return (1/g.groupby('TC_ID').TC_ID.transform('size')).to_numpy()

def score_metrics(g):
    w=base_weights(g);y=g.event.to_numpy(bool);a=g.alert.to_numpy(bool);tp=sum(w*y*a);fn=sum(w*y*~a);fp=sum(w*~y*a);total=sum(w)
    pos=y;neg=~y;score=g.risk_score.to_numpy();den=sum(w[pos])*sum(w[neg]);cmp=(score[pos,None]>score[None,neg])+0.5*(score[pos,None]==score[None,neg])
    auc=float((cmp*w[pos,None]*w[None,neg]).sum()/den) if den else np.nan
    return dict(event_rate=float((tp+fn)/total),precision=float(tp/(tp+fp)) if tp+fp else np.nan,POD=float(tp/(tp+fn)) if tp+fn else np.nan,FAR=float(fp/(tp+fp)) if tp+fp else np.nan,AUC=auc,event_cases=int(y.sum()),alert_cases=int(a.sum()),n_cases=len(g))

def disagreement(d,out):
    full=d[d.case_type=='FULL_120H'];keys=['TC_ID','case_id','lead_hours']
    a=full.pivot(index=keys,columns='model',values=['pred_lat_deg_n','pred_lon_deg_e','track_error_km'])
    truth=full.groupby(keys)[['jma_lat_deg_n','jma_lon_deg_e']].first();dist=[];vec=[]
    for i,m in enumerate(MODELS):
        vec.append(xyz(a['pred_lat_deg_n'][m],a['pred_lon_deg_e'][m]))
        for b in MODELS[i+1:]:dist.append(haversine(a['pred_lat_deg_n'][m],a['pred_lon_deg_e'][m],a['pred_lat_deg_n'][b],a['pred_lon_deg_e'][b]))
    v=np.mean(vec,axis=0);norm=np.linalg.norm(v,axis=1);assert (norm>.1).all();v=v/norm[:,None];lat=np.rad2deg(np.arcsin(v[:,2]));lon=np.rad2deg(np.arctan2(v[:,1],v[:,0]))
    z=pd.DataFrame({'IMD':np.mean(dist,axis=0),'consensus_error':haversine(lat,lon,truth.jma_lat_deg_n,truth.jma_lon_deg_e),'mean_model_error':a.track_error_km.mean(axis=1)},index=a.index)
    for m in MODELS:z[m]=a['track_error_km'][m]
    # Triangle inequality: mean pair distance <= twice mean distance to truth.
    assert (z.IMD<=2*z.mean_model_error+.03).all()
    z=z.reset_index();z.to_csv(out/'data/disagreement_cases.csv',index=False)
    cors=[];validation=[]
    for lead,g in z.groupby('lead_hours'):
        ids=sorted(g.TC_ID.unique());n=len(ids);codes=pd.Categorical(g.TC_ID,categories=ids).codes;w=base_weights(g)
        idx=rng_for(f'cor{lead}').integers(n,size=(B,n));weights=np.zeros((B,n));np.add.at(weights,(np.repeat(np.arange(B),n),idx.ravel()),1);bw=weights[:,codes]*w
        for target in ['consensus_error','mean_model_error',*MODELS]:
            point=wcorr(g.IMD.to_numpy(),g[target].to_numpy(),w[None,:])[0];b=wcorr(g.IMD.to_numpy(),g[target].to_numpy(),bw);lo,hi=np.nanquantile(b,[.025,.975]);cors.append(dict(lead_hours=lead,target=target,rho=point,low=lo,high=hi,n_storms=n,n_cases=len(g)))
        for q in [.75,.8,.9]:
            for tc in ids:
                train=g[g.TC_ID!=tc];test=g[g.TC_ID==tc].copy();tw=base_weights(train);errq=wquantile(train.consensus_error,tw,q);imdq=wquantile(train.IMD,tw,q)
                test['event']=test.consensus_error>errq;test['alert']=test.IMD>imdq;test['risk_score']=test.IMD/imdq;test['error_threshold']=errq;test['imd_threshold']=imdq;test['quantile']=q;validation.append(test)
    cr=pd.DataFrame(cors);cr.to_csv(out/'data/IMD_correlations.csv',index=False);val=pd.concat(validation);val.to_csv(out/'data/IMD_LOCO_predictions.csv',index=False)
    metrics=[]
    for (lead,q),g in val.groupby(['lead_hours','quantile']):metrics.append(dict(lead_hours=lead,quantile=q,**score_metrics(g)))
    mt=pd.DataFrame(metrics);mt['lift']=mt.precision/mt.event_rate;mt.to_csv(out/'data/IMD_LOCO_metrics.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(10,7.3))
    for ax,lead in zip(axs.flat,KEY):
        g=z[z.lead_hours==lead];ax.scatter(g.IMD,g.consensus_error,s=18,alpha=.55,color='#27699d');row=cr[(cr.lead_hours==lead)&(cr.target=='consensus_error')].iloc[0]
        ax.set_title(f'{lead} h: weighted rho = {row.rho:.2f}',fontsize=10);ax.set_xlabel('IMD (km)');ax.set_ylabel('Consensus DPE (km)');ax.grid(alpha=.2)
    ax=axs.flat[-1];t=mt[mt['quantile']==.75];ax.plot(t.lead_hours,t.POD,'o-',label='POD');ax.plot(t.lead_hours,t.FAR,'s-',label='FAR');ax.plot(t.lead_hours,t.AUC,'^-',label='AUC');ax.set(ylim=(0,1.05),xlabel='Lead time (h)',title='Held-out cyclone evaluation');ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=3,fontsize=8,frameon=False)
    fig.tight_layout();save(fig,out,'Figure_5_disagreement');return z,cr,mt

def robustness(d,out,cl,truth):
    allrows=[];bands=[];weightrows=[];geomrows=[]
    for setting in ['primary','all_available']:
        g=d[d.case_type=='FULL_120H'] if setting=='primary' else d
        s=g.groupby(['TC_ID','lead_hours','model']).track_error_km.mean().unstack('model');ids=sorted(g.TC_ID.unique())
        x=np.stack([(s[a]-s[b]).unstack('lead_hours').reindex(index=ids,columns=LEADS).to_numpy() for a,b in [('graphcast','pangu'),('graphcast','fourcastnet'),('pangu','fourcastnet')]],axis=2)
        for k,tc in enumerate(ids):
            v=np.delete(x,k,axis=0);mu=np.nanmean(v,axis=0)
            for j,t in enumerate(LEADS):
                for q,pair in enumerate(['GC-Pangu','GC-FCN','Pangu-FCN']):allrows.append(dict(setting=setting,omitted=tc,lead_hours=t,pair=pair,mean=mu[j,q]))
            if setting=='primary':
                n=len(v);se=v.std(0,ddof=1)/np.sqrt(n);idx=rng_for(f'loco_band{tc}').integers(n,size=(B,n));boot=v[idx].mean(1);crit=np.quantile(np.max(abs((boot-mu)/se),axis=(1,2)),.95)
                for j,t in enumerate(LEADS):bands.append(dict(omitted=tc,lead_hours=t,mean=mu[j,0],low=mu[j,0]-crit*se[j,0],high=mu[j,0]+crit*se[j,0]))
        # case-weighted alternative, cluster ratio bootstrap
        counts=g[g.model=='graphcast'].groupby(['TC_ID','lead_hours']).size().unstack().reindex(index=ids,columns=LEADS).fillna(0).to_numpy()
        rng=rng_for(f'weighted{setting}');idx=rng.integers(len(ids),size=(B,len(ids)));wt=np.zeros((B,len(ids)));np.add.at(wt,(np.repeat(np.arange(B),len(ids)),idx.ravel()),1)
        num=np.nan_to_num(x[:,:,0])*counts;boot=(wt@num)/(wt@counts);mu=num.sum(0)/counts.sum(0);lo,hi=np.quantile(boot,[.025,.975],axis=0)
        for j,t in enumerate(LEADS):weightrows.append(dict(setting=setting,lead_hours=t,mean=mu[j],low=lo[j],high=hi[j]))
    ld=pd.DataFrame(allrows);bd=pd.DataFrame(bands);wr=pd.DataFrame(weightrows)
    ld.to_csv(out/'data/LOCO_all_pairs.csv',index=False);bd.to_csv(out/'data/LOCO_simultaneous_GP.csv',index=False);wr.to_csv(out/'data/case_weighted_GP.csv',index=False)
    for m,g in d[(d.case_type=='FULL_120H')&(d.lead_hours==120)].groupby('model'):
        for kind in ['back12','center12']:
            good=g[g['ATE_'+kind].notna()&g['CTE_'+kind].notna()];s=good.groupby('TC_ID')[['ATE_'+kind,'CTE_'+kind]].agg(lambda v:(v**2).mean());a=s.iloc[:,0].mean();c=s.iloc[:,1].mean();geomrows.append(dict(model=m,direction=kind,cross_squared_fraction=c/(a+c),n_cases=len(good)))
    pd.DataFrame(geomrows).to_csv(out/'data/direction_sensitivity.csv',index=False)
    sensitivity=[]
    for th in [.5,1.,2.]:
        cc=classify(truth,th);cc.to_csv(out/f'data/path_types_threshold_{th}.csv',index=False)
        full=d[(d.case_type=='FULL_120H')&(d.lead_hours==120)].merge(cc[['TC_ID','recurving']],on='TC_ID')
        for (kind,m),g in full.groupby(['recurving','model']):s=g.groupby('TC_ID').track_error_km.mean();sensitivity.append(dict(threshold=th,recurving=kind,model=m,n_storms=len(s),n_cases=len(g),mean=s.mean()))
    pd.DataFrame(sensitivity).to_csv(out/'data/recurvature_sensitivity.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4.6))
    for tc,g in ld[(ld.setting=='primary')&(ld.pair=='GC-Pangu')].groupby('omitted'):axs[0].plot(g.lead_hours,g['mean'],color='gray',alpha=.5,lw=.8)
    original=pd.read_csv(out/'data/paired_by_lead.csv');og=original[(original.analysis_set=='primary_full120')&(original.pair=='graphcast-pangu')]
    axs[0].plot(og.lead_hours,og['mean'],label='Cyclone-weighted mean',color='#27699d');w=wr[wr.setting=='primary'];axs[0].plot(w.lead_hours,w['mean'],'--',label='Case-weighted mean',color='#D55E00');axs[0].axhline(0,color='black',lw=.8);axs[0].set(xlabel='Lead time (h)',ylabel='GC - Pangu DPE (km)',title='Gray: one cyclone removed');axs[0].grid(alpha=.2)
    b=bd[bd.lead_hours==120].sort_values('omitted');x=np.arange(len(b));axs[1].errorbar(x,b['mean'],yerr=[b['mean']-b.low,b.high-b['mean']],fmt='o',color='#27699d',capsize=2);axs[1].axhline(0,color='black',lw=.8);axs[1].set_xticks(x,b.omitted.astype(str),rotation=65,fontsize=8);axs[1].set(xlabel='Omitted cyclone',ylabel='GC - Pangu DPE (km)',title='120 h: recomputed simultaneous bands');axs[1].grid(alpha=.2);outside_legend(fig,axs[0]);save(fig,out,'Figure_6_robustness')
    return ld,bd,wr

def outside_legend(fig,ax):
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,fontsize=9);fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.85,hspace=.38,wspace=.28)

def save(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/f'figures/{name}.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();out=args.output_dir
    for f in ['data','figures']:(out/f).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    raw,audit=load_data(args.project_root);truthpath=args.project_root/'outputs/2025_jma_cases_v2/data/jma_2025_track_points.csv';truth=pd.read_csv(truthpath,dtype={'TC_ID':str})
    d,truth=geometry(raw,truth);d.to_csv(out/'data/error_components_cases.csv',index=False)
    r=components(d,out);full,cl=stratified(d,truth,out);z,cr,mt=disagreement(d,out);ld,bd,wr=robustness(d,out,cl,truth)
    fullids=sorted(full.TC_ID.unique());types=cl[cl.TC_ID.isin(fullids)];case=full[full.model=='graphcast'].groupby('case_id').first()
    init=pd.to_datetime(case.init_utc.str.replace(' UTC',''),utc=True);turn=pd.to_datetime(case.turn_time,utc=True)
    inside=int(((turn>=init)&(turn<=init+pd.Timedelta(hours=120))).sum())
    summary={'B':B,'seed':SEED,'direction_missing_rows':int(d.ATE.isna().sum()),'direction_missing_primary_rows':int(d[d.case_type=='FULL_120H'].ATE.isna().sum()),'primary_type_counts':types.recurving.value_counts().to_dict(),'primary_case_type_counts':case.recurving.value_counts().to_dict(),'cases_containing_turn':inside,'recurving_ids':types[types.recurving].TC_ID.tolist(),'LOCO_120_positive_sim_band_count':int((bd[bd.lead_hours==120].low>0).sum()),'LOCO_120_sim_low_range':[float(bd[bd.lead_hours==120].low.min()),float(bd[bd.lead_hours==120].low.max())],'large_DPE_gt1000_rows':int((d.track_error_km>1000).sum()),'primary_120_components':r[(r.setting=='primary')&(r.lead_hours==120)].to_dict('records'),'IMD_consensus_correlations':cr[cr.target=='consensus_error'].to_dict('records'),'IMD_validation75':mt[mt['quantile']==.75].to_dict('records')}
    audit['truth_sha256']=hashlib.sha256(truthpath.read_bytes()).hexdigest();audit['new_analysis']=summary;(out/'data/analyses3to6_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str));(out/'data/analyses3to6_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str));print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
