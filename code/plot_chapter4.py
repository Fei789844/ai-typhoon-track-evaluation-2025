"""All Chapter 4 figures from saved tables; no synthetic data, no image generator."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tables';F=ROOT/'figures';A=ROOT/'audit'
MODELS=['graphcast','pangu','fourcastnet'];NAMES=['GraphCast','Pangu-Weather','FourCastNet']
COLORS=['#0072B2','#D55E00','#009E73'];STYLES=['-','--','-.'];MARKERS=['o','s','^']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,
                     'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
                     'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
FIGURE_RECORDS=[]

def read(name): return pd.read_csv(T/f'{name}.csv',dtype={'TC_ID':str})
def label(ax,letter,title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.13,1.055,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
def finish(fig,name,sources,caption):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=A/f'{name}.alignment.json',
                                       tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    # Fixed physical dimensions retained; margins allocated before export.
    fig.savefig(F/f'{name}.pdf')
    fig.savefig(F/f'{name}.svg')
    fig.savefig(F/f'{name}.png',dpi=300)
    fig.savefig(F/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    FIGURE_RECORDS.append(dict(figure=name,sources=sources,caption=caption,
                              size_inches=list(fig.get_size_inches()),backend='matplotlib',image_generation=False))
    plt.close(fig)

def overall():
    d=read('overall_by_lead')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),sharex=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.88,hspace=.37,wspace=.28)
    for ax,k,title,l in zip(axs.flat,['absATE','absCTE','ATE','CTE'],['Mean absolute along-track error','Mean absolute cross-track error','Signed along-track bias (+ ahead)','Signed cross-track bias (+ right)'],'abcd'):
        for m,n,c,ls in zip(MODELS,NAMES,COLORS,STYLES):
            g=d[(d.model==m)&(d.metric==k)].sort_values('lead_hours')
            ax.plot(g.lead_hours,g['mean'],color=c,ls=ls,lw=1.3,label=n)
            ax.fill_between(g.lead_hours,g.low,g.high,color=c,alpha=.12,linewidth=0)
        ax.set_ylabel('Error (km)');ax.set_xlim(6,120);ax.set_xticks([6,24,48,72,96,120]);ax.grid(alpha=.16)
        if k in ['ATE','CTE']:ax.axhline(0,color='.25',lw=.6)
        else:ax.set_ylim(bottom=0)
        label(ax,l,title)
    for ax in axs[1]:ax.set_xlabel('Forecast lead time (h)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.99))
    finish(fig,'Figure_4_1_direction',['overall_by_lead.csv'],'14 storms; 73–74 valid cases per lead; storm-equal mean and pointwise 95% cluster bootstrap CI (20,000 draws).')

def support():
    catalog=read('storm_catalog');turns=catalog[catalog.recurving].sort_values('TC_ID');s=read('event_support')
    truth=pd.read_csv(ROOT/'inputs/jma_track_points.csv',dtype={'TC_ID':str});truth['ts']=pd.to_datetime(truth.time_utc.str.replace(' UTC',''),utc=True)
    fig,ax=plt.subplots(figsize=(3.6,3.3));fig.subplots_adjust(left=.16,right=.97,bottom=.25,top=.84)
    palette=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6D1D']
    for row,c in zip(turns.itertuples(),palette):
        g=truth[(truth.TC_ID==row.TC_ID)&truth.jma_grade_code.isin([2,3,4,5])].sort_values('ts')
        t=pd.to_datetime(row.turn_time,utc=True);near=g[g.ts.between(t-pd.Timedelta(hours=24),t+pd.Timedelta(hours=24))]
        ax.plot(g.lon_deg_e,g.lat_deg_n,c=c,alpha=.40,lw=.8)
        ax.plot(near.lon_deg_e,near.lat_deg_n,c=c,lw=1.7,label=row.name)
        at=g[g.ts==t];ax.scatter(at.lon_deg_e,at.lat_deg_n,c=c,s=20,marker='o',edgecolor='white',lw=.4,zorder=3)
    ax.set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)');ax.grid(alpha=.15)
    label(ax,'a','JMA tracks; dots mark diagnosed turns')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    finish(fig,'Figure_4_2a_turn_tracks',['turn_classification.csv','inputs/jma_track_points.csv'],'Full tropical-stage paths faint; +/-24 h paths emphasized; dots mark diagnosed turns. Longitude/latitude display is not a distance projection.')

    fig,ax=plt.subplots(figsize=(3.6,2.4));fig.subplots_adjust(left=.24,right=.97,bottom=.24,top=.80)
    arr=np.zeros((5,20))
    for i,tc in enumerate(turns.TC_ID):
        for h in s.loc[s.TC_ID==tc,'lead_hours']:arr[i,int(h/6)-1]=1
    ax.imshow(arr,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#EEEEEE','#0072B2']),vmin=0,vmax=1)
    ax.set_yticks(range(5),turns.name.to_list(),fontsize=7)
    ax.set_xticks([0,3,7,11,15,19],[6,24,48,72,96,120]);ax.set_xlabel('Forecast lead time (h)')
    label(ax,'b','Blue: pre/post data at the same lead')
    finish(fig,'Figure_4_2b_event_support',['turn_classification.csv','event_support.csv'],'Blue cells show storm-lead combinations with matched pre-turn and post-turn records; at most 2 storms per lead, with no common lead across all 5 storms.')

def event():
    phase=read('event_three_phase_means');st=read('event_storm_effects');inf=read('event_primary_inference')
    ids=st[['TC_ID','name']].drop_duplicates().sort_values('TC_ID');metrics=['DPE','absATE','absCTE']
    titles=['Total distance error','Absolute along-track error','Absolute cross-track error']
    fig,axs=plt.subplots(2,3,figsize=(7.2,5.5));fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.88,hspace=.50,wspace=.35)
    for j,(k,title) in enumerate(zip(metrics,titles)):
        ax=axs[0,j]
        for i,(m,n,c,ls,mk) in enumerate(zip(MODELS,NAMES,COLORS,STYLES,MARKERS)):
            v=phase[(phase.model==m)&(phase.metric==k)].set_index('phase').loc[['pre','near','post']]
            xx=np.arange(3)+(i-1)*.07
            ax.errorbar(xx,v['mean'],yerr=np.array([v['mean']-v.low,v.high-v['mean']]),color=c,marker=mk,ls=ls,lw=1,capsize=2,ms=3,label=n)
        ax.set_xticks(range(3),['Before','Near','After']);ax.set_xlim(-.35,2.35);ax.set_ylim(bottom=0);ax.set_ylabel('Mean error (km)');ax.grid(alpha=.13)
        label(ax,'abc'[j],title)
        ax=axs[1,j];ax.axhline(0,color='.3',lw=.7)
        for i,(m,n,c,mk) in enumerate(zip(MODELS,NAMES,COLORS,MARKERS)):
            v=st[(st.model==m)&(st.metric==k)].set_index('TC_ID').loc[ids.TC_ID]
            ax.scatter(np.arange(5)+(i-1)*.19,v.delta,color=c,marker=mk,s=18,label=n,zorder=3)
            avg=inf[(inf.model==m)&(inf.metric==k)].iloc[0]
            ax.plot([-.4,4.4],[avg['mean'],avg['mean']],c=c,ls=':',lw=.65,alpha=.7)
        ax.set_xticks(range(5),ids.TC_ID.to_list(),fontsize=7)
        ax.set_xlim(-.5,4.5);ax.set_ylabel('After - before (km)');ax.grid(axis='y',alpha=.13)
        label(ax,'def'[j],'Individual storm changes')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.99))
    finish(fig,'Figure_4_3a_event',['event_three_phase_means.csv','event_storm_effects.csv','event_primary_inference.csv'],'Top: common storm/lead support across all 3 phases; storm-equal means with empirical percentile intervals (all 3,125 draws). Bottom: individual matched changes; IDs 2509/2519/2522/2523/2526 denote KROSA/NEOGURI/HALONG/NAKRI/FUNG-WONG. Dotted lines are 5-storm means, not confidence bounds. 5 independent sampling units, not 24 independent lead cells.')

def groups():
    d=read('group_means');fig,axs=plt.subplots(2,3,figsize=(7.2,5.0),sharex=True,sharey='col')
    fig.subplots_adjust(left=.08,right=.98,bottom=.11,top=.88,hspace=.35,wspace=.33)
    for i,kind in enumerate([True,False]):
        for j,(k,title) in enumerate(zip(['DPE','absATE','absCTE'],['DPE','Absolute ATE','Absolute CTE'])):
            ax=axs[i,j]
            for m,n,c,ls in zip(MODELS,NAMES,COLORS,STYLES):
                g=d[(d.recurving==kind)&(d.model==m)&(d.metric==k)].sort_values('lead_hours')
                ax.plot(g.lead_hours,g['mean'],c=c,ls=ls,lw=1.2,label=n);ax.fill_between(g.lead_hours,g.low,g.high,color=c,alpha=.12,linewidth=0)
            label(ax,'abcdef'[i*3+j],title+(' | Recurving (n=5)' if kind else ' | Other (n=9)'))
            ax.set_xlim(6,120);ax.set_ylim(bottom=0);ax.set_xticks([24,48,72,96,120]);ax.grid(alpha=.15)
            if i==1:ax.set_xlabel('Forecast lead time (h)')
            ax.set_ylabel('Mean error (km)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.99))
    finish(fig,'Figure_4_4a_groups',['group_means.csv'],'5 recurving vs 9 other storms; same direction-valid cases across DPE and components; pointwise 95% bootstrap intervals, 20,000 stratified draws. Type classification does not mean all cases include a turn.')

def interactions():
    d=read('group_interactions');pairs=['graphcast-pangu','graphcast-fourcastnet','pangu-fourcastnet']
    titles=['GraphCast - Pangu-Weather','GraphCast - FourCastNet','Pangu-Weather - FourCastNet']
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.85),sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.77,wspace=.28)
    for ax,p,t,l in zip(axs,pairs,titles,'abc'):
        g=d[(d.pair==p)&(d.metric=='DPE')].sort_values('lead_hours')
        ax.fill_between(g.lead_hours,g.low,g.high,color='#0072B2',alpha=.14,lw=0,label='Pointwise 95% CI')
        ax.plot(g.lead_hours,g['mean'],c='#0072B2',lw=1.3,label='Mean interaction')
        ax.plot(g.lead_hours,g.sim_low,c='#D55E00',ls='--',lw=.8,label='Simultaneous 95% band')
        ax.plot(g.lead_hours,g.sim_high,c='#D55E00',ls='--',lw=.8)
        ax.axhline(0,c='.25',lw=.7);ax.set_xlim(6,120);ax.set_xticks([24,72,120]);ax.set_xlabel('Forecast lead time (h)');ax.grid(alpha=.13);label(ax,l,t)
    axs[0].set_ylabel('Pair difference: recurving - other (km)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.0))
    finish(fig,'Figure_4_5_interactions',['group_interactions.csv'],'Difference of paired-model differences between groups. Bands adjusted across 3 pairs x 3 metrics x 20 leads = 180 coordinates; figure displays the DPE subset.')

def main():
    F.mkdir(exist_ok=True);overall();support();event();groups();interactions()
    (A/'figure_manifest.json').write_text(json.dumps(FIGURE_RECORDS,ensure_ascii=False,indent=2))
    print('Exported',len(FIGURE_RECORDS),'figures as PNG/PDF/SVG.')

if __name__=='__main__':main()
