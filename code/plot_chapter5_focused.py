from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
m=pd.read_csv(root/'data/IMD_LOCO_metrics.csv').query('quantile == 0.75').sort_values('lead_hours')
assert len(m)==20
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
sys.path.insert(0, str(root / 'code'))
from audit_panel_alignment import require_matplotlib_panel_alignment
figures = root / 'figures'
audit = root / 'audit' / 'chapter5'
figures.mkdir(parents=True, exist_ok=True)
audit.mkdir(parents=True, exist_ok=True)
# Main: screening enrichment and coverage. Supplement: ranking and underlying event proportions.
for name,main in [('Figure_5_2_screening',True),('Figure_A_4_risk_auxiliary',False)]:
 fig,axes=plt.subplots(1,2,figsize=(7.1,2.8));fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.86,wspace=.30)
 x=m.lead_hours
 for ax in axes:
  ax.set(xlim=(6,120),xticks=[6,24,48,72,96,120],xlabel='Forecast lead time (h)');ax.grid(axis='y',alpha=.15)
 if main:
  axes[0].plot(x,m.lift,'o-',ms=3,color='#0072B2');axes[0].axhline(1,color='.45',ls='--',lw=.8);axes[0].set(ylim=(0,3.3),ylabel='Precision / event rate',title='a  Screening gain')
  axes[1].plot(x,m.POD,'o-',ms=3,color='#0072B2',label='POD');axes[1].plot(x,m.FAR,'s--',ms=3,color='#D55E00',label='FAR');axes[1].set(ylim=(0,1),ylabel='Weighted proportion',title='b  Detection and false alerts');axes[1].legend(frameon=False,fontsize=8,loc='upper right',ncol=2)
 else:
  axes[0].plot(x,m.AUC,'o-',ms=3,color='#0072B2');axes[0].axhline(.5,color='.45',ls='--',lw=.8);axes[0].set(ylim=(0,1),ylabel='AUC',title='a  Ranking ability')
  axes[1].plot(x,m.precision,'o-',ms=3,color='#0072B2',label='Alert precision');axes[1].plot(x,m.event_rate,'s--',ms=3,color='#D55E00',label='Overall event rate');axes[1].set(ylim=(0,1),ylabel='Weighted proportion',title='b  Error concentration');axes[1].legend(frameon=False,fontsize=7,loc='upper right')
 fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(audit/(name+'_alignment.json')),strict=True)
 fig.savefig(figures/(name+'.png'),dpi=600)
 fig.savefig(figures/(name+'.pdf'))
 fig.savefig(figures/(name+'.svg'))
 plt.close(fig)
