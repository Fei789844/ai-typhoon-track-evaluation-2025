"""Geometric illustration only: identical error, two orthogonal bases.
No experimental data are represented. Example DPE=100 km; basis rotation=120 deg.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'font.size': 9,
                     'svg.fonttype':'none', 'pdf.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(7.2,4.3))
fig.subplots_adjust(left=.035,right=.975,top=.84,bottom=.23,wspace=.10)
green, purple = '#00836b','#7160bd'
actual=np.array([0.,0.]); current=np.array([np.cos(np.pi/6),np.sin(np.pi/6)])
pred=-current; turn=np.array([-1.30,-.75])
old=np.array([np.cos(5*np.pi/6),np.sin(5*np.pi/6)])
def arrow(ax,a,b,color,lw=1.2,style='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=lw,linestyle=style,shrinkA=0,shrinkB=0))
for k, ax in enumerate(axes):
    ax.set_aspect('equal'); ax.set_xlim(-1.95,1.05);ax.set_ylim(-1.5,1.2);ax.axis('off')
    ax.plot([turn[0],-.95,0,.72],[turn[1],-.7,0,.42],color='#c4c9cc',lw=1.5,zorder=0)
    ax.scatter(*turn,s=16,color='#8a8f93');arrow(ax,turn,turn+.55*old,'#8a8f93')
    ax.text(-1.90,-1.04,'转向时刻方向',fontsize=8,color='#555555')
    arrow(ax,actual,pred,'#30383b',2)
    basis=current if k==0 else old
    right=np.array([basis[1],-basis[0]])
    arrow(ax,-.95*basis,.95*basis,green,.9)
    arrow(ax,-.9*right,.9*right,purple,.9)
    along=np.dot(pred,basis);cross=np.dot(pred,right);foot=along*basis
    assert np.allclose(foot+cross*right,pred)
    assert np.isclose(along**2+cross**2,1)
    if k==0:
        arrow(ax,actual,pred,green,3)
        ax.text(.36,.65,'当前方向 / 沿轨正向',fontsize=8,ha='center',color=green)
        ax.text(.45,-.98,'横轨正向（右）',fontsize=8,ha='center',color=purple)
    else:
        arrow(ax,actual,foot,green,3);arrow(ax,foot,pred,purple,3)
        ax.plot([pred[0],np.dot(pred,right)*right[0]],[pred[1],np.dot(pred,right)*right[1]],ls=':',color='#9aa1a4',lw=.8)
        # Right-angle mark wholly inside the projection triangle.
        u=-basis*.09;v=-right*.09
        q=np.array([foot+u,foot+u+v,foot+v]);ax.plot(q[:,0],q[:,1],color='#555555',lw=.8)
        ax.text(-.85,.78,'沿用转向时刻方向',color=green,fontsize=8,ha='center')
        ax.text(.48,.65,'横轨正向（右）',color=purple,fontsize=8,ha='center')
    ax.scatter(*actual,s=28,color='#17292f',zorder=5)
    ax.scatter(*pred,s=30,facecolor='white',edgecolor='#17292f',zorder=5)
    ax.text(.18,.10,'当前JMA中心',fontsize=8)
    ax.text(pred[0]-.06,pred[1]-.23,'预测中心',ha='center',fontsize=8)
    ax.set_title(['a  随当前运动方向定轴','b  固定转向时刻的参考方向'][k],fontsize=10,pad=18)
    ax.text(.5,-.08,'DPE = 100 km',transform=ax.transAxes,ha='center',fontsize=9)
    ax.text(.5,-.19,['ATE = −100 km；CTE = 0','ATE = +50 km；CTE ≈ −86.6 km'][k],transform=ax.transAxes,ha='center',fontsize=9)
fig.text(.5,.95,'同一位置误差，改变参考方向后，沿轨与横轨分量随之改变',ha='center',fontsize=11)
fig.text(.5,.035,'几何示例：两种参考方向相差120°，并非实测结果；DPE² = ATE² + CTE²。',ha='center',fontsize=8)
out=ROOT/'figures'/'Archived_basis_schematic'
for ext in ['svg','pdf','png']: fig.savefig(out.with_suffix('.'+ext),dpi=240)
import re
svg=out.with_suffix('.svg')
svg.write_text(re.sub(r'<!DOCTYPE[^>]*>', '', svg.read_text()))
print(out)
