"""User-reference geometry; illustrative absolute component lengths, not observations."""
from pathlib import Path
import re,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
root=Path(__file__).resolve().parents[1]
font=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'font.family':font.get_name(),'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7.2,5.5));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,900),ylim=(680,0));ax.set_aspect('equal');ax.axis('off')
def txt(x,y,t,size=9,color='#444444',weight='normal'):ax.text(x,y,t,fontsize=size,color=color,va='center',fontweight=weight)
def arr(a,b,color,lw=1,style='-'):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=lw,linestyle=style,shrinkA=0,shrinkB=0))
green,purple='#009c7a','#8c7cdd'
o=np.array([165.,155.]);a=np.array([290.,85.]);p=np.array([222.,105.08]);u=(a-o)/np.linalg.norm(a-o)
theta=np.arccos(-.52);f=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])@u
e=p-a
for k,dy in enumerate([15,325]):
 basis=u if k==0 else f
 q=a+np.dot(e,basis)*basis
 assert abs(np.dot(q-a,p-q))<1e-8
 ate=np.linalg.norm(q-a)/np.linalg.norm(e);cte=np.linalg.norm(p-q)/np.linalg.norm(e)
 assert np.isclose(ate**2+cte**2,1)
 print(k,round(100*ate),round(100*cte))
 s=np.array([0.,dy]);O=o+s;A=a+s;P=p+s;Q=q+s
 ax.add_patch(FancyBboxPatch((28,dy+8),842,292,boxstyle='round,pad=0,rounding_size=12',facecolor='white',edgecolor='#d6d6d6',linestyle=(0,(4,4)),linewidth=.6))
 ax.plot([O[0],380],[O[1],dy+35],color='#555555',lw=1)
 ax.plot([O[0],395],[O[1],dy+287],color='#555555',lw=1)
 arr(O,O+65*f,'#90908a');ax.scatter(*O,s=18,color='#666666',zorder=4);txt(40,dy+105,'转向时刻运动方向')
 if k==0:
  arr(A-20*u,A+78*u,green);txt(365,dy+65,'沿轨轴')
 else:
  arr(Q,Q+39*f,'#555555',1,'--');txt(239,dy+37,'平行移来')
 ax.plot([A[0],P[0]],[A[1],P[1]],color='#333333',lw=1)
 ax.plot([A[0],Q[0]],[A[1],Q[1]],color=green,lw=3)
 ax.plot([Q[0],P[0]],[Q[1],P[1]],color=purple,lw=3)
 ax.scatter(*A,s=35,color='#111111',zorder=5);ax.scatter(*P,s=35,facecolor='white',edgecolor='#111111',linewidth=1.1,zorder=5)
 txt(A[0]+18,A[1]+26,'实际中心');txt(P[0],P[1]+29,'预测中心')
 txt(500,dy+32,['主分析：每个时刻按当时运动方向定轴','对照：轴固定为转向时刻方向'][k],10,'#111111','bold')
 labels=['误差长度 DPE：100%',f'沿轨 ATE：约{100*ate:.0f}%',f'横轨 CTE：约{100*cte:.0f}%']
 for j,(label,fill,border) in enumerate(zip(labels,['#f1eee8','#e1f5ed','#eeebfc'],['#9c9991','#60aa96','#9889e8'])):
  y=dy+62+j*49;ax.add_patch(FancyBboxPatch((500,y),313,38,boxstyle='round,pad=0,rounding_size=5',facecolor=fill,edgecolor=border,lw=.6))
  txt(518,y+19,label,10,['#444444','#005c4b','#473d93'][j],'bold')
 txt(500,dy+237,['轴随台风转向而转动','同一误差，拆出的分量不同'][k])
txt(40,650,'示意图：两种参考方向相差约120°；百分比表示分量长度与DPE之比，非实测结果。',8)
out=root/'figures/Figure_4_3c_basis_schematic'
for ext in ['png','pdf','svg']:fig.savefig(out.with_suffix('.'+ext),dpi=240)
svg=out.with_suffix('.svg');svg.write_text(re.sub(r'<!DOCTYPE[^>]*>','',svg.read_text()))
