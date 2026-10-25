from pathlib import Path
import json,re,urllib.parse

root=Path(__file__).resolve().parent
p=root.parent/'AI台风路径论文_第四章实验更新版.md'
s=p.read_text()
def before(marker,text):
    global s
    assert s.count(marker)==1,marker
    s=s.replace(marker,text+'\n\n'+marker)
def after_paragraph(prefix,text):
    global s
    paragraphs=s.split('\n\n');hits=[i for i,x in enumerate(paragraphs) if x.startswith(prefix)]
    assert len(hits)==1,(prefix,hits)
    paragraphs.insert(hits[0]+1,text);s='\n\n'.join(paragraphs)
def append_paragraph(prefix,text):
    global s
    paragraphs=s.split('\n\n');hits=[i for i,x in enumerate(paragraphs) if x.startswith(prefix)]
    assert len(hits)==1,(prefix,hits)
    paragraphs[hits[0]]+=text;s='\n\n'.join(paragraphs)

old='全球人工智能天气预报研究先建立统一数据和评价标准，再逐步发展到高分辨率确定性预报与概率预报。'
assert old in s
s=s.replace(old,'统一评价基准使不同技术路线能够直接比较。')
before('统一评价基准使不同技术路线能够直接比较。','全球人工智能天气预报经历了从简化试验到高分辨率预报的发展。Dueben和Bauer通过简化全球预报试验，讨论神经网络预报系统的数据需求和设计选择；Weyn等将全球场映射到立方球网格，用卷积神经网络学习气象场演变；Keisler进一步使用图神经网络，以6 h为步长推进全球大气状态。[[21]](#ref-21)[[22]](#ref-22)[[23]](#ref-23) 这些工作逐步建立了从历史气象资料学习时间演变、再递推生成预报的技术路线。')
append_paragraph('统一评价基准使不同技术路线能够直接比较。',' Ben Bouallègue等还在相同初始化条件下比较Pangu-Weather与ECMWF IFS，将全球指标和极端事件纳入接近业务条件的检验。[[28]](#ref-28)')
after_paragraph('在时间推进方面，Pangu-Weather','Chen等提出的FengWu以多模态、多任务学习组织不同气象变量，并通过回放缓冲机制减轻递推误差累积；其集合扩展使用条件扩散模型生成多个预报成员。[[24]](#ref-24) 这条路线将确定性预报的长时效推进与多情景预测联系起来。')
after_paragraph('Pathak等提出的FourCastNet','研究路线还拓展到物理过程与机器学习结合、基础模型预训练以及业务部署。Kochkov等的NeuralGCM将可微分大气动力求解器与机器学习组件结合，开展天气与气候预测；Bodnar等的Aurora在多来源地球物理资料上预训练，再适配天气、空气质量、海浪和热带气旋路径等任务。[[25]](#ref-25)[[26]](#ref-26) ECMWF的AIFS已进入业务运行，Moldovan等介绍的AIFS Single 1.1.0进一步加入降水非负性等约束，并扩展训练资料和输出变量。[[27]](#ref-27)')
before('人工智能天气预报模型的热带气旋研究已由','热带气旋路径研究同时关注位置准确度与不确定性的表达。Heming等综述了业务数值模式、共识预报和集合预报在路径预测中的进展，说明平均路径评价与预报置信度评价需要配合开展。[[30]](#ref-30) 在验证资料方面，Knapp等建立的IBTrACS汇集不同业务机构的最佳路径记录，为跨海盆研究提供统一的数据入口。[[34]](#ref-34)')
after_paragraph('热带气旋路径检验还形成了','路径形态分析还需要联系台风运动过程。Chan总结了环境引导气流、β效应、垂直风切变与加热对气旋运动的作用；Carr和Elsberry分析了中纬度环流作用被模式错误描述时产生的路径偏差。[[29]](#ref-29)[[31]](#ref-31) Galarneau和Davis则通过环境风场及引导层的诊断，追查运动预报误差的来源。[[32]](#ref-32) 这些研究为从总体距离评价进一步转向运动方向和转向阶段分析提供了依据。')
before('多模型共识通过综合不同模型的预测位置','集合预报用多条可能的演变轨迹表达未来天气的不确定性。Gneiting和Raftery介绍了集合方法相对单一预报的信息价值；Leutbecher和Palmer进一步说明，初始状态和模式误差的增长依赖具体天气形势，因此集合系统需要评价其概率信息和实际用途。[[36]](#ref-36)[[35]](#ref-35)')
append_paragraph('模型分歧与实际误差并非一一对应。',' Hopson进一步指出，离散度与误差的相关关系受指标定义和系统误差变化特征影响，评价时应同时关注离散度所含的信息及其与误差的对应关系。[[37]](#ref-37)')
before('三模型预报场采用相同追踪规则。','从预报场获得路径，需要先识别各时刻的中心，再将中心连接成连续轨迹。Hodges提出了气象系统特征识别与轨迹连接的一般方法，Heming进一步讨论了热带气旋追踪及验证流程。[[33]](#ref-33)[[2]](#ref-2) 本文按这一处理思路组织中心提取，具体搜索和评分参数如下。')
append_paragraph('同一台风的多个起报案例共享部分天气过程，',' 通过经验样本重采样估计统计量抽样波动的基本思想源于Efron提出的bootstrap方法。[[38]](#ref-38)')
append_paragraph('三组模型配对分别在6、12、…、120 h进行比较，',' Wilks对气象研究中的多重检验问题作了系统讨论，为将单项判断与整体判断区分提供了方法背景。[[39]](#ref-39)')
append_paragraph('AUC越高，表示大误差案例',' AUC作为排序判别指标的解释及ROC分析方法见Fawcett。[[40]](#ref-40)')
append_paragraph('**GraphCast和Pangu-Weather相对FourCastNet的中长时效优势',' 本节按统一的整体统计标准归纳三组配对结果，对应气象验证中对多重比较的处理要求。[[39]](#ref-39)')
append_paragraph('4.2节根据整段路径形态识别了转向台风。',' 既有研究将转向附近的路径偏差与中纬度环流及环境引导气流的预测联系起来。[[31]](#ref-31)[[32]](#ref-32) 本节据此关注转向阶段的误差变化，以位置及其方向分量刻画变化特征。')
append_paragraph('**60–120 h应转向多来源综合判断。',' 这种使用方式也符合集合预报研究对离散度与误差关系的认识：离散度提供风险信息，其对应的误差水平需要通过验证建立。[[35]](#ref-35)[[37]](#ref-37)')

records={}
for n in range(21,41):
    if n==23:
        x={'title':['Forecasting Global Weather with Graph Neural Networks'],'author':[{'family':'Keisler','given':'Ryan'}],'published':{'date-parts':[[2022]]},'DOI':'10.48550/arXiv.2202.07575','container-title':['arXiv预印本'],'type':'posted-content'}
    else:x=json.loads((root/f'verified_{n}.json').read_text())
    records[n]=x
def authors(x):
    names=[]
    for a in x['author'][:3]:
        initials=' '.join(w[0] for w in re.findall(r'[A-Za-zÀ-ž]+',a.get('given','')))
        names.append(a['family']+' '+initials+(' '+a['suffix'] if a.get('suffix') else ''))
    return ', '.join(names)+(', et al' if len(x['author'])>3 else '')
refs=[];bib=[]
for n,x in records.items():
    title=x['title'][0];title='The physics of tropical cyclone motion' if n==29 else title
    year=x['published']['date-parts'][0][0];journal=x['container-title'][0]
    loc=str(year)
    if x.get('volume'):loc+=', '+x['volume']
    if x.get('issue'):loc+='('+x['issue']+')'
    page=x.get('page') or x.get('article-number')
    if page:loc+=': '+page.replace('-','–')
    doi=x['DOI'];url='https://doi.org/'+urllib.parse.quote(doi,safe='/')
    refs.append(f'<a id="ref-{n}"></a>\n\n[{n}] {authors(x)}. {title}[{"EB/OL" if n==23 else "J"}]. {journal}, {loc}. [DOI: {doi}]({url}).')
    fullauthors=' and '.join(a['family']+', '+a.get('given','') for a in x['author'])
    bib.append('@'+('misc' if n==23 else 'article')+'{ref'+str(n)+',\n'+f'  title = {{{title}}},\n  author = {{{fullauthors}}},\n  year = {{{year}}},\n  journal = {{{journal}}},\n  doi = {{{doi}}},\n  url = {{{url}}}'+''.join(',\n  '+k+' = {'+x[v]+'}' for k,v in [('volume','volume'),('number','issue'),('pages','page')] if x.get(v))+'\n}')
before('<a id="acknowledgements"></a>','\n\n'.join(refs))
p.write_text(s)
(root/'新增文献_21-40.bib').write_text('\n\n'.join(bib)+'\n')
(root/'verified_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('Integrated',len(records),'new references')
