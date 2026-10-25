import json,pathlib,time,urllib.request,urllib.parse
out=pathlib.Path(__file__).parent
dois={21:'10.5194/gmd-11-3999-2018',22:'10.1029/2020MS002109',24:'10.1038/s43247-025-02502-y',25:'10.1038/s41586-024-07744-y',26:'10.1038/s41586-025-09005-y',27:'10.5194/gmd-19-4703-2026',28:'10.1175/BAMS-D-23-0162.1',29:'10.1146/annurev.fluid.37.061903.175702',31:'10.1175/1520-0434(2000)015<0662:DTCTFE>2.0.CO;2',32:'10.1175/MWR-D-12-00071.1',33:'10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2',34:'10.1175/2009BAMS2755.1',35:'10.1016/j.jcp.2007.02.014',36:'10.1126/science.1115255',37:'10.1175/MWR-D-12-00111.1',38:'10.1214/aos/1176344552',39:'10.1175/BAMS-D-15-00267.1',40:'10.1016/j.patrec.2005.10.010'}
for n,doi in dois.items():
    dest=out/f'verified_{n}.json'
    try:
        if dest.exists():x=json.loads(dest.read_text())
        else:
            req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),headers={'User-Agent':'ThesisReferenceCheck/1.0'})
            with urllib.request.urlopen(req,timeout=25) as f:x=json.load(f)['message']
            dest.write_text(json.dumps(x,ensure_ascii=False,indent=2))
        print(n,json.dumps({k:x.get(k) for k in ['title','author','published','container-title','volume','issue','page','article-number']},ensure_ascii=False),flush=True)
    except Exception as e:print(n,str(e),flush=True)
    time.sleep(1)
