import concurrent.futures, json, pathlib, urllib.request, urllib.parse

QUERIES = {
21:'Challenges and design choices for global weather and climate models based on machine learning',
22:'Improving Data Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere',
23:'Forecasting Global Weather with Graph Neural Networks Keisler',
24:'FengWu Pushing the Skillful Global Medium range Weather Forecast beyond 10 Days Lead',
25:'Neural general circulation models for weather and climate',
26:'A foundation model for the Earth system',
27:'AIFS ECMWF data driven forecasting system Lang',
28:'The rise of data driven weather forecasting',
29:'The physics of tropical cyclone motion Chan',
30:'Recent developments in tropical cyclone ensemble forecasting Yamaguchi',
31:'Dynamical Tropical Cyclone Track Forecast Errors Part I Tropical Region Error Sources Carr Elsberry',
32:'Diagnosing Forecast Errors in Tropical Cyclone Motion Galarneau Davis',
33:'A General Method for Tracking Analysis and Its Application to Meteorological Data Hodges',
34:'The International Best Track Archive for Climate Stewardship Unifying Tropical Cyclone Data Knapp',
35:'Ensemble forecasting Leutbecher Palmer',
36:'Weather Forecasting with Ensemble Methods Gneiting Raftery',
37:'Assessing the Ensemble Spread Error Relationship Hopson',
38:'Bootstrap Methods Another Look at the Jackknife Efron',
39:'The Stippling Shows Statistically Significant Grid Points Wilks',
40:'An introduction to ROC analysis Fawcett',
}
OUT=pathlib.Path(__file__).parent
def fetch(item):
    n,q=item
    url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':q,'rows':2})
    try:
        with urllib.request.urlopen(url,timeout=40) as f: data=json.load(f)
        (OUT/f'crossref_{n}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        return n,[{k:x.get(k) for k in ['title','DOI','author','published','container-title','volume','issue','page','abstract']} for x in data['message']['items']]
    except Exception as e:return n,str(e)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(fetch,QUERIES.items()))
    (OUT/'candidates.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    for n,items in sorted(results.items()):
        print(n,json.dumps([{k:x.get(k) for k in ['title','DOI','published','container-title','volume','page']} for x in items] if isinstance(items,list) else items,ensure_ascii=False))
