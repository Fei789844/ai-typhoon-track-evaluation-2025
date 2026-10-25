"""Rerun statistics into a fresh temporary directory and compare CSV numerics."""
import tempfile,subprocess,sys,shutil,json
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1]
def main():
 checked=[]
 with tempfile.TemporaryDirectory(prefix='typhoon_repro_') as tmp:
  out=Path(tmp);(out/'data').mkdir();shutil.copy2(W/'data/paired_by_lead.csv',out/'data/paired_by_lead.csv')
  with (W/'provenance/reproduction_repeat.log').open('w') as f:
   subprocess.run([sys.executable,str(W/'code/experiments3to6.py'),'--project-root',str(ROOT),'--output-dir',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
  for p in sorted((out/'data').glob('*.csv')):
   if p.name=='paired_by_lead.csv':continue
   a=pd.read_csv(p);b=pd.read_csv(W/'data'/p.name);assert_frame_equal(a,b,check_exact=True);checked.append({'file':p.name,'rows':len(a),'exact_numeric_match':True})
 (W/'provenance/reproduction_check.json').write_text(json.dumps(checked,indent=2));print('Exactly reproduced',len(checked),'CSV tables')
if __name__=='__main__':main()
