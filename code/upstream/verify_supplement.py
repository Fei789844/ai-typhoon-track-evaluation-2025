"""Repeat supplementary calculations and compare complete numeric products."""
from pathlib import Path
import subprocess,sys,json
import pandas as pd
W=Path(__file__).resolve().parents[1]
def main():
 names=['LOCO_secondary','direction_matched','group_contrasts','near_turn_risk','case_path_classes','group_interactions','component_fraction_ci','path_type_components','MC_seed_sensitivity','retracking_storm_means','retracking_summary','retracking_pairs','high_error_records']
 before={name:pd.read_csv(W/'data'/f'{name}.csv') for name in names}
 with (W/'provenance/supplement_repeat.log').open('w') as f:subprocess.run([sys.executable,str(W/'code/supplement.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 report=[]
 for name,old in before.items():
  new=pd.read_csv(W/'data'/f'{name}.csv');pd.testing.assert_frame_equal(old,new,check_exact=True);report.append({'file':name+'.csv','rows':len(new),'exact_numeric_match':True})
 (W/'provenance/supplement_reproduction_check.json').write_text(json.dumps(report,indent=2));print('Exactly reproduced',len(report),'supplementary CSV tables')
if __name__=='__main__':main()
