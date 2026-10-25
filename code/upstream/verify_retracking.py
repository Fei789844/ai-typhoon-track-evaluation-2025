"""Independent baseline-only GRIB rerun and exact center/DPE comparison."""
from pathlib import Path
import tempfile,subprocess,sys,json
import pandas as pd
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1]
def main():
 with tempfile.TemporaryDirectory(prefix='typhoon_track_repeat_') as folder:
  command=[sys.executable,str(W/'code/retrack_verified.py'),'--root',str(ROOT),'--out',folder,'--baseline-only','--workers','4']
  with (W/'provenance/retracking_repeat.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  old=pd.read_csv(W/'data/retracking_points.csv');old=old[old.variant=='baseline'].reset_index(drop=True);new=pd.read_csv(Path(folder)/'data/retracking_points.csv');pd.testing.assert_frame_equal(old,new,check_exact=True)
  result={'records_checked':len(new),'exact_center_DPE_and_diagnostic_match':True,'workers':4,'method':'independent baseline-only GRIB reread'}
  (W/'provenance/retracking_reproduction_check.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
