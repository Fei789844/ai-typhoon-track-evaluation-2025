"""Reproduce manuscript from the preserved local project.
Example: python code/run_all.py --root /path/to/project
Install requirements.txt first. Default recomputes all GRIB tracking.
"""
import argparse,subprocess,sys,json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
 root=a.root.resolve()
 # supplement/build scripts locate the historical project through output position.
 if W.parents[1]!=root:raise SystemExit('Place this directory at PROJECT/实验结果/研究三至六_20261010')
 commands=[['retrack_verified.py','--root',str(root),'--out',str(W)],['verify_retracking.py'],['verify_tie_case.py'],['verify_sources.py'],['experiment2.py','--project-root',str(root),'--output-dir',str(W)],['experiments3to6.py','--project-root',str(root),'--output-dir',str(W)],['verify_geometry.py'],['supplement.py'],['field_diagnostics.py'],['verify_reproduction.py'],['verify_supplement.py'],['build_manuscript.py']]
 for args in commands:
  command=[sys.executable,str(W/'code'/args[0]),*args[1:]]
  print('RUN',command,flush=True)
  with (W/'provenance'/f'{args[0]}.log').open('w') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
 files=sorted((W/'data').glob('*.csv'))
 manifest={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 (W/'provenance/derived_output_hashes.json').write_text(json.dumps(manifest,indent=2))
 print('All stages completed')
if __name__=='__main__':main()
