import json,os,subprocess,time
from pathlib import Path
ROOT=Path(os.environ['SPECTRA_EXPERIMENT_ROOT'])
HERE=Path(__file__).resolve().parent
PY='/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python'
for position,method in enumerate(('ptgp','task_mean','random_direction','none'),1):
 out=ROOT/'methods'/method
 out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists():continue
 (ROOT/'ablation_stage.json').write_text(json.dumps(dict(phase='convergence',method=method,position=position,total=4,time=time.time())))
 with (ROOT/'logs'/f'convergence_{method}.log').open('a') as log:
  subprocess.run([PY,str(HERE/'train.py'),'--method',method,'--rounds','20','--seed','1234','--checkpoint-every','5','--output',str(out)],cwd='/home/yanch/GMM',env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
(ROOT/'ablation_stage.json').write_text(json.dumps(dict(phase='privacy',time=time.time())))
subprocess.run([PY,str(HERE/'matrix.py')],cwd='/home/yanch/GMM',env=os.environ.copy(),check=True)
