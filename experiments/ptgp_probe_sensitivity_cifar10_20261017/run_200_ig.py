import csv,json,os,subprocess,time
from pathlib import Path
ROOT=Path(os.environ['SPECTRA_EXPERIMENT_ROOT'])
HERE=Path(__file__).resolve().parent
PY='/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python'
def stage(**kw):
 (ROOT/'ablation_stage.json').write_text(json.dumps(dict(time=time.time(),**kw),indent=2))
stage(phase='privacy',method='task_min')
subprocess.run([PY,str(HERE/'matrix.py')],cwd='/home/yanch/GMM',env=os.environ.copy(),check=True)
if not json.loads((ROOT/'status'/'cifar10_geiping_task_min_s1234.json').read_text()).get('complete'):
 raise RuntimeError('Minimum-energy IG attack did not complete')
for position,method in enumerate(('ptgp','task_min','random_direction','none'),1):
 out=ROOT/'methods'/method;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists():continue
 stage(phase='convergence',method=method,position=position,total=4,rounds=200)
 with (ROOT/'logs'/f'convergence_{method}.log').open('a') as log:
  subprocess.run([PY,str(HERE/'train.py'),'--method',method,'--rounds','200','--seed','1234','--checkpoint-every','20','--output',str(out)],cwd='/home/yanch/GMM',env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
stage(phase='COMPLETE')
(ROOT/'COMPLETE').write_text('IG 4-method comparison and 200-round training complete\n')
