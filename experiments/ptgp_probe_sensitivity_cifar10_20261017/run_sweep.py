import csv,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/sda/rainbow/yanch_gmm/ptgp_probe_sensitivity_cifar10_gpu1_20261017')
HERE=Path(__file__).resolve().parent
PY='/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python'
def atomic(path,value):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p)
def refresh():
 rows=[]
 for case in sorted((ROOT/'cases').iterdir()):
  name=case.name;parts=name.split('_');groups=int(parts[1]);per_class=int(parts[3])
  path=case/'methods'/'ptgp'/'round_metrics.csv'
  train=list(csv.DictReader(path.open())) if path.exists() else []
  status=case/'status'/'cifar10_geiping_ptgp_s1234.json'
  item=dict(case=name,groups=groups,per_class=per_class,rounds=len(train),
            round20_test_accuracy=float(train[-1]['test_accuracy']) if len(train)==20 else '',
            round20_train_loss=float(train[-1]['train_loss']) if len(train)==20 else '')
  if status.exists():
   d=json.loads(status.read_text())
   if d.get('complete'):
    for metric in ('mse','lpips','psnr','ssim'):
     vals=[float(r[metric]) for r in d['rows']]
     import statistics as st
     item[metric+'_mean']=st.mean(vals);item[metric+'_sample_sd']=st.stdev(vals)
  rows.append(item)
 cols=['case','groups','per_class','rounds','round20_test_accuracy','round20_train_loss'] + [m+s for m in ('mse','lpips','psnr','ssim') for s in ('_mean','_sample_sd')]
 with (ROOT/'sensitivity_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
case_paths=sorted((ROOT/'cases').iterdir())
case_paths.sort(key=lambda p:(p.name!='groups_032_perclass_200',p.name))
for position,case in enumerate(case_paths,1):
 env=os.environ.copy();env['SPECTRA_EXPERIMENT_ROOT']=str(case)
 train_dir=case/'methods'/'ptgp'
 atomic(ROOT/'stage.json',dict(phase='running',case=case.name,position=position,total=19,time=time.time()))
 if not (train_dir/'COMPLETE').exists():
  with (case/'logs'/'training.log').open('a') as log:
   subprocess.run([PY,str(HERE/'train.py'),'--method','ptgp','--rounds','20','--seed','1234','--checkpoint-every','5','--output',str(train_dir)],cwd='/home/yanch/GMM',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 refresh()
 marker=case/'status'/'cifar10_geiping_ptgp_s1234.json'
 if not (marker.exists() and json.loads(marker.read_text()).get('complete')):
  with (case/'logs'/'privacy_driver.log').open('a') as log:
   subprocess.run([PY,str(HERE/'matrix.py')],cwd='/home/yanch/GMM',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 if not (marker.exists() and json.loads(marker.read_text()).get('complete')):
  raise RuntimeError('IG privacy evaluation incomplete: '+case.name)
 refresh()
 atomic(ROOT/'stage.json',dict(phase='case_complete',case=case.name,position=position,total=19,time=time.time()))
atomic(ROOT/'stage.json',dict(phase='COMPLETE',total=19,time=time.time()))
(ROOT/'COMPLETE').write_text('19 unique cases: 10 values per axis with shared baseline\n')
