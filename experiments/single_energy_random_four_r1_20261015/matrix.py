import os,json,csv,time,subprocess,traceback,math
from pathlib import Path
import subspace as S
E=Path(__file__).resolve().parent;R=S.ROOT;PY='/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python'
def summary():
 records=[]
 for p in (R/'status').glob('*.json'):
  d=json.loads(p.read_text())
  if d.get('complete'):
   for row in d['rows']:records.append(dict(dataset=d['dataset'],attack=d['attack'],method=d['method'],seed=d['seed'],**row))
 if records:
  with (R/'per_target_metrics.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def main():
 jobs=json.loads((R/'privacy_jobs.json').read_text());(R/'status').mkdir(exist_ok=True)
 for idx,j in enumerate(jobs):
  slug,attack,method,seed=[j[k] for k in ['slug','attack','method','seed']];c=j['config'];key=f'{slug}_{attack}_{method}_s{seed}'
  marker=R/'status'/(key+'.json')
  if marker.exists() and json.loads(marker.read_text()).get('complete'):continue
  p=R/'configs'/(key+'.json');p.write_text(json.dumps(c,indent=2))
  env=os.environ.copy();env.update(MATRIX_CONFIG=str(p),TORCH_HOME='/mnt/sda/rainbow/yanch_gmm/cache/cifar10/torch',MPLCONFIGDIR='/mnt/sda/rainbow/yanch_gmm/cache/cifar10/matplotlib',GIA_NUM_WORKERS='0')
  S.atomic_json(R/'stage.json',dict(job=idx+1,total=len(jobs),key=key,phase='RUNNING',time=time.time()))
  args=[PY,str(E/'privacy_runner.py'),'--config',str(E/'original_default_config.yml'),'--batch_size','1']
  for k,v in c.items():args+=['--set',k+'='+json.dumps(v)]
  print('START',idx+1,len(jobs),key,flush=True)
  existing=list(Path(c['output_dir']).glob(c['exp_name']+'*/epoch_0/table_Metrics.csv'))
  complete_tables=[p for p in existing if sorted(int(x['target_id']) for x in csv.DictReader(p.open()))==sorted(c['target_indices'])]
  if not complete_tables:
   with (R/'logs'/(key+'.log')).open('a') as stream:subprocess.run(args,cwd='/home/yanch/GMM',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  files=list(Path(c['output_dir']).glob(c['exp_name']+'*/epoch_0/table_Metrics.csv'))
  f=max(files,key=lambda p:p.stat().st_mtime);rows=list(csv.DictReader(f.open()))
  if sorted(int(x['target_id']) for x in rows)!=sorted(c['target_indices']):raise RuntimeError('Incomplete target set '+key)
  q={'geiping':'geiping','yin':'Yin','ggl':'ggl','gias':'gias','gifd':'Best_first_'+str(c.get('end_layer',3))+'_layer'}[attack]
  normalized=[dict(target=int(x['target_id']),mse=float(x[q+'_mse_i']),lpips=(float(x[q+'_lpips(vgg)'])+float(x[q+'_lpips(alex)']))/2,psnr=float(x[q+'_psnr']),ssim=float(x[q+'_ssim'])) for x in rows]
  if not all(math.isfinite(x[k]) for x in normalized for k in ['mse','lpips','psnr','ssim']):raise RuntimeError('Nonfinite metrics')
  S.atomic_json(marker,dict(complete=True,dataset=slug,attack=attack,method=method,seed=seed,rows=normalized,metrics_path=str(f)))
  # Ensure classifier initialization is matched across defenses and attacks.
  audits=[json.loads(p.read_text()) for p in (R/'model_audit').glob('*.json')]
  hashes={x['initial_state_sha256'] for x in audits if x['dataset']==c['dataset'] and x['seed']==seed}
  if len(hashes)!=1:raise RuntimeError('Classifier hashes differ '+key)
  summary();print('COMPLETE',key,flush=True)
 (R/'COMPLETE').write_text('40 jobs: four datasets x five attacks x two defenses x one seed1234')
 S.atomic_json(R/'stage.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:(R/'ERROR').write_text(traceback.format_exc());raise
