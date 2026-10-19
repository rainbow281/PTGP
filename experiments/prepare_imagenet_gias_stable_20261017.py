"""Create an isolated, matched ImageNet/GIAS rerun after BigGAN NaNs."""
import json
import shutil
from pathlib import Path

project = Path('/home/yanch/GMM')
old_code = project / 'experiments/imagenet_all_methods_rerun_20261016'
new_code = project / 'experiments/imagenet_gias_stable_20261017'
old_root = Path('/mnt/sda/rainbow/yanch_gmm/privacy_imagenet_all_methods_rerun_gpu1_20261016')
root = Path('/mnt/sda/rainbow/yanch_gmm/privacy_imagenet_gias_stable_gpu1_20261017')
if new_code.exists() or root.exists():
    raise RuntimeError('Refusing to overwrite existing experiment')
shutil.copytree(old_code, new_code, ignore=shutil.ignore_patterns('__pycache__'))
for name in ('status', 'logs', 'configs', 'model_audit', 'privacy'):
    (root/name).mkdir(parents=True, exist_ok=True)
(root/'references').symlink_to(old_root/'references', target_is_directory=True)
jobs = json.loads((old_root/'privacy_jobs.json').read_text())
jobs = [j for j in jobs if j['slug']=='imagenet' and j['attack']=='gias']
if len(jobs)!=7:
    raise RuntimeError(f'Expected 7 GIAS methods, got {len(jobs)}')
order = ['spectra','none','noise','clipping','compression','soteria','censor']
jobs.sort(key=lambda j: order.index(j['method']))
for job in jobs:
    config = job['config']
    assert config['gias_lr']==0.001 and config['gias_iterations']==300
    config['gias_lr']=0.00001  # Shared, conservative BigGAN parameter-search LR for every defense.
    config['exp_name']=config['exp_name'].replace('all_methods_rerun','gias_stable_rerun')
    config['output_dir']=str(root/'privacy/imagenet'/job['method']/'seed1234')
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
(root/'privacy_jobs.json').write_text(json.dumps(jobs,indent=2))
(root/'protocol.json').write_text(json.dumps({
    'source_experiment':str(old_root), 'dataset':'ImageNet-1K',
    'attack':'GIAS', 'methods':order, 'targets':jobs[0]['config']['target_indices'],
    'seed':1234, 'batch_size':1, 'random_classifier':True,
    'only_change':'GIAS BigGAN parameter-search learning rate 0.001 -> 0.00001 for every method',
    'reason':'Previous 55,880,833-parameter search diverged to NaN on all seven methods.'
},indent=2))
run = f'''#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export SPECTRA_EXPERIMENT_ROOT={root}
export PYTHONPATH={new_code}:{project}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STYLEGAN2_USE_CUDA_EXT=0
cd {project}
exec flock -x /mnt/sda/rainbow/yanch_gmm/gpu1_privacy_convergence_20261016.lock \\
 nice -n 19 /mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python {new_code}/matrix.py
'''
(new_code/'run.sh').write_text(run)
(new_code/'run.sh').chmod(0o755)
print(root)
print(new_code/'run.sh')
