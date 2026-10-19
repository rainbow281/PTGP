"""Resumable CIFAR-100/FFHQ GIFD + repeated-query EOT comparison."""
import csv
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ['SPECTRA_EXPERIMENT_ROOT'])
HERE = Path(__file__).resolve().parent
PYTHON = '/mnt/sda/rainbow/yanch_gmm/venv_GMM/bin/python'
NAMES = {'spectra': 'PTGP', 'censor': 'CENSOR', 'none': 'No Def.',
         'noise': 'Noise', 'clipping': 'Clipping', 'compression': 'Sparsi',
         'soteria': 'Soteria'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def aggregate(jobs):
    rows = []
    for job in jobs:
        key = f"{job['dataset']}_{job['method']}_eot{job['eot_runs']}"
        marker = ROOT / 'status' / f'{key}.json'
        if not marker.exists():
            continue
        item = json.loads(marker.read_text())
        if not item.get('complete'):
            continue
        for metric in item['rows']:
            rows.append(dict(dataset=job['dataset'], method=NAMES[job['method']],
                             method_key=job['method'], eot_runs=job['eot_runs'],
                             effective_eot_runs=(job['eot_runs'] if job['method'] in ('censor', 'noise') else 1),
                             attack='GIFD', seed=job['seed'], **metric))
    # Deterministic transforms yield identical observations in the EOT arm.
    for row in list(rows):
        if row['method_key'] not in ('censor', 'noise'):
            rows.append({**row, 'eot_runs': 20, 'effective_eot_runs': 1})
    if not rows:
        return
    cols = ['dataset', 'method', 'method_key', 'eot_runs', 'effective_eot_runs',
            'attack', 'seed', 'target', 'mse', 'lpips', 'psnr', 'ssim']
    with (ROOT / 'per_target_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['method'], row['method_key'], row['eot_runs']), []).append(row)
    import statistics as stats
    with (ROOT / 'summary_mean_sd.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'method', 'eot_runs', 'effective_eot_runs', 'metric', 'n', 'mean', 'sample_sd'])
        for (dataset, name, key, eot), values in sorted(grouped.items()):
            for metric in ('mse', 'lpips', 'psnr', 'ssim'):
                samples = [float(v[metric]) for v in values]
                writer.writerow([dataset, name, eot, eot if key in ('censor', 'noise') else 1,
                                 metric, len(samples), stats.mean(samples),
                                 stats.stdev(samples) if len(samples) > 1 else 0])


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    jobs = json.loads((ROOT / 'privacy_jobs.json').read_text())
    for name in ('status', 'logs', 'configs', 'model_audit'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs, 1):
        key = f"{job['dataset']}_{job['method']}_eot{job['eot_runs']}"
        marker = ROOT / 'status' / f'{key}.json'
        if marker.exists() and json.loads(marker.read_text()).get('complete'):
            continue
        config = job['config']
        config_path = ROOT / 'configs' / f'{key}.json'
        atomic_json(config_path, config)
        atomic_json(ROOT / 'stage.json', dict(phase='RUNNING', job=index,
                                            total=len(jobs), key=key, time=time.time()))
        env = os.environ.copy()
        env.update(MATRIX_CONFIG=str(config_path), GIFD_EOT_RUNS=str(job['eot_runs']),
                   TORCH_HOME='/mnt/sda/rainbow/yanch_gmm/cache/cifar10/torch',
                   MPLCONFIGDIR='/mnt/sda/rainbow/yanch_gmm/cache/cifar10/matplotlib',
                   GIA_NUM_WORKERS='0')
        args = [PYTHON, str(HERE / 'privacy_runner.py'), '--config',
                str(HERE / 'original_default_config.yml'), '--batch_size', '1']
        for field, value in config.items():
            args.extend(['--set', field + '=' + json.dumps(value)])
        print('START', index, len(jobs), key, flush=True)
        with (ROOT / 'logs' / f'{key}.log').open('a') as stream:
            subprocess.run(args, cwd='/home/yanch/GMM', env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        tables = list(Path(config['output_dir']).glob(config['exp_name'] + '*/epoch_0/table_Metrics.csv'))
        if not tables:
            raise FileNotFoundError(f'GIFD metrics missing for {key}')
        table = max(tables, key=lambda p: p.stat().st_mtime)
        with table.open(newline='') as stream:
            measured = list(csv.DictReader(stream))
        expected = sorted(config['target_indices'])
        actual = sorted(int(row['target_id']) for row in measured)
        if actual != expected:
            raise RuntimeError(f'Incomplete targets for {key}: {actual}')
        prefix = f"Best_first_{config['end_layer']}_layer"
        values = [dict(target=int(row['target_id']),
                       mse=float(row[prefix + '_mse_i']),
                       lpips=(float(row[prefix + '_lpips(vgg)']) +
                              float(row[prefix + '_lpips(alex)'])) / 2,
                       psnr=float(row[prefix + '_psnr']),
                       ssim=float(row[prefix + '_ssim'])) for row in measured]
        if any(not math.isfinite(value[name]) for value in values
               for name in ('mse', 'lpips', 'psnr', 'ssim')):
            raise RuntimeError(f'Nonfinite attack metric for {key}')
        atomic_json(marker, dict(complete=True, key=key, rows=values,
                                 metrics_path=str(table), eot_runs=job['eot_runs']))
        aggregate(jobs)
        print('COMPLETE', key, flush=True)
    atomic_json(ROOT / 'stage.json', dict(phase='COMPLETE', completed=len(jobs), time=time.time()))
    (ROOT / 'COMPLETE').write_text('All GIFD+EOT comparison jobs completed.\n')


if __name__ == '__main__':
    try:
        run()
    except Exception:
        error = traceback.format_exc()
        atomic_json(ROOT / 'stage.json', dict(phase='ERROR', error=error, time=time.time()))
        (ROOT / 'ERROR').write_text(error)
        raise
