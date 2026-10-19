"""Recover metrics from a completed GIFD table after a layer-name parser mismatch."""
import csv
import json
import math
from pathlib import Path

root = Path('/mnt/sda/rainbow/yanch_gmm/privacy_ptgp_ffhq_imagenet_cifar100_ref_expansion_gpu1_20261016')
job = next(j for j in json.loads((root / 'privacy_jobs.json').read_text())
           if j['slug'] == 'imagenet' and j['attack'] == 'gifd')
files = list(Path(job['config']['output_dir']).glob(job['config']['exp_name'] + '*/epoch_0/table_Metrics.csv'))
assert len(files) == 1, files
source = files[0]
raw = list(csv.DictReader(source.open()))
assert sorted(int(x['target_id']) for x in raw) == sorted(job['config']['target_indices'])
prefix = 'Best_first_12_layer'
rows = []
for x in raw:
    row = {
        'target': int(x['target_id']),
        'mse': float(x[prefix + '_mse_i']),
        'lpips': (float(x[prefix + '_lpips(vgg)']) + float(x[prefix + '_lpips(alex)'])) / 2,
        'psnr': float(x[prefix + '_psnr']),
        'ssim': float(x[prefix + '_ssim']),
    }
    assert all(math.isfinite(v) for k, v in row.items() if k != 'target')
    rows.append(row)
marker = root / 'status' / 'imagenet_gifd_spectra_s1234.json'
marker.write_text(json.dumps(dict(complete=True,dataset='imagenet',attack='gifd',method='spectra',
                                  seed=1234,rows=rows,metrics_path=str(source),
                                  recovered_note='end_layer=13 is exclusive; final recorded layer is 12'),indent=2))
(root / 'status' / 'imagenet_gifd_spectra_s1234.failure.json').unlink(missing_ok=True)
all_rows = []
for path in sorted((root / 'status').glob('*.json')):
    data = json.loads(path.read_text())
    if data.get('complete'):
        for row in data['rows']:
            all_rows.append(dict(dataset=data['dataset'],attack=data['attack'],method=data['method'],
                                 seed=data['seed'],**row))
with (root / 'per_target_metrics.csv').open('w',newline='') as handle:
    writer = csv.DictWriter(handle,fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
print('Recovered',marker,'rows',len(rows),'all rows',len(all_rows))
