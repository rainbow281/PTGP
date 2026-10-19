from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/mnt/sda/rainbow/yanch_gmm/convergence_cifar10_all_methods_censor_protocol_gpu1_20261016')
OUT = ROOT / 'visualization_current'
OUT.mkdir(exist_ok=True)
METHODS = [
    ('none', 'No defense', '#66758c'),
    ('noise', 'Noise', '#d5aa28'),
    ('clipping', 'Clipping', '#23854a'),
    ('sparsi', 'Compression', '#aa3982'),
    ('censor', 'CENSOR', '#377db5'),
    ('spectra', 'PTGP', '#d94d3f'),
    ('soteria', 'Soteria', '#7e64a6'),
]


def read_metrics(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    return rows


def trailing_mean(values, width=25):
    vals = np.asarray(values, dtype=float)
    csum = np.cumsum(np.insert(vals, 0, 0.0))
    return (csum[1:] - csum[np.maximum(0, np.arange(len(vals)) + 1 - width)]) / np.minimum(np.arange(len(vals)) + 1, width)


plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 160,
    'savefig.dpi': 240,
})
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.79, wspace=0.20)
summary = []

for key, name, color in METHODS:
    path = ROOT / 'methods' / key / 'round_metrics.csv'
    if not path.exists():
        continue
    rows = read_metrics(path)
    if not rows:
        continue
    rounds = np.array([int(r['round']) for r in rows])
    loss = np.array([float(r['train_loss']) for r in rows])
    accuracy = 100 * np.array([float(r['test_accuracy']) for r in rows])
    suffix = f' ({len(rows)}/2000)' if len(rows) < 2000 else ''
    label = name + suffix
    for ax, values in zip(axes, (loss, accuracy)):
        ax.plot(rounds, values, color=color, alpha=0.11, lw=0.6)
        ax.plot(rounds, trailing_mean(values), color=color, lw=2.35 if key == 'spectra' else 1.8,
                label=label)
    summary.append([name, len(rows), f'{loss[0]:.4f}', f'{loss[-1]:.4f}',
                    f'{accuracy[0]:.2f}', f'{accuracy[-1]:.2f}',
                    f'{np.mean(accuracy[-min(100,len(rows)):]):.2f}'])

for ax in axes:
    ax.grid(axis='y', color='#e4e9f0', lw=0.8)
    ax.set_xlim(0, 2000)
    ax.set_xlabel('Communication round')
axes[0].set_title('Training loss', loc='left', weight='bold')
axes[0].set_ylabel('Cross-entropy')
axes[1].set_title('Test accuracy', loc='left', weight='bold')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           frameon=False, columnspacing=1.6)
fig.text(0.07, 0.055, 'Faint: recorded per-round value; solid: trailing 25-round mean. Incomplete methods stop at their last recorded round.',
         fontsize=9, color='#64748b')
fig.savefig(OUT / 'cifar10_convergence_current.png')
fig.savefig(OUT / 'cifar10_convergence_current.pdf')
plt.close(fig)
with (OUT / 'summary_current.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method','rounds','initial_train_loss','last_train_loss','initial_test_accuracy_percent','last_test_accuracy_percent','last_100_test_accuracy_percent'])
    writer.writerows(summary)
print(OUT / 'cifar10_convergence_current.png')
for row in summary:
    print(','.join(map(str, row)))
