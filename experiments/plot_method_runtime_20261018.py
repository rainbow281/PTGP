"""Compare measured CIFAR-10 federated-round runtimes on matched rounds.

Usage: python plot_method_runtime.py --root PATH_TO_THIS_FOLDER
Requires matplotlib and numpy. Reads methods/*/round_metrics.csv, then writes
runtime_summary.csv and method_runtime_clear.{png,pdf} under --root.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    ('none', 'No Def.', '#64748b'),
    ('noise', 'Noise', '#b79a20'),
    ('clipping', 'Clipping', '#54a24b'),
    ('sparsi', 'Sparsi', '#9d55a0'),
    ('soteria', 'Soteria', '#2e8b81'),
    ('censor', 'CENSOR', '#4c78a8'),
    ('spectra', 'PTGP', '#e45756'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--matched-rounds', type=int, default=None,
                        help='Freeze the shared comparison window (e.g. 189).')
    args = parser.parse_args()
    root = args.root.resolve()
    data = {}
    for key, _, _ in METHODS:
        with (root / 'methods' / key / 'round_metrics.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'No round data for {key}')
        data[key] = rows
    matched = args.matched_rounds or min(map(len, data.values()))
    if any(len(rows) < matched for rows in data.values()):
        raise ValueError('A method has fewer rounds than --matched-rounds')
    assert matched > 0
    baseline = np.mean([float(row['round_seconds']) for row in data['none'][:matched]])
    summary = []
    for key, label, color in METHODS:
        rows = data[key][:matched]
        times = np.array([float(row['round_seconds']) for row in rows])
        steps = np.array([int(row['local_steps']) for row in rows])
        assert np.all(times > 0) and np.all(steps > 0)
        q1, q3 = np.percentile(times, [25, 75])
        summary.append(dict(method=key, label=label, recorded_rounds=len(data[key]),
                            matched_rounds=matched, mean_seconds_per_round=times.mean(),
                            median_seconds_per_round=np.median(times), q1_seconds=q1,
                            q3_seconds=q3, mean_local_steps_per_round=steps.mean(),
                            seconds_per_local_step=times.sum()/steps.sum(),
                            relative_to_no_defense=times.mean()/baseline,
                            matched_total_hours=times.sum()/3600))

    columns = list(summary[0])
    with (root / 'runtime_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: f'{v:.6f}' if isinstance(v, (float, np.floating)) else v
                             for k, v in row.items()})

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 250})
    fig = plt.figure(figsize=(13.2, 5.8))
    ax = fig.add_axes([.13, .22, .55, .61])
    slow_ax = fig.add_axes([.78, .22, .19, .61])
    regular = [(r, meta) for r, meta in zip(summary, METHODS) if meta[0] != 'soteria']
    for y, (row, (key, label, color)) in enumerate(regular):
        ax.barh(y, row['mean_seconds_per_round'], height=.58, color=color,
                alpha=.90, edgecolor='none')
        ax.text(row['mean_seconds_per_round']+.22, y,
                f"{row['mean_seconds_per_round']:.2f} s  |  {row['relative_to_no_defense']:.2f}×",
                va='center', fontsize=10.5,
                fontweight='bold' if key in ('spectra', 'censor') else 'normal')
    ax.set_yticks(np.arange(len(regular)), [meta[1] for _, meta in regular], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 19)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel('Mean seconds per federated round')
    ax.set_title('Main comparison · 0–19 s', loc='left', fontsize=12, fontweight='bold')
    ax.grid(axis='x', color='#e3e9f0')
    ax.set_axisbelow(True)

    slow = next(r for r in summary if r['method'] == 'soteria')
    slow_ax.barh([0], [slow['mean_seconds_per_round']], height=.30,
                 color=next(c for k, _, c in METHODS if k == 'soteria'), alpha=.90)
    slow_ax.set_yticks([0], ['Soteria'], fontsize=11)
    slow_ax.set_ylim(-.85, .85)
    slow_ax.set_xlim(0, 340)
    slow_ax.set_xticks([0, 100, 200, 300])
    slow_ax.set_xlabel('Seconds per round')
    slow_ax.set_title('Soteria · 0–340 s', loc='left', fontsize=12, fontweight='bold')
    slow_ax.text(7, -.39,
                 f"{slow['mean_seconds_per_round']:.2f} s  |  {slow['relative_to_no_defense']:.1f}× No Def.",
                 fontsize=10.5, fontweight='bold')
    slow_ax.grid(axis='x', color='#e3e9f0')
    slow_ax.set_axisbelow(True)
    fig.suptitle('CIFAR-10 | Runtime per round by defense', fontsize=17,
                 fontweight='bold', y=.965)
    fig.text(.13, .90, f'First {matched} matched rounds · about {summary[0]["mean_local_steps_per_round"]:.1f} local steps per round',
             fontsize=10, color='#475569')
    fig.text(.13, .075,
             'Bars show measured means; the Soteria panel uses a separate, explicitly labeled linear scale.\n'
             'Time includes training and evaluation. Methods ran sequentially, so server load may differ.',
             fontsize=9, color='#64748b')
    fig.savefig(root / 'method_runtime_clear.png')
    fig.savefig(root / 'method_runtime_clear.pdf')
    plt.close(fig)
    print(f'Matched rounds: {matched}; outputs: {root}')


if __name__ == '__main__':
    main()
