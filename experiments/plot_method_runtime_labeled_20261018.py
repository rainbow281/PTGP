"""Original log-scale runtime plot with method names labeled at each marker.

Usage: python plot_method_runtime_labeled.py --root DIRECTORY --matched-rounds 189
Reads methods/*/round_metrics.csv and writes method_runtime_labeled.{png,pdf}.
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
    parser.add_argument('--matched-rounds', type=int, default=189)
    args = parser.parse_args()
    root = args.root.resolve()
    values = {}
    for key, _, _ in METHODS:
        with (root / 'methods' / key / 'round_metrics.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < args.matched_rounds:
            raise ValueError(f'{key}: only {len(rows)} rounds available')
        values[key] = np.array([float(r['round_seconds']) for r in rows[:args.matched_rounds]])
    baseline = values['none'].mean()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 250})
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    fig.subplots_adjust(left=.19, right=.97, bottom=.20, top=.80)
    for y, (key, label, color) in enumerate(METHODS):
        samples = values[key]
        mean = samples.mean()
        q1, q3 = np.percentile(samples, [25, 75])
        ax.scatter(samples, np.full(samples.size, y), s=9, color=color,
                   alpha=.14, linewidths=0, rasterized=True, zorder=2)
        ax.plot([q1, q3], [y, y], color=color, lw=5,
                solid_capstyle='round', zorder=3)
        ax.scatter([mean], [y], s=115, marker='D', color=color,
                   edgecolor='white', linewidth=1.3, zorder=4)
        annotation = f'{label}  {mean:.1f} s  ({mean/baseline:.1f}×)'
        is_soteria = key == 'soteria'
        ax.annotate(annotation, (mean, y),
                    xytext=((-12, 0) if is_soteria else (12, 0)),
                    textcoords='offset points', va='center',
                    ha=('right' if is_soteria else 'left'),
                    fontsize=10, color='#172033',
                    fontweight='bold' if key in ('spectra', 'censor', 'soteria') else 'normal',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=.3),
                    zorder=5)
    ax.axvline(baseline, color='#64748b', lw=1, ls='--', alpha=.7)
    ax.set_xscale('log')
    ax.set_xlim(.9, 650)
    ax.set_ylim(-.65, len(METHODS)-.35)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(METHODS)), [label for _, label, _ in METHODS], fontsize=12)
    for tick, (_, _, color) in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(color)
        tick.set_fontweight('bold')
    ax.set_xlabel('Wall-clock time per federated round (seconds, log scale)')
    ax.grid(axis='x', color='#e3e9f0', which='both')
    ax.set_axisbelow(True)
    fig.suptitle('CIFAR-10 | Measured training cost by defense', fontsize=16,
                 fontweight='bold', y=.96)
    fig.text(.19, .865,
             f'First {args.matched_rounds} matched rounds · about 83.5 local steps/round',
             fontsize=10, color='#475569')
    fig.text(.19, .058,
             'Each method is named beside its mean diamond. Thick line: interquartile range; faint dots: rounds.\n'
             'Dashed line: No Def. mean. Timings include training/evaluation; server load may differ.',
             fontsize=9, color='#64748b')
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'method_runtime_labeled.{ext}')
    plt.close(fig)
    print(root / 'method_runtime_labeled.png')


if __name__ == '__main__':
    main()
