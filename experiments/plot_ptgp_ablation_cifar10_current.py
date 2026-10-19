from pathlib import Path
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/mnt/sda/rainbow/yanch_gmm/ptgp_ablation_cifar10_gpu1_20261017')
OUT = ROOT / 'visualization_current'
OUT.mkdir(exist_ok=True)
METHODS = [
    ('none', 'No defense', '#64748b'),
    ('random_direction', 'Random direction', '#8b5cf6'),
    ('task_mean', 'Mean task direction', '#2a9d8f'),
    ('ptgp', 'PTGP: leading task direction', '#e45745'),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':240})

fig, axes = plt.subplots(1,2,figsize=(12.8,4.7))
fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.77,wspace=.21)
train_summary = []
for key,label,color in METHODS:
    path = ROOT/'methods'/key/'round_metrics.csv'
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    rounds=np.array([int(r['round']) for r in rows])
    loss=np.array([float(r['train_loss']) for r in rows])
    acc=100*np.array([float(r['test_accuracy']) for r in rows])
    axes[0].plot(rounds,loss,color=color,lw=2.25,marker='o',ms=2.4,label=label)
    axes[1].plot(rounds,acc,color=color,lw=2.25,marker='o',ms=2.4)
    train_summary.append([label,len(rows),f'{loss[0]:.4f}',f'{loss[-1]:.4f}',
                          f'{acc[0]:.2f}',f'{acc[-1]:.2f}'])
for ax in axes:
    ax.set_xlim(1,20);ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('Federated round')
    ax.grid(axis='y',color='#e5eaf0')
axes[0].set_title('Training loss',loc='left',weight='bold');axes[0].set_ylabel('Cross-entropy')
axes[1].set_title('Test accuracy',loc='left',weight='bold');axes[1].set_ylabel('Accuracy (%)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False)
fig.text(.08,.055,'All four variants completed 20 rounds under the same CIFAR-10 training protocol.',
         fontsize=9,color='#64748b')
fig.savefig(OUT/'ablation_training_20rounds.png')
fig.savefig(OUT/'ablation_training_20rounds.pdf')
plt.close(fig)
with (OUT/'training_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['method','rounds','initial_training_loss','round20_training_loss',
                                 'initial_test_accuracy_percent','round20_test_accuracy_percent']);w.writerows(train_summary)

privacy_path=ROOT/'per_target_metrics.csv'
with privacy_path.open(newline='') as f: privacy=list(csv.DictReader(f))
metrics=[('mse','MSE ↑'),('lpips','LPIPS ↑'),('psnr','PSNR ↓'),('ssim','SSIM ↓')]
summary=[]
for attack in sorted({r['attack'] for r in privacy}):
    groups=defaultdict(list)
    for row in privacy:
        if row['attack']==attack:groups[row['method']].append(row)
    if any(len(groups[k]) != 10 for k,_,_ in METHODS):
        continue
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.2))
    fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.85,hspace=.34,wspace=.25)
    for ax,(metric,title) in zip(axes.flat,metrics):
        x=np.arange(len(METHODS))
        means=np.array([statistics.mean(float(r[metric]) for r in groups[k]) for k,_,_ in METHODS])
        sds=np.array([statistics.stdev(float(r[metric]) for r in groups[k]) for k,_,_ in METHODS])
        ax.bar(x,means,color=[c for _,_,c in METHODS],width=.65,alpha=.95)
        ax.errorbar(x,means,yerr=sds,fmt='none',ecolor='#273449',capsize=3,lw=1.1)
        ax.set_xticks(x,['None','Random','Task mean','PTGP'])
        ax.tick_params(axis='x',labelsize=9)
        ax.set_title(title,loc='left',weight='bold')
        ax.grid(axis='y',color='#e5eaf0');ax.set_axisbelow(True)
        for idx,key in enumerate([k for k,_,_ in METHODS]):
            summary.append([attack,key,metric,len(groups[key]),f'{means[idx]:.6f}',f'{sds[idx]:.6f}'])
    attack_name={'geiping':'IG','yin':'GI'}.get(attack,attack.upper())
    fig.suptitle(f'CIFAR-10 ablation | {attack_name} reconstruction',fontsize=15,weight='bold',y=.96)
    fig.text(.13,.035,'Bars: mean across 10 matched targets; error bars: sample standard deviation.',
             fontsize=9,color='#64748b')
    fig.savefig(OUT/f'ablation_privacy_{attack}.png')
    fig.savefig(OUT/f'ablation_privacy_{attack}.pdf')
    plt.close(fig)
with (OUT/'privacy_summary_completed_attacks.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['attack','method','metric','n','mean','sample_sd']);w.writerows(summary)
print('OUTPUT',OUT)
print('TRAINING')
for r in train_summary:print(','.join(map(str,r)))
print('COMPLETE_ATTACKS',sorted({r[0] for r in summary}))
