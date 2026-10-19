from pathlib import Path
import csv
import json
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/mnt/sda/rainbow/yanch_gmm/ptgp_ablation_200round_ig_gpu1_20261017')
OUT=ROOT/'visualization_current'
OUT.mkdir(exist_ok=True)
METHODS=[
    ('none','No defense','#64748b'),
    ('random_direction','Random direction','#8b5cf6'),
    ('task_min','Minimum-energy direction','#1c9c89'),
    ('ptgp','PTGP main direction','#df5444'),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':240})

# Show only the new 200-round protocol's recorded points.
fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.76,wspace=.22)
training=[]
for key,label,color in METHODS:
    path=ROOT/'methods'/key/'round_metrics.csv'
    if not path.exists():
        training.append([label,0,'','',''])
        continue
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:
        training.append([label,0,'','',''])
        continue
    rounds=np.array([int(r['round']) for r in rows])
    loss=np.array([float(r['train_loss']) for r in rows])
    acc=100*np.array([float(r['test_accuracy']) for r in rows])
    suffix='' if len(rows)==200 else f' ({len(rows)}/200)'
    axes[0].plot(rounds,loss,color=color,lw=2.2,label=label+suffix)
    axes[1].plot(rounds,acc,color=color,lw=2.2)
    training.append([label,len(rows),f'{loss[0]:.4f}',f'{loss[-1]:.4f}',f'{acc[-1]:.2f}'])
for ax in axes:
    ax.set_xlim(0,200);ax.set_xticks([0,50,100,150,200]);ax.set_xlabel('Federated round')
    ax.grid(axis='y',color='#e4e9f0')
axes[0].set_title('Training loss',loc='left',weight='bold');axes[0].set_ylabel('Cross-entropy')
axes[1].set_title('Test accuracy',loc='left',weight='bold');axes[1].set_ylabel('Accuracy (%)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False)
fig.text(.08,.055,'All four methods completed 200 rounds under the same CIFAR-10 protocol.',fontsize=9,color='#64748b')
fig.savefig(OUT/'ablation_200_training_current.png');plt.close(fig)
with (OUT/'training_progress.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['method','recorded_rounds','initial_train_loss','latest_train_loss','latest_test_accuracy_percent']);w.writerows(training)

rows=[]
with (ROOT/'per_target_metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
groups=defaultdict(list)
for r in rows:
    if r['attack']=='geiping':groups[r['method']].append(r)
assert all(len(groups[k])==10 for k,_,_ in METHODS)
metrics=[('mse','MSE ↑'),('lpips','LPIPS ↑'),('psnr','PSNR ↓'),('ssim','SSIM ↓')]
fig,axes=plt.subplots(2,2,figsize=(10.3,7.2))
fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.85,hspace=.34,wspace=.25)
summary=[]
for ax,(metric,title) in zip(axes.flat,metrics):
    means=np.array([st.mean(float(r[metric]) for r in groups[key]) for key,_,_ in METHODS])
    sds=np.array([st.stdev(float(r[metric]) for r in groups[key]) for key,_,_ in METHODS])
    x=np.arange(4)
    ax.bar(x,means,color=[color for _,_,color in METHODS],width=.66)
    ax.errorbar(x,means,yerr=sds,fmt='none',ecolor='#253043',capsize=3,lw=1.1)
    ax.set_xticks(x,['None','Random','Min energy','PTGP']);ax.tick_params(axis='x',labelsize=9)
    ax.set_title(title,loc='left',weight='bold');ax.grid(axis='y',color='#e4e9f0');ax.set_axisbelow(True)
    for i,(key,label,_) in enumerate(METHODS):
        summary.append([key,label,metric,10,f'{means[i]:.6f}',f'{sds[i]:.6f}'])
fig.suptitle('CIFAR-10 ablation | IG reconstruction',fontsize=15,weight='bold',y=.96)
fig.text(.11,.035,'Mean ± sample standard deviation across the same 10 test targets; privacy measured at the initial model.',fontsize=9,color='#64748b')
fig.savefig(OUT/'ablation_ig_privacy.png');plt.close(fig)
with (OUT/'ig_privacy_mean_sd.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['method','label','metric','n','mean','sample_sd']);w.writerows(summary)

print('FIGURES',OUT)
print('TRAINING')
for row in training:print(','.join(map(str,row)))
print('PRIVACY_MSE')
for row in summary:
    if row[2]=='mse':print(','.join(map(str,row)))
