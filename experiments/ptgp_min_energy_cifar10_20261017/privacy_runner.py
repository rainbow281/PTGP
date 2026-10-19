"""Original attack engine, leading task projector; no attack-guided selection."""
import os,json,sys,torch,numpy as np
from pathlib import Path
import inversefed,defense
import torch.nn.functional as F
from torchvision import datasets
from inversefed.data.datasets import FFHQFolder
import subspace as S
R=S.ROOT; E=Path(__file__).resolve().parent
cfg=json.loads(Path(os.environ['MATRIX_CONFIG']).read_text()); variant=os.environ['PTGP_ABLATION_VARIANT']; name=cfg['dataset']; folder=R/'references'/(name+'_seed'+str(cfg['set_seed']));folder.mkdir(parents=True,exist_ok=True)
FFHQ_EVAL=Path('/mnt/sda/rainbow/yanch_gmm/data/ffhq_eval')
FFHQ_TRAIN=Path('/mnt/sda/rainbow/yanch_gmm/data/ffhq_train64/train')
if name=='FFHQ64':
 manifest=json.loads((FFHQ_EVAL/'ffhq_eval_manifest.json').read_text())
 age_labels={Path(item['path']).name:int(item['age_label']) for item in manifest['samples']}
 assert len(age_labels)==10 and set(age_labels.values()).issubset(set(range(10)))
 original_ffhq_getitem=FFHQFolder.__getitem__
 def corrected_ffhq_getitem(self,index):
  image,label=original_ffhq_getitem(self,index)
  if Path(self.root).resolve()==FFHQ_EVAL.resolve():
   filename=Path(self.samples[index][0]).name
   if filename not in age_labels:raise KeyError('FFHQ evaluation age label missing: '+filename)
   label=age_labels[filename]
  return image,label
 FFHQFolder.__getitem__=corrected_ffhq_getitem
import hashlib
original_model=inversefed.construct_model
def audited_model(*args,**kwargs):
 result=original_model(*args,**kwargs);model=result[0]
 digest=hashlib.sha256()
 for key,value in model.state_dict().items():
  digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
 S.atomic_json(R/'model_audit'/(cfg['exp_name']+'.json'),dict(dataset=name,model=cfg['model'],seed=cfg['set_seed'],trained_model=cfg['trained_model'],checkpoint=cfg.get('model_checkpoint_path'),initial_state_sha256=digest.hexdigest()))
 assert cfg['trained_model'] is False and not cfg.get('model_checkpoint_path')
 return result
inversefed.construct_model=audited_model
original=inversefed.construct_dataloaders
reference_ds=None;basis=None

def capture(*a,**kw):
 global reference_ds
 result=original(*a,**kw);reference_ds=result[2].dataset
 if cfg['dataset']=='FFHQ64':
  transform=reference_ds.dataset.transform if hasattr(reference_ds,'dataset') else reference_ds.transform
  reference_ds=datasets.ImageFolder(str(FFHQ_TRAIN),transform=transform)
  assert len(reference_ds.classes)==10 and reference_ds.classes==[str(i) for i in range(10)]
 elif cfg['dataset']=='CIFAR10':
  transform=reference_ds.dataset.transform if hasattr(reference_ds,'dataset') else reference_ds.transform
  reference_ds=datasets.CIFAR10('/mnt/sda/rainbow/yanch_gmm/data',train=True,download=False,transform=transform)
 elif cfg['dataset']=='CIFAR100':
  transform=reference_ds.dataset.transform if hasattr(reference_ds,'dataset') else reference_ds.transform
  reference_ds=datasets.CIFAR100('/mnt/sda/rainbow/yanch_gmm/data',train=True,
                                 download=False,transform=transform)
 return result
inversefed.construct_dataloaders=capture

def adapted(g,model,x,y,**kw):
 global basis
 if basis is None:
  cache=folder/('basis_'+variant+'.pt')
  if cache.exists():basis=torch.load(cache,map_location=x.device,weights_only=False)['basis']
  else:
   ds=reference_ds;labels=getattr(ds,'targets',None)
   if labels is None and hasattr(ds,'dataset') and hasattr(ds.dataset,'targets'):labels=np.array(ds.dataset.targets)[list(ds.indices)]
   if labels is None:labels=[int(ds[i][1]) for i in range(len(ds))]
   labels=np.array(labels);rng=np.random.default_rng(1234);classes=np.unique(labels)
   expected_classes={'CIFAR10':10,'FFHQ64':10,'IMAGENET_IO':1000,'CIFAR100':100}[name]
   assert len(classes)==expected_classes,(name,len(classes))
   per=20 if name=='IMAGENET_IO' else 100 if name=='CIFAR100' else 200
   excluded=set(cfg['target_indices']) if name=='IMAGENET_IO' else set()
   chosen=[]
   for c in classes:
    eligible=[int(i) for i in np.flatnonzero(labels==c) if int(i) not in excluded]
    if len(eligible)<per:raise RuntimeError(f'Only {len(eligible)} reference images in class {c}; need {per}')
    chosen.append(rng.choice(eligible,per,replace=False).tolist())
   ids=[]
   for j in range(max(map(len,chosen))):
    for c in rng.permutation(len(classes)):
     if j<len(chosen[c]):ids.append(int(chosen[c][j]))
   groups=[v.tolist() for v in np.array_split(ids,32) if len(v)]
   ps=[p for p in model.parameters() if p.requires_grad];G=torch.empty(len(groups),sum(p.numel() for p in ps),device="cpu")
   with S.preserved_probe_state(model):
    # Evaluation mode makes microbatch accumulation exact for larger reference groups.
    model.eval() if name in ('IMAGENET_IO','CIFAR100') else model.train()
    microbatch=32 if name in ('IMAGENET_IO','CIFAR100') else max(map(len,groups))
    for j,group in enumerate(groups):
     total=[torch.zeros_like(p) for p in ps]
     for k in range(0,len(group),microbatch):
      items=[ds[i] for i in group[k:k+microbatch]]
      a=torch.stack([v[0] for v in items]).to(x.device)
      b=torch.tensor([int(v[1]) for v in items],device=x.device)
      loss=F.cross_entropy(model(a),b,reduction='sum')/len(group)
      grad=torch.autograd.grad(loss,ps)
      for dst,src in zip(total,grad):dst.add_(src.detach())
     G[j]=torch.cat([v.flatten() for v in total]).cpu()
   basis=S.ablation_basis(G,variant).detach();error=0.0;torch.save(dict(basis=basis.cpu()),cache)
   S.atomic_json(folder/'reference.json',dict(indices=ids,groups=groups,
    class_counts={str(c):int((labels[ids]==c).sum()) for c in classes},
    source={'FFHQ64':'FFHQ age-labeled training split',
            'IMAGENET_IO':'ImageNet validation, excluding attacked target indices',
            'CIFAR100':'CIFAR-100 training split; disjoint from attacked test samples',
            'CIFAR10':'CIFAR-10 training split; disjoint from attacked test samples'}[name],
    target_indices=cfg['target_indices'],
    probe_mode='eval' if name in ('IMAGENET_IO','CIFAR100') else 'train',
    gradient_microbatch=microbatch,rank=32,variant=variant,orthogonality_error=error))
  S._BASIS=basis;S._LAST=dict(candidate_rank=32,retained_rank=1)
 return S.spectra_gradient(g,model,x,y)
defense.spectra_gradient=adapted
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55,0)
torch.backends.cuda.matmul.allow_tf32=False
source=(E/'original_run_rec.py').read_text()
exec(compile(source,'/home/yanch/GMM/run_rec.py','exec'),{'__name__':'__main__','__file__':'/home/yanch/GMM/run_rec.py'})
