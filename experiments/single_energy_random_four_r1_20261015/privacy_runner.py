"""Original attack engine, leading task projector; no attack-guided selection."""
import os,json,sys,torch,numpy as np
from pathlib import Path
import inversefed,defense
import torch.nn.functional as F
import subspace as S
R=S.ROOT; E=Path(__file__).resolve().parent
cfg=json.loads(Path(os.environ['MATRIX_CONFIG']).read_text()); name=cfg['dataset']; folder=R/'references'/(name+'_seed'+str(cfg['set_seed']));folder.mkdir(parents=True,exist_ok=True)
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
  reference_ds=original('FFHQ64',a[1],data_path='/mnt/sda/rainbow/yanch_gmm/data/ffhq_train64')[2].dataset
 return result
inversefed.construct_dataloaders=capture

def adapted(g,model,x,y,**kw):
 global basis
 if basis is None:
  cache=folder/'basis.pt'
  if cache.exists():basis=torch.load(cache,map_location=x.device,weights_only=False)['basis']
  else:
   ds=reference_ds;labels=getattr(ds,'targets',None)
   if labels is None and hasattr(ds,'dataset') and hasattr(ds.dataset,'targets'):labels=np.array(ds.dataset.targets)[list(ds.indices)]
   if labels is None:labels=[int(ds[i][1]) for i in range(len(ds))]
   labels=np.array(labels);rng=np.random.default_rng(1234);classes=np.unique(labels)
   per=max(1,2000//len(classes));chosen=[rng.choice(np.flatnonzero(labels==c),min(per,int((labels==c).sum())),replace=False).tolist() for c in classes]
   ids=[]
   for j in range(max(map(len,chosen))):
    for c in rng.permutation(len(classes)):
     if j<len(chosen[c]):ids.append(int(chosen[c][j]))
   groups=[v.tolist() for v in np.array_split(ids,32) if len(v)]
   ps=[p for p in model.parameters() if p.requires_grad];G=torch.empty(len(groups),sum(p.numel() for p in ps),device="cpu")
   with S.preserved_probe_state(model):
    model.train()
    for j,group in enumerate(groups):
     items=[ds[i] for i in group];a=torch.stack([v[0] for v in items]).to(x.device);b=torch.tensor([int(v[1]) for v in items],device=x.device)
     grad=torch.autograd.grad(F.cross_entropy(model(a),b),ps);G[j]=torch.cat([v.detach().flatten() for v in grad]).cpu()
   U,spectrum,error=S.decompose(G);basis=U[:1].detach();torch.save(dict(basis=basis.cpu()),cache)
   S.atomic_json(folder/'reference.json',dict(indices=ids,groups=groups,class_counts={str(c):int((labels[ids]==c).sum()) for c in classes},source='original validation/evaluation dataset',rank=len(U),orthogonality_error=error))
  S._BASIS=basis;S._LAST=dict(candidate_rank=32,retained_rank=1)
 return S.spectra_gradient(g,model,x,y)
defense.spectra_gradient=adapted
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60,0)
torch.backends.cuda.matmul.allow_tf32=False
source=(E/'original_run_rec.py').read_text()
exec(compile(source,'/home/yanch/GMM/run_rec.py','exec'),{'__name__':'__main__','__file__':'/home/yanch/GMM/run_rec.py'})
