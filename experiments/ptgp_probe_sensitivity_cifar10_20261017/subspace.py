"""Whole-model stable task subspace with continuous projection and transport."""
import os,json,contextlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
import inversefed

ROOT=Path(os.environ['SPECTRA_EXPERIMENT_ROOT'])
_REFERENCE=None
_BASIS=None
_ANCHOR=None
_LAST={}

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

@contextlib.contextmanager
def preserved_probe_state(model):
    modes=[(m,m.training) for m in model.modules()]
    buffers=[(b,b.detach().clone()) for b in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for b,v in buffers:b.copy_(v)
        for m,was in modes:m.training=was

def reference():
    global _REFERENCE
    if _REFERENCE is None:
        t=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
            inversefed.consts.cifar10_mean,inversefed.consts.cifar10_std)])
        ds=datasets.CIFAR10('/mnt/sda/rainbow/yanch_gmm/data',train=True,download=False,transform=t)
        groups=json.loads((ROOT/'reference.json').read_text())['groups']
        ids=[i for group in groups for i in group]
        assert len(ids)==len(set(ids)) and len(groups)>=2
        items=[ds[i] for i in ids]
        x=torch.stack([a for a,b in items]);y=torch.tensor([b for a,b in items])
        sizes=[len(g) for g in groups];_REFERENCE=(x,y,sizes)
    return _REFERENCE

def gradient_rows(model):
    ps=[p for p in model.parameters() if p.requires_grad];dev=ps[0].device
    x,y,sizes=reference()
    G=torch.empty(len(sizes),sum(p.numel() for p in ps),device='cpu',dtype=torch.float32)
    with preserved_probe_state(model):
        model.train();start=0
        for j,n in enumerate(sizes):
            gs=torch.autograd.grad(F.cross_entropy(model(x[start:start+n].to(dev)),y[start:start+n].to(dev)),ps)
            G[j].copy_(torch.cat([g.detach().reshape(-1) for g in gs]).cpu());start+=n
    if not torch.isfinite(G).all():raise FloatingPointError('Nonfinite reference gradients')
    return G

def orthogonalize(rows):
    output=[]
    for row in rows:
        v=row.clone()
        for _ in range(2):
            for q in output:v-=torch.dot(v,q)*q
        n=v.norm()
        if float(n)>1e-7:output.append(v/n)
    if not output:raise RuntimeError('No valid task directions')
    return torch.stack(output)

def decompose(G):
 gram=torch.zeros(G.shape[0],G.shape[0],dtype=torch.float64)
 for j in range(0,G.shape[1],500000):
  b=G[:,j:j+500000].double();gram+=b@b.T
 vals,vec=torch.linalg.eigh(gram);v=vals[-1]
 if not torch.isfinite(v) or v<=0:raise RuntimeError('Invalid leading energy')
 u=(vec[:,-1].float()@G)/v.sqrt().float();u=u/u.double().norm().float()
 return u.cuda().unsqueeze(0),v.sqrt().cuda().reshape(1),float(abs(u.double().square().sum()-1))

def candidate_indices(candidate=None):
    candidate=candidate or os.environ.get('SPECTRA_CANDIDATE')
    if candidate:
        return json.loads((ROOT/'candidates.json').read_text())[candidate]['indices']
    return json.loads((ROOT/'selected_policy.json').read_text())['indices']

def initial_anchor(indices,device):
    state=torch.load(ROOT/'initial_basis.pt',map_location='cpu',weights_only=False)
    return state['basis'][indices].to(device)

def restore_anchor(value):
    global _ANCHOR
    _ANCHOR=None if value is None else value.detach()

def reset():
    global _BASIS,_ANCHOR,_LAST
    _BASIS=None;_ANCHOR=None;_LAST={};torch.cuda.empty_cache()

def prepare_basis(model,candidate=None,initial=False):
    global _BASIS,_ANCHOR,_LAST
    _BASIS=None
    indices=candidate_indices(candidate)
    if initial:
        _BASIS=initial_anchor(indices,next(model.parameters()).device)
        _ANCHOR=_BASIS
        _LAST=dict(candidate_rank=json.loads((ROOT/'basis_metadata.json').read_text())['rank'],
            retained_rank=len(indices),transport_min_cosine=1.,transport_mean_cosine=1.,rank_reduced=False)
        return _LAST
    G=gradient_rows(model);U,spectrum,error=decompose(G);del G
    chosen=U[:1]
    principal=torch.ones(1,device=U.device) if _ANCHOR is None else torch.abs(chosen@_ANCHOR.T).flatten()
    _BASIS=chosen;_ANCHOR=chosen.detach()
    _LAST=dict(candidate_rank=len(U),retained_rank=1,
        transport_min_cosine=float(principal.min()),transport_mean_cosine=float(principal.mean()),
        rank_reduced=False,basis_orthogonality_error=error)
    return _LAST

def project(gs):
    if _BASIS is None:raise RuntimeError('Prepare a round-global basis before projection')
    flat=torch.cat([g.detach().reshape(-1) for g in gs]);coeff=_BASIS@flat;out=coeff@_BASIS
    raw=flat.square().sum().clamp_min(1e-30)
    if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite projected gradient')
    identity=float((torch.dot(flat,out)-out.square().sum()).abs()/raw)
    if identity>1e-4 or float(out.square().sum()/raw)>1.0001:
        raise FloatingPointError(f'Projection invariant failed: identity={identity:.8g}, basis_norm={float(_BASIS.double().norm()):.8g}, retained={float(out.square().sum()/raw):.8g}')
    result=[v.reshape_as(g).detach() for v,g in zip(out.split([g.numel() for g in gs]),gs)]
    return result,dict(retained_energy_fraction=float(out.square().sum()/raw),projection_identity_error=identity)

def spectra_gradient(g,model,x,y,**kw):
    h,d=project(g);inner=sum((a*b).sum() for a,b in zip(g,h))
    raw=sum(a.square().sum() for a in g).sqrt();norm=sum(a.square().sum() for a in h).sqrt()
    d.update(method='spectra_stable_task_attack_selected_projection',protected_virtual_loss=float('nan'),
        protected_clean_cosine=float(inner/(raw*norm+1e-30)),protected_gradient_norm=float(norm),
        continuous_coefficients=True,norm_cap=False,quantization=False,**_LAST)
    return h,d

def anchor_cpu():
    return None if _ANCHOR is None else _ANCHOR.detach().cpu()

def ablation_basis(G, method):
    """All protected variants retain exactly one normalized direction."""
    if method == 'ptgp':
        return decompose(G)[0]
    if method == 'task_min':
        gram = torch.zeros(G.shape[0], G.shape[0], dtype=torch.float64)
        for j in range(0, G.shape[1], 500000):
            block = G[:, j:j+500000].double()
            gram += block @ block.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eligible = torch.nonzero(eigenvalues > eigenvalues[-1] * 1e-8).flatten()
        if eligible.numel() == 0:
            raise RuntimeError('No reliably nonzero reference-gradient direction')
        index = int(eligible[0])
        v = eigenvectors[:, index] @ G.double()
        v = v / eigenvalues[index].sqrt()
    elif method == 'task_mean':
        # Summing 32 long float32 gradients before normalization accumulates
        # enough error to violate the orthogonal-projection invariant.
        v = G.double().mean(0)
    elif method == 'random_direction':
        generator = torch.Generator(device='cpu').manual_seed(1234)
        v = torch.randn(G.shape[1], generator=generator).double()
    else:
        raise ValueError(method)
    norm = torch.linalg.vector_norm(v)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise RuntimeError('Invalid ablation direction')
    v = (v / norm).float().to('cuda')
    v = v / torch.linalg.vector_norm(v.double()).float()
    return v.unsqueeze(0)
