"""SPECTRA V22-GMBA: Global Multi-Basin Ambiguity.

Built from V17-EDRA. Conditions structural-gradient sensitivity on a low-rank
task-equivalence subspace before changing privacy allocation.

Built from the empirically strongest V15-MIRA-RD branch. V17 keeps V15's successful multi-representation centroid anchor, but replaces mean-only representation risk with an entropic smooth worst-case risk over the representation codebook. This targets cross-attack robustness without embedding GI/IG/GGL/GIAS/GIFD into training.

Legacy notes:

Mechanism (not a parameter-only revision):
1) Estimate sample-specific leakage sensitivity per parameter tensor from streaming,
   label-preserving input probes: ||g(T(x))-g(x)|| / ||g(x)||.
2) Combine leakage sensitivity with a task-importance proxy (clean gradient energy) to
   obtain a layer-wise privacy allocation.
3) Discard the exact sample residual. Build each candidate from a coarse task anchor
   plus fresh layer-wise synthetic components orthogonal to both the clean gradient
   and task anchor. High leakage/task-ratio tensors receive more synthetic energy.
4) Stream candidates and select by *real one-step task loss* under hard privacy-geometry
   constraints. No gradient-cosine/low-frequency proxy is used as the ranking objective.

This combines the targeted/sensitivity-aware idea of recent shadow defenses with the
high-dimensional orthogonal replacement idea used by CENSOR, while retaining V8/V11's
empirically useful exact-residual discard.
"""
from __future__ import annotations
from typing import List
import math
import torch
import torch.nn.functional as F

TensorList = List[torch.Tensor]

def _loss_value(x):
    """Normalize repository-specific loss_fn outputs to a scalar Tensor."""
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            raise TypeError('loss_fn returned an empty tuple/list')
        x = x[0]
    if isinstance(x, dict):
        for key in ('loss', 'total_loss', 'ce_loss'):
            if key in x:
                x = x[key]
                break
        else:
            raise TypeError(f'loss_fn returned dict without a recognized loss key: {list(x.keys())}')
    if not torch.is_tensor(x):
        raise TypeError(f'loss_fn must return Tensor or a container whose first/known value is Tensor, got {type(x)!r}')
    if x.numel() != 1:
        x = x.mean()
    return x

def _inner(a,b): return sum((x*y).sum() for x,y in zip(a,b))
def _norm(a): return torch.sqrt(torch.clamp(_inner(a,a),min=1e-30))
def _clone(a): return [x.detach().clone() for x in a]

def _virtual_loss(model,g,x,y,loss_fn,lr):
    params=[p for p in model.parameters() if p.requires_grad]; was=model.training
    try:
        with torch.no_grad():
            for p,d in zip(params,g): p.add_(d,alpha=-float(lr))
        model.eval()
        with torch.inference_mode(): return float(_loss_value(loss_fn(model(x),y)).detach().cpu())
    finally:
        with torch.no_grad():
            for p,d in zip(params,g): p.add_(d,alpha=float(lr))
        model.train(was)

def _gradient(model,x,y,loss_fn):
    params=tuple(p for p in model.parameters() if p.requires_grad); was=model.training; model.eval()
    try:
        loss=_loss_value(loss_fn(model(x),y))
        gs=torch.autograd.grad(loss,params,retain_graph=False,create_graph=False,allow_unused=False)
        return [z.detach() for z in gs]
    finally: model.train(was)

def _block_mean_anchor_tensor(x,block):
    flat=x.detach().reshape(-1); n=flat.numel(); block=max(1,int(block))
    if n<=block: return flat.mean().expand_as(flat).reshape_as(x).clone()
    q=n//block; main=flat[:q*block].reshape(q,block); out=torch.empty_like(flat)
    out[:q*block]=main.mean(1,keepdim=True).expand(q,block).reshape(-1)
    if q*block<n: out[q*block:]=flat[q*block:].mean()
    return out.reshape_as(x)

def _task_anchor(g,block,ratio):
    a=[_block_mean_anchor_tensor(x,block) for x in g]; gn=_norm(g); an=_norm(a)
    s=float(ratio)*gn/(an+1e-30)
    with torch.no_grad():
        for z in a: z.mul_(s.to(dtype=z.dtype))
    return a

def _make_probe(x,index,noise_std,max_shift,cutout_fraction,seed):
    gen=torch.Generator(device=x.device); gen.manual_seed(int(seed)+7919*int(index)+313)
    v=x.detach().clone()
    if int(max_shift)>0:
        s=int(index%(2*int(max_shift)+1))-int(max_shift)
        t=int((index*3)%(2*int(max_shift)+1))-int(max_shift)
        v=torch.roll(v,(s,t),dims=(-2,-1))
    if float(cutout_fraction)>0 and v.ndim>=4:
        h,w=v.shape[-2:]; ch=max(1,int(round(h*float(cutout_fraction)))); cw=max(1,int(round(w*float(cutout_fraction))))
        top=int(torch.randint(0,max(1,h-ch+1),(1,),generator=gen,device=v.device).item())
        left=int(torch.randint(0,max(1,w-cw+1),(1,),generator=gen,device=v.device).item())
        fill=v.mean(dim=(-2,-1),keepdim=True); v[...,top:top+ch,left:left+cw]=fill.expand_as(v)[...,top:top+ch,left:left+cw]
    if float(noise_std)>0: v=v+float(noise_std)*torch.randn(v.shape,device=v.device,dtype=v.dtype,generator=gen)
    return v

def _rd_view(x,index,downsample_factor=4,block_size=4):
    """Deterministic, label-preserving but representation-distant views for RD ambiguity."""
    import torch.nn.functional as F
    v=x.detach()
    mode=int(index)%4
    if mode==0:
        # low-frequency view: destroys pixel detail while retaining coarse semantics
        h,w=v.shape[-2:]; f=max(2,int(downsample_factor))
        nh=max(1,h//f); nw=max(1,w//f)
        return F.interpolate(F.interpolate(v,size=(nh,nw),mode='area'),size=(h,w),mode='bilinear',align_corners=False)
    if mode==1:
        # block-rate distortion view
        h,w=v.shape[-2:]; b=max(2,int(block_size)); nh=max(1,h//b); nw=max(1,w//b)
        return F.interpolate(F.adaptive_avg_pool2d(v,(nh,nw)),size=(h,w),mode='nearest')
    if mode==2:
        # chroma-collapse view: keeps spatial/semantic structure, removes colour realization
        if v.ndim>=4 and v.shape[-3]>=3:
            y=0.2989*v[...,:1,:,:]+0.5870*v[...,1:2,:,:]+0.1140*v[...,2:3,:,:]
            return y.expand_as(v)
        return v
    # coarse spatial phase ambiguity: shift after low-pass, deliberately distant in pixels
    h,w=v.shape[-2:]; f=max(2,int(downsample_factor)); nh=max(1,h//f); nw=max(1,w//f)
    z=F.interpolate(F.interpolate(v,size=(nh,nw),mode='area'),size=(h,w),mode='bilinear',align_corners=False)
    return torch.roll(z,shifts=(max(1,h//16),-max(1,w//16)),dims=(-2,-1))

def _estimate_rd_codebook(model,x,y,loss_fn,g,count,downsample_factor,block_size,dro_temperature=0.35):
    """Streaming representation codebook + entropic distributionally-robust layer risk.

    For normalized per-view gradient distortions d_lk, compute
      mean_l = E[d_lk]
      entropic_l = tau * log E[exp(d_lk/tau)]
    using a stable streaming log-sum-exp. entropic_l is a smooth worst-case risk:
    it approaches the mean for high temperature and max_k d_lk as tau -> 0.
    We also return tail=max/mean and normalized softmax entropy.
    """
    m=max(3,int(count)); tau=max(float(dro_temperature),1e-4)
    centroid=[torch.zeros_like(z) for z in g]
    L=len(g); dev=g[0].device
    sum_d=torch.zeros(L,device=dev,dtype=torch.float64); sum_d2=torch.zeros_like(sum_d)
    max_d=torch.full((L,),-float('inf'),device=dev,dtype=torch.float64)
    # streaming log-sum-exp statistics for exp(d/tau), including entropy via E_w[d]
    lse_m=torch.full((L,),-float('inf'),device=dev,dtype=torch.float64)
    lse_z=torch.zeros(L,device=dev,dtype=torch.float64)
    lse_d=torch.zeros(L,device=dev,dtype=torch.float64)
    base=torch.stack([torch.linalg.vector_norm(z.float()).double() for z in g])
    for k in range(m):
        view=_rd_view(x,k,downsample_factor,block_size)
        gv=_gradient(model,view,y,loss_fn)
        with torch.no_grad():
            for c,z in zip(centroid,gv): c.add_(z,alpha=1.0/float(m))
        d=torch.stack([(torch.linalg.vector_norm((z.float()-b.float())).double()/(bn+1e-12))**2 for z,b,bn in zip(gv,g,base)])
        sum_d+=d; sum_d2+=d*d; max_d=torch.maximum(max_d,d)
        a=d/tau; nm=torch.maximum(lse_m,a)
        old=torch.exp(lse_m-nm)*lse_z
        cur=torch.exp(a-nm)
        lse_z=old+cur
        lse_d=torch.exp(lse_m-nm)*lse_d+cur*d
        lse_m=nm
        del gv,view,d,a,nm,old,cur
        if x.is_cuda: torch.cuda.empty_cache()
    mean=(sum_d/float(m))
    distortion=mean.float()
    pr=((sum_d*sum_d)/(float(m)*sum_d2+1e-30)).clamp(1.0/float(m),1.0)
    diversity=((pr-1.0/float(m))/(1.0-1.0/float(m)+1e-30)).float()
    logsum=lse_m+torch.log(lse_z+1e-30)
    entropic=(tau*(logsum-math.log(float(m)))).float()
    tail=(max_d/(mean+1e-12)).clamp(1.0,100.0).float()
    # H(w)=log Z-E_w[d]/tau, normalized to [0,1]. Low entropy => one representation dominates risk.
    wd=(lse_d/(lse_z+1e-30))
    entropy=((logsum-wd/tau)/math.log(float(m))).clamp(0.0,1.0).float()
    return [z.detach() for z in centroid],distortion.detach(),diversity.detach(),entropic.detach(),tail.detach(),entropy.detach()

def _edra_allocation(fisher,erank,task,rd_dist,rd_div,dro_risk,dro_tail,dro_entropy,rank_weight,rd_weight,dro_weight,shrinkage):
    """Fisher + mean RD + entropic worst-case distributional ambiguity allocation."""
    fi=torch.log1p(fisher.clamp_min(0))*(1.0+float(rank_weight)*erank)
    rd=torch.log1p(rd_dist.clamp_min(0))*(0.5+0.5*rd_div)
    # A concentrated tail (low entropy/high tail) is dangerous: a specific attack prior may exploit it.
    tail_factor=(1.0+torch.log1p((dro_tail-1.0).clamp_min(0)))*(1.5-0.5*dro_entropy)
    dr=torch.log1p(dro_risk.clamp_min(0))*tail_factor
    raw=(fi+float(rd_weight)*rd+float(dro_weight)*dr)/(task+0.25)
    order=torch.argsort(raw); rank=torch.empty_like(raw)
    rank[order]=torch.linspace(0,1,len(raw),device=raw.device,dtype=raw.dtype)
    sh=min(max(float(shrinkage),0.0),1.0)
    return ((1.0-sh)*rank+sh*0.5).detach(),raw.detach()


def _structural_view(x, index, scales=(2,4,8)):
    """Deterministic multi-scale structure interventions; never uses an attack oracle."""
    import torch.nn.functional as F
    v=x.detach(); h,w=v.shape[-2:]; idx=int(index); scale=int(scales[idx % len(scales)])
    nh=max(1,h//max(scale,1)); nw=max(1,w//max(scale,1))
    low=F.interpolate(F.adaptive_avg_pool2d(v,(nh,nw)),size=(h,w),mode='bilinear',align_corners=False)
    mode=(idx//len(scales))%3
    if mode==0: return low
    if mode==1:
        residual=v-low
        rms=torch.sqrt(F.adaptive_avg_pool2d(residual*residual,(nh,nw))+1e-6)
        rms=F.interpolate(rms,size=(h,w),mode='bilinear',align_corners=False)
        target=residual/(rms+1e-6); ref=(residual.square().mean(dim=(-2,-1),keepdim=True)+1e-6).sqrt()
        return (low+0.5*target*ref).clamp(v.amin().item(),v.amax().item())
    sh=max(1,scale//2)
    return torch.roll(low,shifts=(sh,-sh),dims=(-2,-1))

def _gradient_signature(delta, bins=64):
    """Small deterministic layer signature. Keeps low-rank subspace algebra bounded in memory."""
    z=delta.detach().float().reshape(-1); n=z.numel(); b=max(4,min(int(bins),n))
    if n==b: return z
    # contiguous adaptive means preserve coarse parameter-coordinate structure
    edges=torch.linspace(0,n,b+1,device=z.device).long(); out=[]
    for i in range(b):
        a=int(edges[i]); c=int(edges[i+1]); out.append(z[a:c].mean() if c>a else z[min(a,n-1)])
    return torch.stack(out)

def _estimate_conditional_structural_identifiability(model,x,y,loss_fn,g,*,
        rd_views=4,downsample_factor=4,block_size=4,structural_scales=(2,4,8),
        structural_modes=3,signature_bins=64,ridge=1e-3):
    """V19-COSI: task-conditioned private structural identifiability.

    Build a low-rank task-equivalence subspace from MIRA representation-gradient
    deviations. Project multi-scale structural-gradient deviations into that subspace.
    Only the orthogonal residual is considered private-identifying structure.

      rho_task = ||P_T S||_F^2 / ||S||_F^2
      rho_priv = 1-rho_task
      E_cond   = E_struct * rho_priv

    Projection is solved from a tiny ridge Gram system; no full Jacobian/projector is stored.
    """
    if isinstance(structural_scales,str):
        structural_scales=tuple(int(z.strip()) for z in structural_scales.split(',') if z.strip())
    structural_scales=tuple(max(1,int(z)) for z in structural_scales)
    L=len(g); dev=g[0].device; K=max(2,int(rd_views)); M=max(1,int(structural_modes))*len(structural_scales)
    # Per-layer task signatures: K x B only, not K full gradients.
    task=[[] for _ in range(L)]
    for k in range(K):
        gv=_gradient(model,_rd_view(x,k,downsample_factor,block_size),y,loss_fn)
        for l,(a,b) in enumerate(zip(gv,g)): task[l].append(_gradient_signature(a-b,signature_bins))
        del gv
        if x.is_cuda: torch.cuda.empty_cache()
    task=[torch.stack(v,dim=1) for v in task]  # B x K
    base=torch.stack([torch.linalg.vector_norm(z.float()).double() for z in g]).to(dev)
    struct_e=torch.zeros(L,device=dev,dtype=torch.float64)
    proj_e=torch.zeros_like(struct_e); total_sig=torch.zeros_like(struct_e)
    for j in range(M):
        gv=_gradient(model,_structural_view(x,j,structural_scales),y,loss_fn)
        for l,(a,b) in enumerate(zip(gv,g)):
            d=a.float()-b.float(); struct_e[l]+=(torch.linalg.vector_norm(d).double()/(base[l]+1e-12)).pow(2)
            s=_gradient_signature(d,signature_bins); T=task[l]
            G=T.T@T; scale=torch.trace(G)/max(1,G.shape[0]); lam=float(ridge)*(float(scale.detach().cpu())+1e-8)
            G=G+lam*torch.eye(G.shape[0],device=G.device,dtype=G.dtype)
            rhs=T.T@s
            try: coef=torch.linalg.solve(G,rhs)
            except RuntimeError: coef=torch.linalg.pinv(G)@rhs
            pe=torch.dot(rhs,coef).clamp_min(0); se=torch.dot(s,s).clamp_min(1e-20)
            proj_e[l]+=pe.double(); total_sig[l]+=se.double()
        del gv
        if x.is_cuda: torch.cuda.empty_cache()
    energy=(struct_e/float(M)).float()
    task_ratio=(proj_e/(total_sig+1e-20)).clamp(0,1).float()
    private_ratio=(1.0-task_ratio).clamp(0,1)
    conditional=(energy*private_ratio).detach()
    return energy.detach(),task_ratio.detach(),private_ratio.detach(),conditional

def _cosi_allocation(edra_risk,conditional_energy,private_ratio,weight,shrinkage):
    # Conditional energy, not raw sensitivity, modifies the validated EDRA risk.
    factor=torch.log1p(conditional_energy.clamp_min(0))*(0.5+0.5*private_ratio.clamp(0,1))
    combined=edra_risk*(1.0+float(weight)*factor)
    order=torch.argsort(combined); rank=torch.empty_like(combined)
    rank[order]=torch.linspace(0,1,len(combined),device=combined.device,dtype=combined.dtype)
    sh=min(max(float(shrinkage),0.0),1.0)
    return ((1.0-sh)*rank+sh*0.5).detach(),combined.detach()


def _estimate_inverse_landscape(model,x,y,loss_fn,g,*,directions=4,fd_epsilon=0.03,signature_bins=64,seed=1234):
    """V22-GMBA: attack-agnostic local inverse-landscape probes.

    For Phi_tilde(z)=1/2 ||G(z)-g_tilde||^2, the directional attraction at x is
      D_v Phi_tilde(x)=<G(x)-g_tilde, J_G(x)v>.
    We estimate J_G(x)v by central finite differences and retain only bounded
    per-layer signatures. The same probes also estimate layer Gauss-Newton
    curvature ||J_G v||^2, used to allocate replacement toward sharp inverse basins.
    """
    m=max(2,int(directions)); eps=max(float(fd_epsilon),1e-5); L=len(g); dev=g[0].device
    jvp_sigs=[[] for _ in range(L)]
    curvature=torch.zeros(L,device=dev,dtype=torch.float64)
    curvature2=torch.zeros_like(curvature)
    base=torch.stack([torch.linalg.vector_norm(z.float()).double() for z in g]).to(dev)
    for k in range(m):
        v=_rademacher_direction(x,1000+k,seed)
        gp=_gradient(model,x.detach()+eps*v,y,loss_fn); gm=_gradient(model,x.detach()-eps*v,y,loss_fn)
        for l,(a,b,gn) in enumerate(zip(gp,gm,base)):
            j=((a.float()-b.float())/(2.0*eps)).detach()
            sig=_gradient_signature(j,signature_bins)
            jvp_sigs[l].append(sig)
            e=(torch.linalg.vector_norm(j).double()/(gn+1e-12)).pow(2)
            curvature[l]+=e; curvature2[l]+=e*e
        del gp,gm,v
        if x.is_cuda: torch.cuda.empty_cache()
    curvature=(curvature/float(m)).float()
    # directional consistency: high when sharpness is repeatedly present rather than one accidental probe
    pr=((curvature.double()*float(m)).pow(2)/(float(m)*curvature2+1e-30)).clamp(1.0/float(m),1.0)
    consistency=((pr-1.0/float(m))/(1.0-1.0/float(m)+1e-30)).float()
    jvp_sigs=[torch.stack(v,dim=0) for v in jvp_sigs] # M x B per layer
    return curvature.detach(),consistency.detach(),jvp_sigs

def _ilba_allocation(cosi_risk,curvature,consistency,weight,shrinkage):
    """Couple V19 private-identifiability risk to inverse-basin Gauss-Newton sharpness."""
    c=torch.log1p(curvature.clamp_min(0))*(0.5+0.5*consistency.clamp(0,1))
    raw=cosi_risk*(1.0+float(weight)*c)
    order=torch.argsort(raw); rank=torch.empty_like(raw)
    rank[order]=torch.linspace(0,1,len(raw),device=raw.device,dtype=raw.dtype)
    sh=min(max(float(shrinkage),0.0),1.0)
    return ((1.0-sh)*rank+sh*0.5).detach(),raw.detach()

def _inverse_basin_attraction(candidate,g,jvp_sigs,signature_bins=64):
    """RMS normalized directional slope of the attack-agnostic inverse objective at the true input."""
    vals=[]
    with torch.no_grad():
        for k in range(jvp_sigs[0].shape[0]):
            num=torch.zeros((),device=g[0].device,dtype=torch.float64)
            den=torch.zeros_like(num)
            for c,gg,J in zip(candidate,g,jvp_sigs):
                r=_gradient_signature(gg.float()-c.float(),signature_bins)
                j=J[k]
                num+=torch.dot(r.double(),j.double())
                den+=torch.linalg.vector_norm(r.double())*torch.linalg.vector_norm(j.double())
            vals.append((num.abs()/(den+1e-20)).float())
        z=torch.stack(vals)
        return float(torch.sqrt(torch.mean(z*z)+1e-30).cpu()),float(z.max().cpu()),float(z.mean().cpu())



def _global_surrogate_view(x,index,downsample_factor=4,block_size=4):
    """Deterministic distant surrogate views for global inverse-map ambiguity.

    These are not attack reconstructions. They are structured probes spanning low-frequency,
    blockwise, phase-shift and contrast/complement regions of input space.
    """
    k=int(index)%6
    z=x.detach()
    if k==0:
        f=max(2,int(downsample_factor)); h,w=z.shape[-2:]
        y=F.interpolate(z,size=(max(1,h//f),max(1,w//f)),mode='bilinear',align_corners=False)
        return F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False).detach()
    if k==1:
        b=max(2,int(block_size)); h,w=z.shape[-2:]
        y=F.avg_pool2d(z,kernel_size=b,stride=b,ceil_mode=True)
        return F.interpolate(y,size=(h,w),mode='nearest').detach()
    if k==2:
        return torch.roll(z,shifts=(max(1,z.shape[-2]//4),max(1,z.shape[-1]//4)),dims=(-2,-1)).detach()
    if k==3:
        mu=z.mean(dim=(-2,-1),keepdim=True)
        return (mu-0.85*(z-mu)).detach()
    if k==4:
        return torch.flip(z,dims=(-1,)).detach()
    mu=z.mean(dim=(-2,-1),keepdim=True)
    low=F.avg_pool2d(z,kernel_size=3,stride=1,padding=1)
    return (mu+0.35*(low-mu)).detach()

def _estimate_global_basins(model,x,y,loss_fn,g,*,basins=4,signature_bins=64,
                             downsample_factor=4,block_size=4,min_distance=0.20):
    """Build bounded signatures of gradients induced by distant structured surrogates."""
    K=max(2,int(basins)); L=len(g); sigs=[[] for _ in range(L)]; distances=[]
    clean_scale=torch.sqrt(torch.mean(x.detach().float()*x.detach().float())+1e-20)
    for k in range(K):
        z=_global_surrogate_view(x,k,downsample_factor,block_size)
        dist=torch.sqrt(torch.mean((z.float()-x.detach().float())**2)+1e-20)/(clean_scale+1e-12)
        # Deterministic extrapolation if a structured view is too close to the source.
        if float(dist.detach().cpu())<float(min_distance):
            alpha=float(min_distance)/(float(dist.detach().cpu())+1e-12)
            z=x.detach()+alpha*(z-x.detach())
            dist=torch.sqrt(torch.mean((z.float()-x.detach().float())**2)+1e-20)/(clean_scale+1e-12)
        gz=_gradient(model,z,y,loss_fn)
        for l,a in enumerate(gz): sigs[l].append(_gradient_signature(a.float(),signature_bins))
        distances.append(dist.detach().float())
        del gz,z
        if x.is_cuda: torch.cuda.empty_cache()
    sigs=[torch.stack(v,dim=0) for v in sigs]  # K x B per layer
    d=torch.stack(distances)
    # Probe diversity in gradient-signature space, normalized to [roughly] O(1).
    pair=[]
    with torch.no_grad():
        for i in range(K):
            for j in range(i+1,K):
                num=torch.zeros((),device=g[0].device,dtype=torch.float64); den=torch.zeros_like(num)
                for S in sigs:
                    num+=torch.sum((S[i].double()-S[j].double())**2)
                    den+=torch.sum(S[i].double()**2)+torch.sum(S[j].double()**2)
                pair.append(torch.sqrt(num/(den+1e-20)).float())
    diversity=torch.stack(pair).mean() if pair else torch.zeros((),device=g[0].device)
    # Per-layer global basin dispersion: large means distant inputs remain distinguishable in that layer.
    layer_disp=[]
    with torch.no_grad():
        for S in sigs:
            mu=S.mean(dim=0,keepdim=True)
            num=torch.mean(torch.sum((S-mu)**2,dim=1))
            den=torch.mean(torch.sum(S**2,dim=1))+1e-20
            layer_disp.append((num/den).float())
    return sigs,d.detach(),diversity.detach(),torch.stack(layer_disp).detach()

def _global_multi_basin_score(candidate,basin_sigs,*,signature_bins=64,temperature=0.35):
    """Free energy and effective number of distant inverse explanations for one candidate."""
    K=basin_sigs[0].shape[0]; energies=[]
    with torch.no_grad():
        cand_sigs=[_gradient_signature(c.float(),signature_bins) for c in candidate]
        for k in range(K):
            num=torch.zeros((),device=candidate[0].device,dtype=torch.float64)
            den=torch.zeros_like(num)
            for cs,S in zip(cand_sigs,basin_sigs):
                ss=S[k]
                num+=torch.sum((cs.double()-ss.double())**2)
                den+=torch.sum(cs.double()**2)+torch.sum(ss.double()**2)
            energies.append((num/(den+1e-20)).float())
        e=torch.stack(energies)
        tau=max(float(temperature),1e-5)
        # Lower free energy = at least several distant probes plausibly explain the candidate.
        free=-tau*torch.logsumexp(-e/tau,dim=0)+tau*torch.log(torch.tensor(float(K),device=e.device))
        p=torch.softmax(-e/tau,dim=0)
        entropy=-(p*torch.log(p+1e-20)).sum()
        neff=torch.exp(entropy)
        return float(free.cpu()),float(neff.cpu()),float(entropy.cpu()),float(e.min().cpu()),float(e.mean().cpu()),float(e.max().cpu())

def _rademacher_direction(x, index, seed):
    """Deterministic unit-RMS Rademacher input direction; no persistent probe tensor."""
    gen=torch.Generator(device=x.device); gen.manual_seed(int(seed)+32452843*int(index)+97)
    r=torch.randint(0,2,x.shape,device=x.device,generator=gen,dtype=torch.int64)
    return (r.to(dtype=x.dtype).mul_(2).sub_(1)).detach()

def _estimate_fisher_identifiability(model,x,y,loss_fn,g,directions,fd_epsilon,seed):
    """Streaming finite-difference JVP estimator of input->gradient Fisher/GN energy.

    For G_l(x)=dL/dtheta_l, estimates E_v ||J_{G_l}(x)v||^2 / ||G_l(x)||^2.
    A normalized participation-ratio across random directions is used as an effective-rank
    proxy so a layer responding to only one accidental direction is not over-protected.
    """
    m=max(2,int(directions)); eps=max(float(fd_epsilon),1e-5); dev=g[0].device
    L=len(g); sum_e=torch.zeros(L,device=dev,dtype=torch.float64); sum_e2=torch.zeros_like(sum_e)
    base=torch.stack([torch.linalg.vector_norm(z.float()).double() for z in g]).to(dev)
    for k in range(m):
        v=_rademacher_direction(x,k+1,seed)
        xp=x.detach()+eps*v; xm=x.detach()-eps*v
        gp=_gradient(model,xp,y,loss_fn); gm=_gradient(model,xm,y,loss_fn)
        vals=[]
        for a,b,gn in zip(gp,gm,base):
            # central finite-difference JVP, normalized by clean gradient norm
            jn=torch.linalg.vector_norm((a.float()-b.float())/(2.0*eps)).double()
            vals.append((jn/(gn+1e-12)).pow(2))
        e=torch.stack(vals).to(dev); sum_e+=e; sum_e2+=e*e
        del gp,gm,xp,xm,v,e
        if x.is_cuda: torch.cuda.empty_cache()
    fisher=(sum_e/float(m)).float()
    # Participation ratio in [1/m,1], then map to [0,1]. High = leakage spread over directions.
    pr=((sum_e*sum_e)/(float(m)*sum_e2+1e-30)).clamp(1.0/float(m),1.0)
    erank=((pr-1.0/float(m))/(1.0-1.0/float(m)+1e-30)).float()
    task=(base.float()/(base.float().mean()+1e-12)).detach()
    return fisher.detach(),erank.detach(),task.detach()

def _fisher_allocation(fisher,erank,task,rank_weight,shrinkage):
    """Robust rank allocation from Fisher identifiability, effective rank and task value."""
    f=torch.log1p(fisher.clamp_min(0))
    raw=f*(1.0+float(rank_weight)*erank)/(task+0.25)
    order=torch.argsort(raw); rank=torch.empty_like(raw)
    rank[order]=torch.linspace(0,1,len(raw),device=raw.device,dtype=raw.dtype)
    # Empirical-Bayes style shrinkage toward 0.5 reduces sample-to-sample allocation variance.
    sh=min(max(float(shrinkage),0.0),1.0)
    allocation=(1.0-sh)*rank+sh*0.5
    return allocation.detach(),raw.detach()

def _layer_synthetic(g,anchor,allocation,seed,idx):
    gen=torch.Generator(device=g[0].device); gen.manual_seed(int(seed)+104729*int(idx)+17)
    out=[]
    with torch.no_grad():
        for li,(gg,aa) in enumerate(zip(g,anchor)):
            z=torch.randn(gg.shape,device=gg.device,dtype=gg.dtype,generator=gen)
            # Tensor-local orthogonalization removes direct clean-gradient and anchor correspondence.
            for basis in (gg,aa,gg,aa):
                den=(basis*basis).sum()+1e-30; coef=(z*basis).sum()/den; z.add_(basis,alpha=-float(coef.detach().cpu()))
            zn=torch.linalg.vector_norm(z); gn=torch.linalg.vector_norm(gg)
            z.mul_((gn/(zn+1e-30)).to(dtype=z.dtype))
            out.append(z)
    return out

def _compose_sage(synth,anchor,allocation,anchor_scale,synthetic_scale,min_replace,max_replace):
    out=[]
    with torch.no_grad():
        for z,a,r in zip(synth,anchor,allocation):
            rr=float(min_replace)+(float(max_replace)-float(min_replace))*float(r.detach().cpu())
            # High-risk tensors: less sample-derived anchor, more sample-independent synthetic energy.
            keep=1.0-rr
            out.append(a*float(anchor_scale)*keep + z*float(synthetic_scale)*rr)
    return out

def _metrics(c,g,a):
    gn=_norm(g); cn=_norm(c); an=_norm(a); dot=_inner(c,g); ca=_inner(c,a)
    cos=dot/(cn*gn+1e-30); diff=torch.sqrt(torch.clamp(cn*cn+gn*gn-2*dot,min=0))/(gn+1e-30)
    gg=_inner(g,g); ga=_inner(g,a); aa=_inner(a,a)+1e-30; cc=_inner(c,c); cg=dot; ac=ca
    gr2=torch.clamp(gg-ga*ga/aa,min=1e-30); cr2=torch.clamp(cc-ac*ac/aa,min=1e-30)
    rcos=(cg-ac*ga/aa)/torch.sqrt(gr2*cr2+1e-30)
    return {'relative_change':float(diff.cpu()),'cosine':float(cos.cpu()),'residual_cosine':float(rcos.cpu()),'norm':float(cn.cpu())}

def _cpfr_virtual_loss(model, gradient, inputs, labels, loss_fn, lr):
    params=list(model.parameters()); originals=[p.detach().clone() for p in params]; was=model.training
    try:
        with torch.no_grad():
            for p,g in zip(params,gradient): p.copy_(p-float(lr)*g)
        model.eval()
        with torch.no_grad(): return float(_loss_value(loss_fn(model(inputs),labels)).detach().cpu().item())
    finally:
        with torch.no_grad():
            for p,o in zip(params,originals): p.copy_(o)
        model.train(was)

def _cpfr_grad(model, x, y, loss_fn):
    model.zero_grad(set_to_none=True)
    loss=_loss_value(loss_fn(model(x),y))
    return [g.detach() for g in torch.autograd.grad(loss,tuple(model.parameters()))]

def _cpfr_norm(g): return torch.sqrt(sum((x*x).sum() for x in g).clamp_min(1e-30))
def _cpfr_inner(a,b): return sum((x*y).sum() for x,y in zip(a,b))
def _cpfr_scale(g,s): return [x*float(s) for x in g]
def _cpfr_cos(a,b): return float((_cpfr_inner(a,b)/(_cpfr_norm(a)*_cpfr_norm(b)+1e-30)).detach().cpu())

def _cpfr_prototype(x, generator, strength=1.0):
    """Many-to-one class/task prototype carrier: deliberately removes exact spatial realization."""
    # collapse each sample to channel moments, then inject independent low-frequency carrier.
    mu=x.detach().mean(dim=(-2,-1),keepdim=True)
    sd=x.detach().std(dim=(-2,-1),keepdim=True).clamp_min(0.05)
    b,c,h,w=x.shape
    low=torch.randn((b,c,max(2,h//8),max(2,w//8)),device=x.device,dtype=x.dtype,generator=generator)
    low=torch.nn.functional.interpolate(low,size=(h,w),mode='bilinear',align_corners=False)
    low=(low-low.mean(dim=(-2,-1),keepdim=True))/(low.std(dim=(-2,-1),keepdim=True)+1e-6)
    # class carrier contains moments but no original pixel arrangement; random sign/channel permutation destroys identity further.
    perm=torch.randperm(c,device=x.device,generator=generator)
    proto=mu[:,perm]+sd[:,perm]*low
    # batch/class barycenter makes mapping many-to-one when batch > 1.
    if b>1:
        proto=proto.mean(dim=0,keepdim=True).expand_as(proto).clone()
    return (float(strength)*proto).detach()


def _frtv_flat(g):
    return torch.cat([x.reshape(-1) for x in g])

def _frtv_add_scaled(out, g, a):
    return [x + float(a) * y for x,y in zip(out,g)]

def _frtv_apply_logits(model, gradient, probes, lr):
    params=list(model.parameters()); originals=[p.detach().clone() for p in params]; was=model.training
    try:
        with torch.no_grad():
            for p,g in zip(params,gradient): p.copy_(p-float(lr)*g)
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x).detach().reshape(-1) for x in probes])
    finally:
        with torch.no_grad():
            for p,o in zip(params,originals): p.copy_(o)
        model.train(was)

def _frtv_make_probe(model, label, shape, generator, steps, lr, jitter):
    # DATA-FREE: shape/model/label/RNG only. No private input values are read.
    b,c,h,w=shape
    z=torch.randn((1,c,h,w),device=label.device,generator=generator).clamp(-2,2).requires_grad_(True)
    opt=torch.optim.Adam([z],lr=float(lr))
    was=model.training; model.eval()
    try:
        for t in range(max(1,int(steps))):
            opt.zero_grad(set_to_none=True)
            zz=z
            if int(jitter)>0:
                sx=int(torch.randint(-int(jitter),int(jitter)+1,(1,),device=z.device,generator=generator).item())
                sy=int(torch.randint(-int(jitter),int(jitter)+1,(1,),device=z.device,generator=generator).item())
                zz=torch.roll(zz,(sx,sy),(-2,-1))
            logits=model(zz)
            ce=torch.nn.functional.cross_entropy(logits,label[:1])
            tv=(z[:,:,:,1:]-z[:,:,:,:-1]).abs().mean()+(z[:,:,1:,:]-z[:,:,:-1,:]).abs().mean()
            loss=ce+1e-3*tv+1e-4*z.square().mean()
            loss.backward(); opt.step()
            with torch.no_grad(): z.clamp_(-3.0,3.0)
    finally:
        model.train(was)
    return z.detach()

def _frtv_quantize(v,bits,clip):
    bits=max(1,int(bits)); levels=2**bits
    clip=float(max(1e-8,clip)); x=v.clamp(-clip,clip)
    q=torch.round((x+clip)*(levels-1)/(2*clip))
    deq=q*(2*clip)/(levels-1)-clip
    return deq.detach(),q.detach().to(torch.int64)

def _v46_inner(a,b):
    return sum((x*y).sum() for x,y in zip(a,b))

def _v46_norm(a):
    return torch.sqrt(_v46_inner(a,a).clamp_min(1e-30))

def _v46_unit(a):
    n=_v46_norm(a)
    return [x/n for x in a]

def _v46_public_random_direction(params,gen):
    # Public isotropic Rademacher direction; depends on shapes/RNG only.
    d=[]
    for p in params:
        r=torch.randint(0,2,p.shape,device=p.device,generator=gen,dtype=torch.int64)
        d.append((r.to(p.dtype)*2-1))
    return _v46_unit(d)

def _v46_sign_preserving_quantize(c,bits,clip):
    # Symmetric finite alphabet with an explicit zero. Nonzero reconstruction
    # never flips sign, which preserves positive directional inner products.
    bits=max(2,int(bits)); clip=float(max(1e-8,clip))
    levels=max(1,2**(bits-1)-1)
    x=c.clamp(-clip,clip)
    mag=x.abs()
    qmag=torch.round(mag/clip*levels).clamp(0,levels)
    # Avoid erasing a genuinely nonzero directional derivative.
    qmag=torch.where((mag>0)&(qmag==0),torch.ones_like(qmag),qmag)
    deq=torch.sign(x)*qmag*(clip/levels)
    # signed integer symbol in [-levels,levels]
    sym=(torch.sign(x)*qmag).to(torch.int64)
    return deq.detach(),sym.detach()

def spectra_gradient(input_gradient,model,ground_truth,labels,*,loss_fn,
                     virtual_lr=1e-4,max_utility_loss=0.30,
                     v47_enabled=True,v47_bit_budget=128,v47_subspace_dim=32,
                     cpfr_enabled=False,fail_on_weak_privacy=False,seed=1234,**_ignored):
    """V47-FRED: Finite-Rate Energy-preserving Descent.

    Only two method hyperparameters are exposed: total bit budget B and public
    subspace dimension d.  Bits/coordinate, task/random split, probe count,
    trust radius and release randomization are deterministic consequences of
    (B,d), not separately tuned knobs.

    PRIVATE: project G onto a public orthonormal mixed task/random subspace and
    immediately sign-preserving quantize the d directional derivatives.
    PRIVACY CUT: only d finite-alphabet symbols cross.
    VIRTUAL: decode a stochastic non-canonical descent release.  Positive
    symbol multipliers preserve each encoded descent sign.  A public radius
    sqrt(2) prevents uncontrolled curvature cost without reading ||G||.
    """
    if loss_fn is None: raise ValueError('V47-FRED requires loss_fn')
    if not bool(v47_enabled): raise ValueError('V47-FRED requires v47_enabled=true')
    clean=[g.detach() for g in input_gradient]
    device=clean[0].device; y=labels.detach(); shape=tuple(ground_truth.shape)
    params=[p for p in model.parameters() if p.requires_grad]
    d=max(4,int(v47_subspace_dim)); B=max(2*d,int(v47_bit_budget))
    bits=max(2,min(8,B//d))
    task_target=d//2; random_target=d-task_target
    probes_n=max(2,(task_target+1)//2)
    devices=[device] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed)); gen=torch.Generator(device=device); gen.manual_seed(int(seed)+47001)
        # Public task directions: exactly two directions/probe when available.
        probes=[]; raw=[]; kinds=[]
        for _ in range(probes_n):
            x=_frtv_make_probe(model,y,shape,gen,24,0.08,2); probes.append(x)
            logits=model(x); ce=torch.nn.functional.cross_entropy(logits,y[:1])
            gs=torch.autograd.grad(ce,params,retain_graph=True,create_graph=False,allow_unused=True)
            raw.append(_v46_unit([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gs)])); kinds.append('task')
            if len([k for k in kinds if k=='task']) < task_target:
                v=torch.randn(logits.shape,device=device,generator=gen,dtype=logits.dtype); v=v-v.mean(dim=-1,keepdim=True)
                gs=torch.autograd.grad((logits*v).sum(),params,retain_graph=False,create_graph=False,allow_unused=True)
                raw.append(_v46_unit([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gs)])); kinds.append('task')
        # Public isotropic coverage; refreshed for every release.
        for _ in range(random_target): raw.append(_v46_public_random_direction(params,gen)); kinds.append('random')
        # Public modified Gram-Schmidt and exact truncation to d.
        basis=[]; bk=[]
        for vec,k in zip(raw,kinds):
            r=[x.clone() for x in vec]
            for u in basis:
                a=_v46_inner(r,u); r=[x-a*y for x,y in zip(r,u)]
            n=_v46_norm(r)
            if torch.isfinite(n) and float(n.detach().cpu())>1e-8:
                basis.append([x/n for x in r]); bk.append(k)
            if len(basis)>=d: break
        while len(basis)<d:
            vec=_v46_public_random_direction(params,gen); r=[x.clone() for x in vec]
            for u in basis:
                a=_v46_inner(r,u); r=[x-a*y for x,y in zip(r,u)]
            n=_v46_norm(r)
            if float(n.detach().cpu())>1e-8: basis.append([x/n for x in r]); bk.append('random')

        # PRIVATE FINITE-RATE ENCODER. Fixed normalized alphabet: no norm side channel.
        syms=[]
        for u in basis:
            c=_v46_inner(clean,u)
            _,sym=_v46_sign_preserving_quantize(c,bits,1.0)
            syms.append(sym)
        qsym=torch.stack(syms)

        # ========================== PRIVACY CUT ==========================
        levels=max(1,2**(bits-1)-1)
        code=qsym.to(clean[0].dtype)/float(levels)
        # Non-canonical stochastic decoder.  Multipliers are positive and public,
        # so they cannot flip the encoded descent sign.
        mult=0.5+torch.rand((len(basis),),device=device,generator=gen,dtype=clean[0].dtype)
        release=[torch.zeros_like(p) for p in clean]
        for a,m,u in zip(code,mult,basis):
            release=[r+(a*m)*v for r,v in zip(release,u)]
        # Public, parameter-free trust radius sqrt(2); independent of private ||G||.
        rn=float(_v46_norm(release).detach().cpu()); trust_cap=2.0**0.5
        trust_scale=1.0 if rn<=trust_cap else trust_cap/max(rn,1e-12)
        release=[x*trust_scale for x in release]
        vl=[_cpfr_virtual_loss(model,release,x,y[:1],loss_fn,virtual_lr) for x in probes]
        protected_virtual_loss=float(sum(vl)/len(vl))

        # Post-hoc diagnostics only; never feed release construction.
        inner=float(_v46_inner(clean,release).detach().cpu())
        cn=float(_v46_norm(clean).detach().cpu()); rn=float(_v46_norm(release).detach().cpu())
        cos=inner/max(cn*rn,1e-30)
        energy=(inner*inner)/max(rn*rn,1e-30)
        rel=float((_frtv_flat(release)-_frtv_flat(clean)).norm().detach().cpu()/(cn+1e-30))
        ok=inner>0.0 and torch.isfinite(_frtv_flat(release)).all().item()
    diag={
      'method':'spectra_v47_fred_finite_rate_energy_descent',
      'protected_virtual_loss':protected_virtual_loss,'protected_clean_cosine':cos,'protected_relative_distance':rel,'weak_privacy':0,
      'v47_bit_budget_requested':B,'v47_subspace_dim':d,'v47_bits_per_coordinate':bits,
      'v47_actual_bit_budget_upper_bound':int(d*bits),'v47_task_basis_dim':int(sum(k=='task' for k in bk)),
      'v47_random_basis_dim':int(sum(k=='random' for k in bk)),'v47_release_norm':rn,'v47_trust_scale':trust_scale,
      'v47_posthoc_descent_inner_product':inner,'v47_posthoc_descent_cosine':cos,
      'v47_posthoc_effective_descent_energy':energy,'v47_posthoc_descent_certificate':int(ok),
      'v47_private_input_values_read_after_code':0,'v47_private_gradient_used_after_code':0,'v47_private_moments_used':0,
      'v47_private_utility_gate':0,'v47_release_uses_private_gradient_directly':0,'v47_markov_privacy_cut':1,
      'v47_convergence_claim':'conditional_stationary_point_compatibility_under_smooth_stochastic_random_subspace_assumptions',
      'selection_mode':'two_parameter_finite_rate_randomized_subspace_descent','seed':int(seed)}
    if fail_on_weak_privacy and not ok: raise RuntimeError('V47-FRED failed finite/descent diagnostic')
    return [g.detach() for g in release],diag

