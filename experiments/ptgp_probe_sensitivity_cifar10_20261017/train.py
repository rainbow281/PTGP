"""Shared, explicit FL protocol for calibration and 2,000-round validation."""
import argparse,csv,json,os,time,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
import inversefed,defense
import subspace as S

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)

@torch.no_grad()
def evaluate(model,loader):
    modes=[(m,m.training) for m in model.modules()];model.eval()
    loss=correct=count=0
    try:
        for x,y in loader:
            x,y=x.cuda(),y.cuda();z=model(x)
            loss+=float(F.cross_entropy(z,y,reduction='sum'));correct+=int((z.argmax(1)==y).sum());count+=len(y)
        return loss/count,correct/count
    finally:
        for m,was in modes:m.training=was

def protect(method,g,model,x,y):
    if method=="none":return g,{}
    if method not in ('ptgp','task_mean','task_min','random_direction'):
        with S.preserved_probe_state(model):
            if method=="censor":return defense.orthogonal_gradient(g,model,x,y,trials=20,best_loss=float("inf"))[0],{}
            if method=="noise":return defense.additive_noise(model,[v.clone() for v in g],None,std=.01),{}
            if method=="clipping":return defense.gradient_clipping([v.clone() for v in g],bound=4),{}
            if method=="sparsi":return defense.gradient_compression(g,percentage=20),{}
            if method=="soteria":return defense.perturb_representation(g,model,x,pruning_rate=10),{}
        raise ValueError(method)
    return S.project(g)

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45,0)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    seed(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    norm=(inversefed.consts.cifar10_mean,inversefed.consts.cifar10_std)
    train=datasets.CIFAR10('/mnt/sda/rainbow/yanch_gmm/data',train=True,download=False,
        transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(*norm)]))
    test=datasets.CIFAR10('/mnt/sda/rainbow/yanch_gmm/data',train=False,download=False,
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(*norm)]))
    # CENSOR reports a non-IID Dirichlet split but does not specify alpha.
    # Use a fixed alpha=0.5 and save the exact split for reproducibility.
    rng=np.random.default_rng(a.seed)
    labels=np.asarray(train.targets)
    for attempt in range(1000):
        clients=[[] for _ in range(100)]
        for cls in range(10):
            ids=rng.permutation(np.flatnonzero(labels==cls))
            sizes=rng.multinomial(len(ids),rng.dirichlet(np.full(100,.5)))
            for client_id,part in enumerate(np.split(ids,np.cumsum(sizes)[:-1])):
                clients[client_id].extend(part.tolist())
        if min(map(len,clients))>=64:break
    else:raise RuntimeError('Could not construct 100 nonempty Dirichlet client shards')
    shards=[np.asarray(x,dtype=np.int64) for x in clients]
    np.savez_compressed(out/'partition.npz',**{str(i):s for i,s in enumerate(shards)})
    loader=DataLoader(test,batch_size=128,shuffle=False,num_workers=0)
    model,_=inversefed.construct_model('ResNet18',num_classes=10,num_channels=3,seed=a.seed);model.cuda()
    if a.method in ('ptgp','task_mean','task_min','random_direction') and not (S.ROOT/'reference.json').exists():
        ref_rng=np.random.default_rng(a.seed)
        pools=[ref_rng.choice(np.flatnonzero(labels==cls),200,replace=False).tolist() for cls in range(10)]
        ids=[pools[cls][j] for j in range(200) for cls in ref_rng.permutation(10)]
        groups=[part.tolist() for part in np.array_split(ids,32)]
        S.atomic_json(S.ROOT/'reference.json',dict(indices=ids,groups=groups,
            class_counts={str(cls):200 for cls in range(10)},source='CIFAR-10 training split',seed=a.seed))
    initial_loss,initial_acc=evaluate(model,loader)
    S.atomic_json(out/'config.json',dict(method=a.method,candidate=a.candidate,rounds=a.rounds,seed=a.seed,
        clients=100,clients_per_round=10,batch_size=64,local_epochs=1,optimizer='SGD',lr=.1,momentum=0,
        weight_decay=0,partition='non-IID classwise Dirichlet, alpha=0.5 (explicit assumption)',initial_loss=initial_loss,initial_accuracy=initial_acc,
        test_protocol='class-balanced 2000-image training reference; evaluation split excluded from direction construction',
        initial_model='random ResNet18 seed1234',
        basis_refresh='One rank-1 direction recomputed per global round for task variants; random direction fixed by seed',
        checkpoint_every=a.checkpoint_every,privacy_attacks_in_training=False))
    fields=['round','train_loss','train_accuracy','test_loss','test_accuracy','local_steps','examples',
        'mean_descent_cosine','mean_raw_norm','mean_gradient_norm','mean_retained_energy',
        'candidate_rank','retained_rank','transport_min_cosine','transport_mean_cosine','round_seconds']
    checkpoint=out/'checkpoint.pt';start=1
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);start=state['round']+1

    csvpath=out/'round_metrics.csv'
    if csvpath.exists():
        with csvpath.open() as f:old=[r for r in csv.DictReader(f) if int(r['round'])<start]
        with csvpath.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(old)
    with csvpath.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if stream.tell()==0:writer.writeheader()
        for round_id in range(start,a.rounds+1):
            begun=time.time();basis={}
            if a.method in ('ptgp','task_mean','task_min','random_direction'):
                G=S.gradient_rows(model) if a.method!='random_direction' else torch.empty(1,sum(p.numel() for p in model.parameters()))
                S._BASIS=S.ablation_basis(G,a.method);del G
                basis=dict(candidate_rank=len(S.reference()[2]) if a.method!='random_direction' else 0,retained_rank=1)
            global_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            aggregate={k:torch.zeros_like(v) for k,v in global_state.items() if v.is_floating_point()}
            clients=np.random.default_rng(a.seed+round_id).choice(100,10,replace=False)
            total_loss=correct=count=steps=0;cosines=[];raw_norms=[];norms=[];energy=[]
            for client in clients:
                s=a.seed+1009*round_id+int(client);seed(s);model.load_state_dict(global_state)
                opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=0,weight_decay=0)
                local=DataLoader(Subset(train,shards[int(client)]),batch_size=64,shuffle=True,num_workers=0,
                    generator=torch.Generator().manual_seed(s))
                for batch_id,(x,y) in enumerate(local):
                    model.train();x,y=x.cuda(),y.cuda();z=model(x);loss=F.cross_entropy(z,y)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    total_loss+=float(loss)*len(y);correct+=int((z.argmax(1)==y).sum());count+=len(y)
                    g=[q.detach() for q in torch.autograd.grad(loss,tuple(model.parameters()))]
                    # Defense RNG must not change subsequent training augmentations.
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(s+7919*batch_id)
                        h,diag=protect(a.method,g,model,x,y)
                    raw=sum(q.square().sum() for q in g).sqrt();n=sum(q.square().sum() for q in h).sqrt()
                    inner=sum((a*b).sum() for a,b in zip(g,h))
                    cosines.append(float(inner/(raw*n+1e-30)));raw_norms.append(float(raw));norms.append(float(n))
                    energy.append(float(n.square()/raw.square().clamp_min(1e-30)))
                    opt.zero_grad(set_to_none=True)
                    for p,q in zip(model.parameters(),h):
                        if not torch.isfinite(q).all():raise FloatingPointError('Nonfinite protected update')
                        p.grad=q
                    opt.step();steps+=1
                with torch.no_grad():
                    for k,v in model.state_dict().items():
                        if k in aggregate:aggregate[k].add_(v,alpha=.1)
            model.load_state_dict({k:aggregate.get(k,v) for k,v in global_state.items()})
            test_loss,test_acc=evaluate(model,loader)
            row=dict(round=round_id,train_loss=total_loss/count,train_accuracy=correct/count,
                test_loss=test_loss,test_accuracy=test_acc,local_steps=steps,examples=count,
                mean_descent_cosine=float(np.mean(cosines)),mean_raw_norm=float(np.mean(raw_norms)),
                mean_gradient_norm=float(np.mean(norms)),mean_retained_energy=float(np.mean(energy)),
                candidate_rank=basis.get('candidate_rank',''),retained_rank=basis.get('retained_rank',''),
                transport_min_cosine=basis.get('transport_min_cosine',''),
                transport_mean_cosine=basis.get('transport_mean_cosine',''),round_seconds=time.time()-begun)
            writer.writerow(row);stream.flush();print('ROUND_DONE',a.method,a.candidate,json.dumps(row),flush=True)
            if round_id==1 or round_id%a.checkpoint_every==0 or round_id==a.rounds:
                tmp=checkpoint.with_suffix('.tmp')
                torch.save(dict(round=round_id,model={k:v.cpu() for k,v in model.state_dict().items()},
                    anchor=None),tmp);os.replace(tmp,checkpoint)
    (out/'COMPLETE').write_text(str(a.rounds))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ptgp','task_mean','task_min','random_direction','none'],required=True)
    p.add_argument('--candidate');p.add_argument('--rounds',type=int,default=2000)
    p.add_argument('--seed',type=int,default=1234);p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as e:
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/'ERROR').write_text(repr(e));raise
