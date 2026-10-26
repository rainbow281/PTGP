"""Versioned evaluation of GMM only; invokes unmodified repo attacks/metrics/CENSOR.

Attack instances are fresh per method and attack. Test splits are explicit.
No update of victim weights, no changes to attack implementation, no filtering.
"""
import argparse,copy,csv,hashlib,json,logging,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch,torchvision,yaml,lpips
import inversefed,defense
from gmm_pdf import gmm_pdf_gradient
from inversefed.reconstruction_algorithms import DEFAULT_CONFIG

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True)
 p.add_argument('--methods',nargs='+',default=['none','censor','pdf32'])
 p.add_argument('--targets',nargs='+',type=int,default=[0])
 p.add_argument('--seeds',nargs='+',type=int,default=[1234])
 p.add_argument('--iterations',type=int,default=200)
 p.add_argument('--data',default='/Users/yancaihong/data')
 p.add_argument('--dataset',choices=['CIFAR10','ImageNet'],default='CIFAR10')
 p.add_argument('--image-size',type=int,default=64)
 p.add_argument('--model',default='ConvNet')
 p.add_argument('--checkpoint')
 p.add_argument('--width',type=int)
 p.add_argument('--gmm-config')
 p.add_argument('--restarts',type=int,default=1)
 p.add_argument('--attacks',nargs='+',choices=['GI','IG','GGL','GIAS','GIFD'],default=['GI','IG'])
 p.add_argument('--biggan',default='method_eval/assets/biggan-deep-256')
 p.add_argument('--gan-iterations',type=int,default=20)
 p.add_argument('--gias-iterations',type=int,default=20)
 p.add_argument('--gifd-stage-iterations',type=int,default=3)
 p.add_argument('--cma-rounds',type=int,default=15)
 p.add_argument('--cma-population',type=int,default=5)
 p.add_argument('--split',default='development')
 p.add_argument('--threads',type=int,default=2)
 p.add_argument('--signed',action='store_true')
 p.add_argument('--utility-samples',type=int,default=0)
 a=p.parse_args(); torch.set_num_threads(a.threads)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'results.jsonl').exists():raise RuntimeError('Use a NEW output directory to preserve every result')
 logging.basicConfig(filename=out/'attacks.log',level=logging.INFO)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 source=Path(__file__).resolve().parents[1]
 manifest={'args':vars(a),'device':str(device),'torch':torch.__version__,
  'source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in
   ['gmm_pdf.py','gmm_subspace.py','gmm_probes.py','gmm_layerwise_v3.py','defense.py','inversefed/reconstruction_algorithms.py','inversefed/metrics.py']}}
 if a.checkpoint:manifest['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (out/'source_snapshot').mkdir()
 for name in ['gmm_pdf.py','gmm_subspace.py','gmm_probes.py','gmm_layerwise_v3.py']:
  (out/'source_snapshot'/name).write_bytes((source/name).read_bytes())
 base=yaml.safe_load((source/'default_config.yml').read_text())
 overrides={} if not a.gmm_config else yaml.safe_load(Path(a.gmm_config).read_text())
 manifest['method_config']=overrides
 manifest['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (out/'source_snapshot'/'evaluate_versions.py').write_bytes(Path(__file__).read_bytes())
 utility_indices=[i for i in range(300,300+a.utility_samples+len(a.targets)) if i not in a.targets][:a.utility_samples]
 vgg=lpips.LPIPS(net='vgg').to(device).eval();alex=lpips.LPIPS(net='alex').to(device).eval()
 if a.dataset=='CIFAR10':
  ds=torchvision.datasets.CIFAR10(a.data,train=False,download=False,transform=torchvision.transforms.ToTensor())
  mean,stdev=inversefed.consts.cifar10_mean,inversefed.consts.cifar10_std
  manifest['data_manifest']={'dataset':'CIFAR10','split':'test','path':a.data,
                             'targets':a.targets,'utility_indices':utility_indices}
 else:
  from imagenet_eval_data import ParquetImageNet
  ds=ParquetImageNet(a.data,a.image_size,sorted(set(a.targets+utility_indices)))
  mean,stdev=inversefed.consts.imagenet_io_mean,inversefed.consts.imagenet_io_std
  manifest['data_manifest']=ds.manifest
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 dm=torch.tensor(mean,device=device)[:,None,None]
 std=torch.tensor(stdev,device=device)[:,None,None]
 allrows=[]
 for seed in a.seeds:
  if a.width is None:
   model,_=inversefed.construct_model(a.model,num_classes=1000 if a.dataset=='ImageNet' else 10,seed=seed)
  else:
   from inversefed.nn.models import ConvNet
   torch.manual_seed(seed)
   model=ConvNet(width=a.width,num_classes=10)
  if a.checkpoint:
   state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
   state=state.get('state_dict',state.get('model_state_dict',state))
   model.load_state_dict(state,strict=True)
  model=model.to(device).eval()
  for target in a.targets:
   pixel,label=ds[target];pixel=pixel[None].to(device);x=(pixel-dm)/std;y=torch.tensor([label],device=device)
   hooks=[inversefed.BNStatisticsHook(m) for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
   loss=torch.nn.functional.cross_entropy(model(x),y)
   g=[v.detach() for v in torch.autograd.grad(loss,model.parameters())]
   prior=[[v.detach().clone() for v in h.mean_var] for h in hooks]
   for h in hooks:h.close()
   for method in a.methods:
    random.seed(seed+target);np.random.seed(seed+target);torch.manual_seed(seed+target)
    started=time.time();diag={}
    if method=='none': protected=[v.clone() for v in g]
    elif method=='censor':
     protected,censorloss=defense.orthogonal_gradient(g,model,x,y,trials=20,epsilon=1e-4,best_loss=float(loss.detach()))
     diag={'protected_virtual_loss':float(censorloss)}
    elif method=='v3':
     from gmm_layerwise_v3 import gmm_layerwise_gradient
     cfg=dict(overrides);cfg['seed']=seed+target
     protected,diag=gmm_layerwise_gradient(g,model,x,y,**cfg)
    elif method in ('pdf8','pdf32','gmm'):
     cfg={'probe_count':8 if method=='pdf8' else 32,
          'energy_threshold':.95 if method=='pdf8' else 1.0,
          'seed':seed+target}
     if method=='gmm':cfg.update(overrides)
     protected,diag=gmm_pdf_gradient(g,model,x,y,**cfg)
    else:raise ValueError(method)
    model.eval()
    defense_seconds=time.time()-started
    if a.utility_samples:
     def accuracy():
      correct=0
      with torch.no_grad():
       for begin in range(0,len(utility_indices),8):
        batch=[ds[i] for i in utility_indices[begin:begin+8]]
        xx=torch.stack([v[0] for v in batch]).to(device)
        yy=torch.tensor([v[1] for v in batch],device=device)
        correct+=int((model((xx-dm)/std).argmax(1)==yy).sum())
      return correct/len(utility_indices)
     before=accuracy();saved=[p.detach().clone() for p in model.parameters()]
     try:
      with torch.no_grad():
       for p,u in zip(model.parameters(),protected):p.add_(u,alpha=-1e-4)
      after=accuracy()
     finally:
      with torch.no_grad():
       for p,v in zip(model.parameters(),saved):p.copy_(v)
     diag.update(accuracy_before=before,accuracy_after=after,accuracy_delta=after-before,
                 utility_validation_samples=len(utility_indices),utility_update_lr=1e-4)
    for attack in a.attacks:
     random.seed(seed+target);np.random.seed(seed+target);torch.manual_seed(seed+target)
     if device.type=='cuda':torch.cuda.manual_seed_all(seed+target)
     cfg=copy.deepcopy(DEFAULT_CONFIG)
     for key in ['cost_fn','indices','weights','lr','total_variation','bn_stat','image_norm','group_lazy','init']:
      cfg[key]=base[key]
     cfg.update(optim='adam',signed=a.signed,restarts=a.restarts,max_iterations=a.iterations,
      dataset='IMAGENET_IO' if a.dataset=='ImageNet' else 'CIFAR10',geiping=attack=='IG',yin=attack=='GI',generative_model='',gias=False,
      gifd=False,ggl=False,z_norm=0.,lr_decay=True)
     G=None
     if attack in ('GGL','GIAS','GIFD'):
      if a.dataset!='ImageNet':raise ValueError('This GAN protocol uses imagenet BigGAN')
      from inversefed import porting
      G=porting.load_decoder_biggan_io({'biggan_model':a.biggan,'biggan_cache_dir':str(out/'gan_cache')}).to(device).eval()
      cfg.update(generative_model='BigGAN',gen_dataset='ImageNet64',start_layer=0,end_layer=13,
       geiping=False,yin=False,ggl=attack=='GGL',gias=attack=='GIAS',gifd=attack=='GIFD',
       max_iterations=a.gan_iterations,gias_iterations=a.gias_iterations,gias_lr=1e-5,
       cma_budget=a.cma_rounds,num_sample=a.cma_population,KLD=.1,
       steps=[a.gifd_stage_iterations]*13,lr_io=[.1]*13,project=True,do_project_gen_out=True,
       do_project_noises=False,do_project_latent=False,
       max_radius_gen_out=[1000,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500],
       max_radius_noises=[-1]*13,max_radius_latent=[-1]*13)
     recon=inversefed.GradientReconstructor(model,device,(dm,std),cfg,num_images=1,bn_prior=prior,G=G)
     started=time.time()
     try:
      result=recon.reconstruct(protected,y,img_shape=tuple(x.shape[1:]))
      if attack=='GIFD':
       candidates=[r for r in result if r[0].startswith('Best_') and r[1] is not None]
       selected=candidates[-1]  # final cumulative best, chosen by gradient loss only
      else:
       assert len(result)==1
       selected=result[0]
      rec=selected[1]
      with torch.no_grad():
       rec01=(rec*std+dm).clamp(0,1)
       mse=float((rec01-pixel).square().mean())
       psnr=float(inversefed.metrics.psnr(rec01,pixel,factor=1))
       ssim=float(inversefed.metrics.ssim_batch(rec01,pixel)[0])
       vg=float(vgg(rec01*2-1,pixel*2-1).mean());al=float(alex(rec01*2-1,pixel*2-1).mean())
      row=dict(method=method,attack=attack,seed=seed,target=target,split=a.split,
       PSNR=psnr,SSIM=ssim,MSE=mse,LPIPS_VGG=vg,LPIPS_ALEX=al,
       inferred_label_correct=bool(torch.equal(recon.last_inferred_labels,y)),
       defense_seconds=defense_seconds,attack_seconds=time.time()-started,
       diagnostic=diag,selected_stage=selected[0],status='completed')
      torchvision.utils.save_image(rec01,out/f'{method}_{attack}_s{seed}_t{target}.png')
     except Exception as exc:
      row=dict(method=method,attack=attack,seed=seed,target=target,split=a.split,
               status='failed',error=repr(exc),diagnostic=diag)
     finally:
      for h in recon.bn_layers:h.close()
      del recon,G
      import gc;gc.collect()
     allrows.append(row)
     with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     print(json.dumps(row),flush=True)
  del model
 fields=['method','attack','seed','target','split','status','PSNR','SSIM','MSE','LPIPS_VGG','LPIPS_ALEX','inferred_label_correct']
 with (out/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allrows)
 if any(r['status']!='completed' for r in allrows):raise RuntimeError('Some attacks failed; see recorded rows')
if __name__=='__main__':main()
