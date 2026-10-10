"""Run the official ACT loss/update recipe with auditable outer orchestration."""
from common import *
import argparse
import math
import random
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from examples.baselines.act.train_act_imitator import ACTAgent


def make_optimizer(agent,args,batches):
    backbone=[p for n,p in agent.named_parameters() if 'backbone' in n and p.requires_grad]
    other=[p for n,p in agent.named_parameters() if 'backbone' not in n and p.requires_grad]
    opt=torch.optim.AdamW([{'params':other},{'params':backbone,'lr':args.lr_backbone}],lr=args.lr,weight_decay=1e-4)
    scheduler=get_scheduler('cosine',optimizer=opt,num_warmup_steps=args.warmup_epochs*batches,num_training_steps=args.total_epochs*batches)
    scaler=torch.amp.GradScaler('cuda',enabled=False)  # Official BF16 path.
    return opt,scheduler,scaler


def update(agent,batch,opt,scheduler,scaler):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        losses=agent.compute_loss(batch)
    assert all(torch.isfinite(v).all() for v in losses.values()),losses
    opt.zero_grad(set_to_none=True)
    scaler.scale(losses['loss']).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for n,p in agent.named_parameters() if 'backbone' in n and p.grad is not None],max_norm=0.1)
    torch.nn.utils.clip_grad_norm_([p for n,p in agent.named_parameters() if 'backbone' not in n and p.grad is not None],max_norm=1.0)
    scaler.step(opt)
    scaler.update()
    scheduler.step()
    return {k:float(v.detach()) for k,v in losses.items()}


def lean_state(agent):
    # Frozen DINO is reconstructed from the already pinned pretrained model.
    # Keep ACT/ResNet, adapters, norms and all of their buffers unchanged.
    return {k:v for k,v in agent.state_dict().items() if not k.startswith(('task_encoder.backbone._backbone.','task_encoder._backbone.'))}


def load_weights(agent,path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    missing,unexpected=agent.load_state_dict(saved['agent_state_dict'],strict=False)
    assert not unexpected,unexpected
    assert all(k.startswith('task_encoder.backbone._backbone.') for k in missing),missing
    for n,p in agent.named_parameters():
        if p.requires_grad:
            assert n in saved['agent_state_dict'] and torch.equal(p.detach().cpu(),saved['agent_state_dict'][n]),n
    return {'checkpoint':str(path),'sha256':sha256(path),'epoch':saved['epoch'],'iteration':saved['iteration'],
            'optimizer_restored':False,'scheduler_restored':False,'verified_all_trainable_tensors':True}


def save(path,agent,opt,scheduler,scaler,args,epoch,iteration,experiment):
    ckpt={'agent_state_dict':lean_state(agent),'optimizer_state_dict':opt.state_dict(),
          'lr_scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),
          'args':vars(args),'epoch':epoch,'iteration':iteration,'best_eval_metrics':{},'experiment':experiment,
          'rng_state':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(ckpt,tmp)
    tmp.replace(path)
    write_json(path.with_suffix('.ready.json'),{'epoch':epoch,'iteration':iteration,'checkpoint':str(path),
               'sha256':sha256(path),'bytes':path.stat().st_size,'saved_at':time.time(),
               'frozen_dino_weights_omitted':True,'trainable_tensors':sum(p.requires_grad for p in agent.parameters())})


def verify_loss(agent,batch,out):
    # Compare the real forward result to its two independently exposed terms.
    rng_cpu=torch.get_rng_state()
    rng_cuda=torch.cuda.get_rng_state_all()
    agent.eval()
    torch.manual_seed(7001)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        values=agent.compute_loss(batch)
    torch.testing.assert_close(values['loss'],values['l1']+10*values['kl'])
    assert tuple(batch['robot_actions'].shape)==(len(batch['human_repo_id']),24,16)
    before={k:v.detach().cpu().clone() for k,v in lean_state(agent).items()}
    tmp=out/'roundtrip.pt'
    torch.save({'agent_state_dict':before},tmp)
    loaded=torch.load(tmp,map_location='cpu',weights_only=False)['agent_state_dict']
    assert set(loaded)==set(before)
    for k,v in loaded.items():assert torch.equal(v,before[k]),k
    tmp.unlink()  # Remove only this newly created verification artifact.
    write_json(out/'real-batch-check.json',{'status':'passed','losses':{k:float(v) for k,v in values.items()},
        'action_shape':list(batch['robot_actions'].shape),'robot_obs':{k:list(v.shape) for k,v in batch['robot_obs'].items()},
        'roundtrip_tensors':len(loaded),'human_repos':sorted(set(batch['human_repo_id']))})
    agent.train()
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_cuda)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',choices=['pretrain15','scratch5','finetune5'],required=True)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--benchmark-steps',type=int,default=0)
    ap.add_argument('--initialize',type=Path)
    cli=ap.parse_args()
    offline();os.chdir(SOURCE)
    args=build_args(cli.condition,cli.workers)
    out=ROOT/('calibration' if cli.benchmark_steps else 'runs')/(cli.condition+f'_w{cli.workers}' if cli.benchmark_steps else cli.condition)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    if (out/'train.jsonl').exists():raise RuntimeError(f'Partial run requires diagnosis before restarting: {out}')
    seed_all(args.seed)
    init=time.monotonic()
    dataset=build_dataset(args)
    init_loader=make_loader(dataset,args,0)
    agent=ACTAgent(args,torch.device('cuda'),dataloader=init_loader)
    del init_loader
    initialization=None
    if cli.condition=='finetune5':
        assert cli.initialize and cli.initialize.exists()
        initialization=load_weights(agent,cli.initialize)
    else:assert cli.initialize is None
    loader=make_loader(dataset,args,cli.workers)
    batches=len(loader)
    expected=json.loads((ROOT/'checks/data-boundaries.json').read_text())['pretrain_frames' if cli.condition=='pretrain15' else 'fewshot_frames']
    assert len(dataset)==expected,(len(dataset),expected)
    opt,scheduler,scaler=make_optimizer(agent,args,batches)
    write_json(out/'config.json',{'driver':vars(cli)|{'initialize':str(cli.initialize) if cli.initialize else None},
                 'args':vars(args),'frames':len(dataset),'batches_per_epoch':batches,'total_updates':args.total_epochs*batches,
                 'initialization':initialization,'normalization':'released per-repo metadata'})
    agent.train()
    torch.cuda.reset_peak_memory_stats()
    trace=(out/'train.jsonl').open('a',buffering=1)
    epoch_file=(out/'epochs.jsonl').open('a',buffering=1)
    iteration=0
    start=time.monotonic()
    timings=[]
    waits=[]
    checkpoint_epochs=json.loads((ROOT/'plan.json').read_text())['checkpoint_epochs']
    experiment={'condition':cli.condition,'data_boundaries_sha256':sha256(ROOT/'checks/data-boundaries.json'),
                'plan_sha256':sha256(ROOT/'plan.json'),'normalization':'released metadata unchanged','initialization':initialization}
    previous_end=time.monotonic()
    for epoch in range(args.total_epochs):
        epoch_start=time.monotonic()
        totals={'loss':0.,'l1':0.,'kl':0.}
        count=0
        for batch in loader:
            received=time.monotonic()
            wait=received-previous_end
            if iteration==0:verify_loss(agent,batch,out)
            values=update(agent,batch,opt,scheduler,scaler)
            now=time.monotonic()
            elapsed=now-previous_end
            iteration+=1;count+=1
            for key in totals:totals[key]+=values[key]
            if iteration>20:timings.append(elapsed);waits.append(wait)
            trace.write(json.dumps({'time':time.time(),'epoch':epoch+1,'iteration':iteration,'loss':values['loss'],
                       'l1':values['l1'],'kl':values['kl'],'lr':opt.param_groups[0]['lr'],'seconds':elapsed,'data_wait_seconds':wait})+'\n')
            if iteration%20==0:
                print('TRAIN',cli.condition,'epoch',epoch+1,'iter',iteration,'loss',round(values['loss'],5),'seconds/step',round(float(np.mean(timings[-50:])),4) if timings else 0,flush=True)
                set_status('calibration' if cli.benchmark_steps else 'training',status='running',condition=cli.condition,
                    trainer_pid=os.getpid(),epoch=epoch+1,iteration=iteration,total_epochs=args.total_epochs,
                    total_iterations=args.total_epochs*batches,steps_per_epoch=batches,
                    recent_seconds_per_step=float(np.mean(timings[-50:])) if timings else None)
            previous_end=now
            if cli.benchmark_steps and iteration>=cli.benchmark_steps:
                report={'status':'passed','workers':cli.workers,'steps':iteration,'batch':args.batch_size,'batches_per_epoch':batches,
                        'seconds_per_step':float(np.median(timings)),'mean_seconds_per_step':float(np.mean(timings)),
                        'data_wait_fraction':float(sum(waits)/sum(timings)),
                        'samples_per_second':args.batch_size/float(np.mean(timings)),
                        'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'initialization_seconds':start-init,
                        'estimated_training_hours':args.total_epochs*batches*float(np.mean(timings))/3600}
                write_json(out/'benchmark.json',report)
                save(out/'checkpoints/benchmark.pt',agent,opt,scheduler,scaler,args,0,iteration,experiment)
                print('BENCHMARK',json.dumps(report),flush=True)
                trace.close();epoch_file.close()
                return
        row={'epoch':epoch+1,'iteration':iteration,'seconds':time.monotonic()-epoch_start,
             **{k:v/count for k,v in totals.items()}}
        epoch_file.write(json.dumps(row)+'\n')
        save(out/'checkpoints/latest.pt',agent,opt,scheduler,scaler,args,epoch+1,iteration,experiment)
        if epoch+1 in checkpoint_epochs:
            save(out/f'checkpoints/epoch_{epoch+1:03d}.pt',agent,opt,scheduler,scaler,args,epoch+1,iteration,experiment)
        print('EPOCH COMPLETE',cli.condition,json.dumps(row),flush=True)
    final=out/'checkpoints/final_model.pt'
    save(final,agent,opt,scheduler,scaler,args,args.total_epochs,iteration,experiment)
    assert iteration==args.total_epochs*batches
    assert scheduler.last_epoch==iteration
    opt_steps=sorted({int(v['step']) for v in opt.state.values()})
    assert opt_steps==[iteration],opt_steps
    trace.close();epoch_file.close()
    write_json(out/'complete.json',{'status':'complete','condition':cli.condition,'checkpoint':str(final),'checkpoint_sha256':sha256(final),
              'epochs':args.total_epochs,'iterations':iteration,'optimizer_steps':opt_steps,'elapsed_seconds':time.monotonic()-init,
              'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'frames':len(dataset),'finished_at':time.time()})


if __name__=='__main__':main()
