"""Fixed-step outer driver, using the upstream ACT epoch-update semantics."""
from common import *
import argparse
import copy
import shutil
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from examples.baselines.act.train_act_imitator import ACTAgent, get_collate_fn
from diffusers.optimization import get_scheduler


class OrderedSampler(Sampler):
    def __init__(self, size, seed):
        self.size, self.seed, self.pass_index = size, seed, 0
    def order(self):
        g=torch.Generator().manual_seed(self.seed+1000003*self.pass_index)
        return torch.randperm(self.size,generator=g).tolist()
    def __iter__(self):
        return iter(self.order())
    def __len__(self):
        return self.size


def update(agent, batch, optimizer, scheduler, scaler):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=True):
        losses=agent.compute_loss(batch)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(losses['loss']).backward()
    scaler.unscale_(optimizer)
    back=[p for n,p in agent.named_parameters() if 'backbone' in n and p.grad is not None]
    other=[p for n,p in agent.named_parameters() if 'backbone' not in n and p.grad is not None]
    torch.nn.utils.clip_grad_norm_(back,max_norm=.1)
    torch.nn.utils.clip_grad_norm_(other,max_norm=1.)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return losses


def make_optimizer(agent,args,total_steps,warmup):
    back=[p for n,p in agent.named_parameters() if 'backbone' in n and p.requires_grad]
    other=[p for n,p in agent.named_parameters() if 'backbone' not in n and p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':other},{'params':back,'lr':args.lr_backbone}],lr=args.lr,weight_decay=1e-4)
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=warmup,num_training_steps=total_steps)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    return optimizer,scheduler,scaler


def equivalence(agent,batch,args,out):
    """Run the actual epoch-update source block and compare against our driver."""
    reference=copy.deepcopy(agent)
    aopt,asch,asca=make_optimizer(agent,args,18000,900)
    bopt,bsch,bsca=make_optimizer(reference,args,18000,900)
    # A non-zero LR makes this a parameter-update check, not just a forward test.
    for o in [aopt,bopt]:
        o.param_groups[0]['lr']=1e-4; o.param_groups[1]['lr']=1e-5
    import inspect, textwrap
    from examples.baselines.act import train_act_imitator as upstream
    source=inspect.getsource(upstream.train)
    start=source.index('                with torch.autocast("cuda", dtype=_amp_dtype, enabled=_use_amp):')
    end=source.index('\n                epoch_loss',start)
    block=textwrap.dedent(source[start:end])
    seed_all(741)
    ours=update(agent,batch,aopt,asch,asca)
    seed_all(741)
    ns={'torch':torch,'agent':reference,'batch':batch,'optimizer':bopt,'lr_scheduler':bsch,'scaler':bsca,'_amp_dtype':torch.bfloat16,'_use_amp':True}
    exec(compile(block,'upstream_epoch_update','exec'),ns)
    diffs={k:float((v-reference.state_dict()[k]).abs().max()) for k,v in agent.state_dict().items() if v.is_floating_point()}
    assert max(diffs.values())==0, sorted(diffs.items(),key=lambda x:-x[1])[:5]
    for k in ours:
        assert torch.equal(ours[k],ns['loss_dict'][k]),k
    fn=asch.lr_lambdas[0]
    assert fn(0)==0 and fn(900)==1 and fn(18000)==0
    write_json(out,{'status':'passed','actual_upstream_epoch_source_sha256':__import__('hashlib').sha256(block.encode()).hexdigest(),'state_tensor_count':len(diffs),'max_parameter_difference':max(diffs.values()),'losses':{k:float(v.detach()) for k,v in ours.items()},'scheduler_factors':{str(i):fn(i) for i in [0,1,900,9000,18000]}})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',choices=['A50','A10','D2'],default='A50')
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--steps',type=int,default=18000)
    ap.add_argument('--warmup',type=int,default=900)
    ap.add_argument('--run',required=True)
    ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--verify-update',action='store_true')
    ap.add_argument('--measure-warmup',type=int,default=50)
    ap.add_argument('--eval-envs',type=int,default=1)
    ap.add_argument('--evaluate',action='store_true')
    ap.add_argument('--async-eval',action='store_true')
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--benchmark-save',action='store_true')
    ap.add_argument('--keep-small-batch',action='store_true')
    cli=ap.parse_args()
    out=ROOT/cli.run
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError(f'Already complete: {out}')
    started=time.time()
    seed_all(cli.seed)
    cache=out/'te_cache'
    shutil.copytree(ROOT/'prepared/human_cache_ep0',cache,dirs_exist_ok=True)
    args=build_args(cli.condition,cli.seed,cli.batch,cli.workers,cache)
    args.total_iters=cli.steps
    # Saved for provenance; all scheduling is explicit in this driver.
    args.use_epoch_training=False
    write_json(out/'config.json',{'driver':vars(cli),'upstream_args':vars(args),'stats_sha256':sha256(Path(args.sim_root)/TASK/'meta/stats.json'),'human_cache_sha256':sha256(cache/'backbone_dinov2_vitl14/raw_features.pt')})
    dataset=build_dataset(args)
    sampler=OrderedSampler(len(dataset),cli.seed+10000)
    generator=torch.Generator().manual_seed(cli.seed+20000)
    def loader(workers):
        kw=dict(dataset=dataset,batch_size=cli.batch,sampler=sampler,drop_last=not cli.keep_small_batch,collate_fn=get_collate_fn(args.input_mode),num_workers=workers,pin_memory=True,generator=generator,worker_init_fn=seed_worker)
        if workers: kw.update(persistent_workers=True,prefetch_factor=4,multiprocessing_context='forkserver')
        return DataLoader(**kw)
    # No worker is started during cache-hit agent construction.
    init_loader=loader(0)
    seed_all(cli.seed)
    agent=ACTAgent(args,torch.device('cuda'),dataloader=init_loader)
    agent.train()
    del init_loader
    dl=loader(cli.workers)
    if cli.workers==0:
        for name in ['transform','rgb_transform']:
            trans=getattr(dataset.sim_dataset,name)
            if hasattr(trans,'set_random_seed'): trans.set_random_seed(cli.seed+30000)
    iterator=iter(dl)
    first=next(iterator)
    write_json(out/'batch_shapes.json',{'robot_obs':{k:list(v.shape) for k,v in first['robot_obs'].items()},'robot_actions':list(first['robot_actions'].shape),'first_frame':{k:list(v.shape) for k,v in first['robot_first_frame_obs'].items()}})
    if cli.verify_update:
        equivalence(agent,first,args,out/'update-equivalence.json')
        print('UPDATE EQUIVALENCE PASSED',flush=True)
        return
    optimizer,scheduler,scaler=make_optimizer(agent,args,cli.steps,cli.warmup)
    seed_all(cli.seed+40000)
    torch.cuda.reset_peak_memory_stats()
    trace=open(out/'train.jsonl','a',buffering=1)
    timings=[]; waits=[]; events=[]; aggregate={k:[] for k in ['loss','l1','kl']}
    updates=0; in_pass=0; completed_samples=0
    milestones=set(int(cli.steps*f) for f in [.1,.3,.5,.7,1.]) if not cli.benchmark else {cli.steps}
    train_start=time.perf_counter()
    initialization_seconds=time.time()-started
    measure_start=None
    step_start=train_start
    initial_hash=__import__('hashlib').sha256()
    for name,value in agent.state_dict().items(): initial_hash.update(value.detach().cpu().numpy().tobytes())
    write_json(out/'initialization.json',{'state_sha256':initial_hash.hexdigest(),'seed':cli.seed,'all_backbone_named_trainable':[n for n,p in agent.named_parameters() if p.requires_grad and 'backbone' in n]})
    pending_eval=None

    def finish_eval():
        nonlocal pending_eval
        if pending_eval is None: return
        process,log,eval_out=pending_eval
        code=process.wait(); log.close()
        if code: raise RuntimeError(f'Evaluation failed: {eval_out}; exit={code}')
        result=json.loads((eval_out/'result.json').read_text())
        print('DEVELOPMENT',json.dumps({'step':result['checkpoint_step'],'summary':result['summary'],'stages':result['stage_counts']}),flush=True)
        pending_eval=None

    def checkpoint(step):
        if shutil.disk_usage(ROOT).free < 32*1024**3:
            raise RuntimeError('Checkpoint storage reserve would be violated')
        dest=out/'checkpoints'/f'step_{step:06d}.pt'
        dest.parent.mkdir(exist_ok=True)
        payload={'iteration':step,'epoch':sampler.pass_index,'agent_state_dict':agent.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'lr_scheduler_state_dict':scheduler.state_dict(),'best_eval_metrics':{},'args':vars(args),
                 'experiment':{'condition':cli.condition,'seed':cli.seed,'steps':cli.steps,'batch_size':cli.batch,'samples':completed_samples,'stats_sha256':sha256(Path(args.sim_root)/TASK/'meta/stats.json')},
                 'sampling':{'pass_index':sampler.pass_index,'consumed_batches_in_pass':in_pass,'permutation':sampler.order(),'loader_generator_state':generator.get_state(),'worker_rng_exact_resume_supported':False},
                 'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
        temp=dest.with_suffix('.tmp')
        torch.save(payload,temp)
        with open(temp,'rb') as f: os.fsync(f.fileno())
        digest=sha256(temp)
        temp.replace(dest)
        # Checkpoint roundtrip checks all tensors and every active optimizer's step.
        restored=torch.load(dest,map_location='cpu',weights_only=False)
        assert restored['iteration']==step
        assert all(torch.equal(v.detach().cpu(),restored['agent_state_dict'][k]) for k,v in agent.state_dict().items())
        active_steps={int(v['step']) for v in restored['optimizer_state_dict']['state'].values()}
        assert active_steps=={step},active_steps
        assert scheduler.last_epoch==step
        write_json(dest.with_suffix('.ready.json'),{'path':str(dest),'sha256':digest,'bytes':dest.stat().st_size,'iteration':step,'roundtrip_equal':True,'optimizer_steps':sorted(active_steps)})
        return dest

    for step in range(1,cli.steps+1):
        if step==cli.measure_warmup+1:
            torch.cuda.synchronize(); measure_start=time.perf_counter(); step_start=measure_start
        before=time.perf_counter()
        if step==1: batch=first
        else:
            try: batch=next(iterator)
            except StopIteration:
                sampler.pass_index+=1; in_pass=0
                iterator=iter(dl); batch=next(iterator)
        data_wait=time.perf_counter()-before
        in_pass+=1
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        losses=update(agent,batch,optimizer,scheduler,scaler)
        end.record()
        # A single synchronization for numerical checks; upstream reads loss each step too.
        values={k:float(v.detach()) for k,v in losses.items()}
        if not all(np.isfinite(list(values.values()))): raise FloatingPointError(values)
        now=time.perf_counter()
        if step>cli.measure_warmup:
            timings.append(now-step_start); waits.append(data_wait); events.append((begin,end))
        step_start=now
        updates=step; completed_samples+=len(batch['robot_actions'])
        for k,v in values.items(): aggregate[k].append(v)
        if step%50==0 or step==1:
            record={'step':step,'samples':completed_samples,'pass':sampler.pass_index,'elapsed_train_s':now-train_start,'lr':[g['lr'] for g in optimizer.param_groups],**{k:float(np.mean(v)) for k,v in aggregate.items()},'memory_allocated_GiB':torch.cuda.memory_allocated()/1024**3}
            trace.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            aggregate={k:[] for k in aggregate}
        if step in milestones and (not cli.benchmark or cli.benchmark_save):
            path=checkpoint(step)
            if cli.evaluate:
                finish_eval()  # at most one pending evaluation per run
                eval_out=out/'dev'/f'step_{step:06d}'
                command=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',str(path),'--condition',cli.condition,'--out',str(eval_out),'--seed-start','1000','--episodes','20','--num-envs',str(cli.eval_envs)]
                log=open(out/f'eval_step_{step:06d}.log','w')
                pending_eval=(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT),log,eval_out)
                if not cli.async_eval or step==cli.steps or (cli.pilot and step in [int(cli.steps*.3),int(cli.steps*.5)]):
                    finish_eval()
                if cli.pilot and step==int(cli.steps*.5):
                    reviews=[json.loads((out/'dev'/f'step_{int(cli.steps*f):06d}'/'result.json').read_text()) for f in [.3,.5]]
                    if all(r['summary']['success_once']['mean']==0 for r in reviews):
                        gate=out/'pilot-review-decision.json'
                        write_json(out/'pilot-review-needed.json',{'status':'awaiting_local_analysis','step':step,'reviews':[{'step':r['checkpoint_step'],'stages':r['stage_counts'],'summary':r['summary']} for r in reviews]})
                        print('PILOT REVIEW NEEDED; waiting for local diagnostic decision',flush=True)
                        while not gate.exists(): time.sleep(5)
                        decision=json.loads(gate.read_text())
                        if decision['decision']!='continue':
                            write_json(out/'stopped.json',{'status':'stopped_at_pilot_gate','iteration':step,'decision':decision})
                            return
            # Saving/evaluation are excluded from steady step-time samples.
            step_start=time.perf_counter()
    torch.cuda.synchronize()
    finish_eval()
    wall=time.time()-started
    gpu_ms=[a.elapsed_time(b) for a,b in events]
    report={'status':'complete','condition':cli.condition,'seed':cli.seed,'steps':updates,'samples':completed_samples,'batch_size':cli.batch,'workers':cli.workers,'wall_seconds':wall,'initialization_seconds':initialization_seconds,'steady_steps':len(timings),'steady_step_median_s':float(np.median(timings)),'steady_step_p95_s':float(np.quantile(timings,.95)),'steady_mean_step_s':float(np.mean(timings)),'samples_per_second':cli.batch/float(np.mean(timings)),'data_wait_mean_s':float(np.mean(waits)),'gpu_update_ms_median':float(np.median(gpu_ms)),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3}
    trace.close()
    write_json(out/'complete.json',report)
    print('TRAIN COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':
    main()
