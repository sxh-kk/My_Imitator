"""Observe released ACT decoder gradients on a real batch, without updates."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/act_dinov2_15task'))
from common import *
import torch
from examples.baselines.act.train_act_imitator import ACTAgent
from train import load_weights


def main():
    seed_all(9010)
    args=build_args('finetune5',0)
    args.batch_size=2
    dataset=build_dataset(args)
    loader=make_loader(dataset,args,0)
    agent=ACTAgent(args,torch.device('cuda'),dataloader=loader)
    checkpoint=ROOT/'runs/finetune5/checkpoints/final_model.pt'
    load_weights(agent,checkpoint)
    batch=next(iter(loader))
    agent.train()
    agent.zero_grad(set_to_none=True)
    shapes={}
    def observe(name):
        def hook(module,inputs,output):
            if isinstance(output,torch.Tensor):shapes[name]=list(output.shape)
        return hook
    handles=[agent.model.transformer.decoder.register_forward_hook(observe('decoder_all_layers')),
             agent.model.action_head.register_forward_hook(observe('predicted_action'))]
    with torch.autocast('cuda',dtype=torch.bfloat16):losses=agent.compute_loss(batch)
    losses['loss'].backward()
    report={'status':'diagnosed','diagnostic':'real batch forward/backward only; no optimizer or checkpoint mutation',
            'checkpoint_sha256':sha256(checkpoint),'robot_actions_shape':list(batch['robot_actions'].shape),
            'shapes':shapes,'loss':float(losses['loss'].detach()),'decoder_layers':[]}
    for index,layer in enumerate(agent.model.transformer.decoder.layers):
        params=list(layer.parameters())
        grads=[p.grad for p in params if p.grad is not None]
        report['decoder_layers'].append({'index':index,'parameter_tensors':len(params),'tensors_with_gradient':len(grads),
            'nonzero_gradient_tensors':sum(bool(torch.count_nonzero(g)) for g in grads),
            'gradient_abs_sum':sum(float(g.detach().abs().sum()) for g in grads)})
    for handle in handles:handle.remove()
    write_json(ROOT/'checks/decoder-gradient-diagnosis.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
