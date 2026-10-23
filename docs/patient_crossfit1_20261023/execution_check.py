"""Bounded engineering admission only; never launches scientific bulk."""
import copy, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import NativeEvidence, pools, representation_objective, np_evidence
from execution_core import CheckpointEvidence, graph_update
from run import build_backbone, prefix, evidence, body_state, save, write
from data import read_rows, Images

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
out = Path(cfg['output']); out.mkdir(exist_ok=False)
start = time.time()
cost = dict(prefix_rows=0, teacher_body_rows=0, parity_body_forward_rows=0, parity_body_backward_rows=0, discarded_updates=0, timed_arms={}, science_rows=0, validation_rows=0)
def receipt(name, value):
    write(out/name, value)
    write(out/'EXECUTION_COSTS.json', dict(**cost, at=time.time(), elapsed_from_original=time.time()-cfg['original_start_epoch']))
def parity(backbone, prompts, x, teacher, labels, weights, micro, scope):
    results = {}
    for arm in cfg['arms'][1:]:
        a = NativeEvidence(backbone, prompts); b = CheckpointEvidence(backbone, prompts)
        h = F.normalize(torch.randn(10, teacher.shape[1], device=teacher.device), dim=1)
        ha = torch.nn.Parameter(h.clone()); hb = torch.nn.Parameter(h.clone())
        pa=[a.prompts]+([ha] if arm=='CE_RC' else []); pb=[b.prompts]+([hb] if arm=='CE_RC' else [])
        oa=torch.optim.AdamW(pa,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']); ob=torch.optim.AdamW(pb,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        za=torch.cat([a(x[i:i+micro]) for i in range(0,len(x),micro)])
        loss,_,_=representation_objective(za,teacher,labels,weights,arm,cfg['ridge'],cfg['consistency_weight'],ha if arm=='CE_RC' else None)
        loss.backward();torch.nn.utils.clip_grad_norm_(pa,cfg['gradient_clip']); grads=[p.grad.detach().clone() for p in pa];oa.step()
        cc={}; val=graph_update(b,ob,x,teacher,labels,weights,arm,{**cfg,'microbatch':micro},cc,hb if arm=='CE_RC' else None)
        ge=max(float((g-p.grad).abs().max()) for g,p in zip(grads,pb)); ue=max(float((p-q).abs().max()) for p,q in zip(pa,pb))
        results[arm]=dict(gradient_error=ge,actual_AdamW_error=ue,objective_error=abs(float(loss.detach())-val['objective']),rows=len(x),body_microbatch=micro)
        cost['parity_body_forward_rows']+=2*len(x);cost['parity_body_backward_rows']+=2*len(x);cost['discarded_updates']+=2
        receipt(scope+'.json',results)
        assert ge<3e-4 and ue<2e-5
    return results

def main():
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from timm.models.vision_transformer import Block
    from functools import partial
    norm=partial(torch.nn.LayerNorm,eps=1e-6)
    toy=torch.nn.Module();toy.blocks=torch.nn.Sequential(*[Block(12,3,mlp_ratio=2,qkv_bias=True,norm_layer=norm) for _ in range(2)]);toy.norm=norm(12)
    for block in toy.blocks:block.attn.fused_attn=False
    x=torch.randn(8,5,12);t=F.normalize(torch.randn(8,24),dim=1);y=torch.tensor([0,1,2,3]*2);w=torch.full((8,),.25)
    parity(toy,torch.randn(2,2,12)*.02,x,t,y,w,2,'CPU_EQUIVALENCE')
    train=read_rows(cfg,'train');cs=cfg['order'][:10];base=[r for r in train if r['label'] in cs];pp=pools(base,cs,cfg['seed']);rows=[]
    for c in cs:
        for side in ['S','Q']:
            for g in sorted(pp[c][side])[:min(4,len(pp[c]['S']),len(pp[c]['Q']))]:rows.append(base[pp[c][side][g][0]])
    assert len(rows)==78
    b=build_backbone(cfg,'cuda'); tokens=[]
    for images,idx in DataLoader(Images(rows,cfg),batch_size=cfg['encoder_batch'],shuffle=False,num_workers=0):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):z=prefix(b,images.cuda())
        tokens.append(z.cpu().half());cost['prefix_rows']+=len(idx)
    tokens=torch.cat(tokens); teacher=evidence(NativeEvidence(b),tokens,8);cost['teacher_body_rows']=78
    prompts=torch.randn(8,6,768,device='cuda')*.02
    qp=pools(rows,cs,cfg['seed'])
    def fixture(classes):
        ix=[];yy=[];ww=[]
        for side in ['S','Q']:
            for c in classes:
                n=min(4,len(qp[c]['S']),len(qp[c]['Q']))
                for g in list(qp[c][side])[:n]:ix.append(qp[c][side][g][0]);yy.append(cs.index(c));ww.append(1/(8*n))
        return torch.tensor(ix),torch.tensor(yy),torch.tensor(ww)
    ordinary=[c for c in cs if c!=15][:8]
    ix,y,w=fixture(ordinary);small=torch.cat([torch.arange(4),torch.arange(len(ix)//2,len(ix)//2+4)])
    parity(b,prompts,tokens[ix[small]].cuda().float(),teacher[ix[small]].cuda(),y[small].cuda(),w[small].cuda(),2,'GPU_EQUIVALENCE')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(); times=[]
    for arm in cfg['arms'][1:]:
        model=CheckpointEvidence(b,prompts); head=torch.nn.Parameter(F.normalize(torch.randn(10,1536,device='cuda'),dim=1));opt=torch.optim.AdamW([model.prompts]+([head] if arm=='CE_RC' else []),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);cc={}
        torch.cuda.synchronize();tick=time.time()
        for classes in [ordinary,[15]+[c for c in cs if c!=15][:7]]:
            ix,y,w=fixture(classes);graph_update(model,opt,tokens[ix],teacher[ix],y,w,arm,cfg,cc,head if arm=='CE_RC' else None)
        torch.cuda.synchronize();times.append(time.time()-tick);cost['timed_arms'][arm]=cc;cost['discarded_updates']+=2
        receipt('GPU_RESOURCES.json',dict(peak_reserved_bytes=torch.cuda.max_memory_reserved(),limit=cfg['gpu_reserved_budget_bytes']))
        assert torch.cuda.max_memory_reserved()<cfg['gpu_reserved_budget_bytes']
        # Only changed execution state proof; prior full-class storage chain is reused explicitly.
        path=out/f'bounded_{arm}.private.pt';save(path,dict(prompts=model.prompts,bounded_tokens=tokens[:2],bounded_features=evidence(model,tokens[:2],2)))
        saved=torch.load(path,map_location='cpu',weights_only=False)
        cpu=NativeEvidence(copy.deepcopy(b).cpu().double(),saved['prompts']).double(); actual=evidence(cpu,saved['bounded_tokens'],2).numpy(); independent=np_evidence(saved['bounded_tokens'].numpy(),body_state(cpu.backbone),saved['prompts'].numpy())
        err=max(float(abs(actual-independent).max()),float(abs(actual-saved['bounded_features'].numpy()).max()))
        receipt(arm+'_RESPONSE.json',dict(error=err,FP64_NumPy_rows=2,FP32_saved_rows=2))
        assert err<cfg['FP32_probe_tolerance'];del cpu,model,opt,head
    step=sum(times)/2; remaining=cfg['hard_deadline_epoch']-time.time(); forecast=1.5*cfg['base_updates']*step+10800
    receipt('EXECUTION_FEASIBILITY.json',dict(state='PASS' if forecast<remaining-60 else 'FAIL',paired_step_seconds=step,forecast_seconds=forecast,remaining_seconds=remaining,original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch'],scientific_bulk_started=False,prior_full_class_storage_evidence='repair2/run/QUALIFICATION.json',new_evidence_scope='CPU/GPU all-three objective gradients and optimizer, real62/64 wholegraph, bounded full1536response state; no allclass rerun'))
    receipt('EXECUTION_COMPLETION.json',dict(state='PASS' if forecast<remaining-60 else 'BUDGET_GATE_FAILED',finished=time.time(),resident_seconds=time.time()-start,science_rows=0,validation_rows=0))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        receipt('EXECUTION_FAILURE.json',dict(type=type(exc).__name__,message=str(exc),peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None));raise
