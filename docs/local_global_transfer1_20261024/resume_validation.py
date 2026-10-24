"""User-authorized validation-only recovery from the existing 24 sealed systems."""
import json,os,time,signal,traceback,copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from foundation import write,save,prefix,build_backbone
from core import NativeUpper,affine_descriptor
from data import read_rows,UnnormalizedImages
from training import ARMS

def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=Path(cfg['output']);assert time.time()<cfg['hard_deadline_epoch']-60
    assert cfg['original_start_epoch']==1792820251.646123 and cfg['hard_deadline_epoch']==1792877851.646123
    assert (p/'FAILURE.json').exists() and not (p/'query.private.pt').exists() and not (p/'RESUME_VALIDATION_RECEIPT.json').exists()
    for task in range(6):assert not (p/f'scores_{task}.private.pt').exists()
    cost=json.loads((p/'COSTS.json').read_text());original=copy.deepcopy(cost);events=[]
    assert all(v['updates']==2000 for v in cost['arms'].values()) and all(v==6 for v in cost['ridge_solves'].values()) and all(v==0 for v in cost['score_rows'].values())
    assert cost['validation_encodes']==8
    immutable=[p/'FAILURE.json',p/'MAP_SEAL.json',p/'FIT_SEAL.json',p/'STAGE_SEALS.json',p/'QUERY_OPEN.private.json',p/'training_endpoint.private.pt',p/'training_proof.private.pt']+[p/f'stage_{i}.private.pt' for i in range(6)]+[p/f'map_{a}.private.pt' for a in ARMS[1:]]
    stamps={f.name:(f.stat().st_size,f.stat().st_mtime_ns) for f in immutable}
    write(p/'RESUME_COSTS_BEFORE.json',original)
    def record(value):
        events.append(value);write(p/'RESUME_NUMERIC_GATES.json',events)
        assert value['PASS']
    def status(phase,**extra):
        cost['elapsed_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=max(original['GPU_peak_reserved_bytes'],torch.cuda.max_memory_reserved());write(p/'COSTS.json',cost)
        write(p/'resume_status.private.json',dict(phase=phase,updated=time.time(),**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(SystemExit(143)))
    val=read_rows(cfg,'val');assert len(val)==300 and [r['id'] for r in val]==list(range(300))
    fit=json.loads((p/'FIT_SEAL.json').read_text());assert fit['models']==24
    write(p/'RESUME_QUERY_OPEN.private.json',dict(at=time.time(),prior_partial_validation_encodes=8,additional_optimizer_updates=0))
    backbone=build_backbone(cfg,'cuda');uppers={'NATIVE_RIDGE':NativeUpper(backbone).requires_grad_(False)}
    for a in ARMS[1:]:
        upper=NativeUpper(backbone);upper.load_state_dict(torch.load(p/f'map_{a}.private.pt',map_location='cpu',weights_only=False));upper.requires_grad_(False);upper.eval();uppers[a]=upper
    assert all(not v.requires_grad for u in uppers.values() for v in u.parameters())
    def describe(rows,uppers,field,keep_query=False):
            tables={a:torch.empty(len(rows),1537,dtype=torch.float64) for a in uppers};proof={};query_emb=[];query_lower=[]
            wanted=set()
            for c in sorted({r['label'] for r in rows}):
                candidates=sorted((i for i,r in enumerate(rows) if r['label']==c),key=lambda i:(rows[i]['group'],rows[i]['relative_path']))
                patients=set()
                for i in candidates:
                    if rows[i]['group'] not in patients:wanted.add(i);patients.add(rows[i]['group'])
                    if len(patients)==2:break
            with torch.no_grad():
                for images,ix in DataLoader(UnnormalizedImages(rows,cfg),batch_size=cfg['microbatch'],num_workers=0,shuffle=False):
                    raw=prefix(backbone,((images-.5)/.5).cuda().double());emb=raw.half();lower=backbone.blocks[:8](emb.double());stored=lower.half()
                    finite=bool(torch.isfinite(lower).all());record(dict(check='finite_canonical_lower',field=field,through=int(ix[-1])+1,PASS=finite));assert finite
                    cost[field]=cost.get(field,0)+len(ix)
                    for a,upper in uppers.items():
                        result=upper(stored.to(next(upper.parameters())));tables[a][ix]=affine_descriptor(result).cpu();cost['descriptor_rows'][a]+=len(ix)
                    if keep_query:query_emb.append(emb.cpu());query_lower.append(stored.cpu())
                    for j,i in enumerate(ix.tolist()):
                        if i in wanted:proof[rows[i]['id']]=dict(embedding=emb[j].cpu().clone(),lower=stored[j].cpu().clone(),lower_FP64=lower[j].cpu().clone(),label=rows[i]['label'],features={a:tables[a][i].clone() for a in uppers})
                    if int(ix[-1])%800<cfg['microbatch']:status(field,through=int(ix[-1])+1,total=len(rows))
            query=None if not keep_query else dict(embedding=torch.cat(query_emb),lower=torch.cat(query_lower),labels=torch.tensor([r['label'] for r in rows]))
            return tables,proof,query
    table,_,query=describe(val,uppers,'validation_encodes',True);save(p/'query.private.pt',query);save(p/'query_features.private.pt',table)
    cost['successful_validation_encodes']=len(val)
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);ix=torch.isin(query['labels'],torch.tensor(pack['seen']));scores={a:table[a][ix]@pack['heads'][a] for a in ARMS}
        save(p/f'scores_{task}.private.pt',scores)
        for a in ARMS:cost['score_rows'][a]+=int(ix.sum())
        status('resume_scored',task=task)
    assert cost['arms']==original['arms'] and cost['qualification_arms']==original['qualification_arms'] and cost['ridge_solves']==original['ridge_solves']
    assert stamps=={f.name:(f.stat().st_size,f.stat().st_mtime_ns) for f in immutable}
    write(p/'RESUME_VALIDATION_RECEIPT.json',dict(additional_optimizer_updates=0,training_counters_unchanged=True,sealed_files_unchanged=True,prior_partial_validation_encodes=8,successful_validation_encodes=300,total_validation_encodes=cost['validation_encodes'],at=time.time(),original_start_epoch=cfg['original_start_epoch'],hard_deadline_epoch=cfg['hard_deadline_epoch']))
    status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        write(Path(cfg['output'])/'RESUME_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
