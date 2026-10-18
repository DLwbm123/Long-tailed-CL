"""One fixed penalty matching check, using only sealed sufficient statistics."""
import json,os,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from core_attribution import solve,features
from run import Images,write_json,save
from run_transfer import aggregate,paired
from run_fair_benchmark import bootstrap

def main(cfg):
    start=time.time();torch.set_num_threads(4);root=Path(cfg['output']);prior=Path(cfg['prior']);bc=json.loads(Path(cfg['benchmark_config']).read_text());source=json.loads(Path(bc['transfer_config']).read_text());testroot=Path(bc['output']);legacy=Path(bc['prior_rows']).resolve()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();assert not p.is_relative_to(legacy),'Support per-image inputs forbidden'
    sys.addaudithook(guard)
    write_json(root/'status.json',dict(state='RUNNING',phase='fit_from_sealed_statistics'))
    selection=json.loads((prior/'SELECTION.json').read_text());alpha=selection['HK']['arms']['N']['alpha'];assert alpha==.01
    norm=torch.load(prior/'HK.normalization.private.pt',weights_only=False);stats=torch.load(prior/'HK.joint_stats.private.pt',weights_only=True);oldheads=torch.load(prior/'HK.heads.private.pt',weights_only=True)
    heads={k:solve(*stats[k],alpha,15) for k in ['G0','G1']};save(root/'heads.private.pt',heads);del stats
    write_json(root/'MODEL_SEAL.json',dict(alpha=alpha,heads=2,old_per_image_inputs=False,normalization_refit=False,selection=False))
    rows=json.loads((testroot/'HK.test.rows.private.json').read_text());y=torch.tensor([r['label'] for r in rows]);groups=[r['identity_component'] for r in rows];cached=torch.load(testroot/'HK.test.features.private.pt',weights_only=True)
    assert torch.equal(cached['y'],y) and cached['components']==groups
    from safetensors.torch import load_file
    import timm
    torch.manual_seed(source['seed']);model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);model.load_state_dict({k:v for k,v in load_file(source['weight']).items() if not k.startswith('head.')},strict=True);model.eval().requires_grad_(False).cuda();h=[]
    extractstart=time.time()
    for x,idx in DataLoader(Images(rows,dict(images=bc['datasets']['HK']['images'])),batch_size=64,num_workers=0,shuffle=False):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):v=model.forward_features(x.cuda()).float()
        h.append(v[:,0].cpu().double());write_json(root/'status.json',dict(state='RUNNING',phase='extract_HK_CLS',processed=int(idx[-1])+1,total=len(rows)))
    raw={'g':F.normalize(cached['x'].double(),dim=1),'h':torch.cat(h)};z={k:(x-norm[k][0])/norm[k][1] for k,x in raw.items()};previous=torch.load(prior/'HK.predictions.private.pt',weights_only=False)
    assert torch.equal(previous['y'],y) and previous['groups']==groups;outputs={k:previous['scores'][k] for k in ['B','G0','G1','M']};parity={}
    for arm in ['G0','G1']:
        x=features(z,arm);s=x@oldheads[arm];err=float((s-outputs[arm]).abs().max());assert err<1e-5 and torch.equal(s.argmax(1),outputs[arm].argmax(1));parity[arm]=err;outputs[arm+'_matched']=x@heads[arm]
    write_json(root/'PARITY.json',parity);assert all(torch.isfinite(s).all() for s in outputs.values());metrics={k:aggregate(s,y,13) for k,s in outputs.items()};intervals={};pairs={}
    for a,b in [('G0_matched','G0'),('G1_matched','G1'),('G1_matched','G0_matched'),('M','G1_matched')]:
        key=a+'-'+b;intervals[key]=bootstrap({'B':outputs[b],'candidate':outputs[a]},y,groups,13,10000,74003)['arms']['candidate'];pairs[key]=paired(outputs[b],outputs[a],y,13)
    write_json(root/'results.json',dict(dataset='HK',alpha=alpha,metrics=metrics,intervals=intervals,paired=pairs));save(root/'predictions.private.pt',dict(scores=outputs,y=y,groups=groups));write_json(root/'costs.json',dict(wall_seconds=time.time()-start,CLS_extraction_and_scoring_seconds=time.time()-extractstart,peak_GPU_bytes=torch.cuda.max_memory_allocated(),prior_statistics_and_caches_excluded=True));write_json(root/'status.json',dict(state='COMPLETE',dataset='HK',new_heads=2,alpha=alpha,parameter_scan=False));print('COMPLETE',flush=True)
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.private.json',dict(error=traceback.format_exc()));write_json(Path(cfg['output'])/'status.json',dict(state='FAILED'));raise
