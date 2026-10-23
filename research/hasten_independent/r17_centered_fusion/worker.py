"""Single fixed centering mechanism on H0 branch scores; no fitted calibration."""
import os,json,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

def centered(h,p):
    return .5*F.normalize(h-h.mean(1,keepdim=True),dim=1).softmax(1)+.5*F.normalize(p-p.mean(1,keepdim=True),dim=1).softmax(1)

def selfcheck():
    torch.manual_seed(31);h=torch.randn(11,7,dtype=torch.double);p=torch.randn_like(h)
    a=centered(h,p);b=centered(3*h+7,2*p-13)
    error=float((a-b).abs().max());assert error<1e-12
    zero=centered(torch.zeros_like(h),torch.zeros_like(p));assert torch.allclose(zero,torch.full_like(zero,1/7))
    assert torch.isfinite(a).all() and torch.allclose(a.sum(1),torch.ones(11,dtype=torch.double))
    return dict(passed=True,shift_positive_scale_invariance_max_error=error,uniform_tie_passed=True)

def run():
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(2)
    def read(p):return json.loads(p.read_text())
    def write(name,obj):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    write('SYNTHETIC_ACCEPTANCE.json',selfcheck())
    write('RULE_SEALED.json',dict(time=time.time(),center_before_L2=True,stat=.5,epsilon=1e-12,labels_for_fitting=0,no_ridge=True))
    all_results={}
    for ds in ['NIH','ISIC','HK']:
        source=Path(cfg['h0_sources'][ds]);split=read(source/'SPLIT_SEAL.json');agg=read(source/'aggregate.json');records=[]
        refs={(r['seed'],r['stage']):r for r in agg['results'] if r['arm']=='H-unit'}
        for seed in [74002,74003,74004]:
            for stage in range(len(split['stages'])):
                write('STATUS.json',dict(phase='EVALUATE_FIXED_FUSION',dataset=ds,seed=seed,stage=stage,time=time.time()))
                saved=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}_responses.private.pt',map_location='cpu',weights_only=False)
                h,p,old=[saved['scores'][k] for k in ['hyper','prototype','final']];y=saved['labels'];n=h.shape[1]
                replay=.5*(h/h.norm(dim=1,keepdim=True)).softmax(1)+.5*(p/p.norm(dim=1,keepdim=True)).softmax(1)
                error=float((replay-old).abs().max());assert error<1e-7 and torch.equal(replay.argmax(1),old.argmax(1))
                logits=centered(h,p);assert torch.isfinite(logits).all()
                count=torch.bincount(y,minlength=n);assert count.tolist()==split['eval_counts'][:n]
                pred=logits.argmax(1);correct=torch.bincount(y[pred==y],minlength=n);rec=correct.double()/count
                tail=[c for c in range(n) if split['train_counts'][c]<250]
                m=dict(BA=float(rec.mean()*100),tail_BA=float(rec[tail].mean()*100),counts=count.tolist(),correct=correct.tolist())
                newok=pred==y;oldok=old.argmax(1)==y
                rescue=torch.bincount(y[newok&~oldok],minlength=n).tolist();harm=torch.bincount(y[~newok&oldok],minlength=n).tolist()
                base=refs[seed,stage]['metrics']['final'];assert [a+b-c for a,b,c in zip(base['correct'],rescue,harm)]==m['correct']
                record=dict(seed=seed,stage=stage,seen=n,metrics=m,H0=base,rescue=rescue,harm=harm,H0_fusion_replay_max_error=error)
                records.append(record)
                folder=out/ds;folder.mkdir(exist_ok=True);torch.save(dict(ids=saved['ids'],labels=y,scores=logits),folder/f'{seed}_{stage}_responses.private.pt')
        all_results[ds]=dict(records=records,split=split);write('RESULTS.partial.json',all_results)
    write('RESULTS.json',all_results);write('STATUS.json',dict(phase='COMPLETE',optimizer_steps=0,new_fits=0,time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        p=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output']);(p/'FAILURE.private.txt').write_text(traceback.format_exc());raise
