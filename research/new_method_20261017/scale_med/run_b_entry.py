"""Decision-entry veto followed by an equal mean of two frozen fusions."""
import json,os,sys,time,traceback
from pathlib import Path
import torch
from run import write_json
from run_transfer import aggregate,paired
from run_b_consensus import retention


def entry_scores(b,density,distance):
    assert all(torch.isfinite(v).all() for v in [b,density,distance])
    bp=b.argmax(1);veto=(density.argmax(1)==bp)|(distance.argmax(1)==bp)
    score=(density+distance)/2
    # Vetoed rows retain B scores, including its deterministic argmax tie policy.
    score[veto]=b[veto]
    assert not ((score.argmax(1)!=bp)&veto).any()
    return score,veto


def main(cfg):
    torch.set_num_threads(4);start=time.time();root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.private.json',cfg);roots={k:Path(cfg[k]) for k in ['density','distance','consensus']}
    allowed={p/n for p in roots.values() for n in ['status.json','calibration.json','results.json']}
    allowed|={p/(n+'_'+q+'.scores.private.pt') for p in roots.values() for n in ['ISIC','HK'] for q in ['Qfit','Qgate']}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if any(p.is_relative_to(r) for r in roots.values()):assert p in allowed,'Input allowlist'
    sys.addaudithook(audit)
    assert cfg['local_lambda']==.5 and cfg['parameter_scan'] is False
    for p in roots.values():assert json.loads((p/'status.json').read_text())['state']=='COMPLETE'
    scale=json.loads((roots['density']/'calibration.json').read_text())
    assert all(json.loads((p/'calibration.json').read_text())==scale for p in roots.values())
    write_json(root/'calibration.json',scale);refs=json.loads((roots['consensus']/'results.json').read_text());records=[]
    for ref in refs:
        n,q,old=ref['dataset'],ref['query'],ref['old'];name=n+'_'+q+'.scores.private.pt'
        x,z,c=[torch.load(roots[k]/name,weights_only=True) for k in ['density','distance','consensus']]
        b,y=x['B'],x['labels'];assert torch.equal(b,z['B']) and torch.equal(b,c['B']) and torch.equal(y,z['labels']) and torch.equal(y,c['labels'])
        df,vf=x['fusion'],z['fusion'];t,veto=entry_scores(b/scale[n]['scales'][0],df,vf)
        outputs={'B':b,'density_fusion':df,'distance_fusion':vf,'consensus':c['scores'],'entry':t}
        metrics={k:aggregate(v,y,old) for k,v in outputs.items()}
        for k in ref['metrics']:assert metrics[k]==ref['metrics'][k],'Reference metrics changed'
        rr=retention(b,{'density_fusion':df,'distance_fusion':vf},t,y,old)
        bc=b.argmax(1)==y;cc=c['scores'].argmax(1)==y;tc=t.argmax(1)==y
        for group,m in [('all',torch.ones_like(y,dtype=torch.bool)),('old',y<old),('new',y>=old)]:
            saved=m&~bc&cc
            rr[group]['consensus']=dict(reference_rescues=int(saved.sum()),retained=int((saved&tc).sum()),lost=int((saved&~tc).sum()),additional_rescues=int((m&~bc&~cc&tc).sum()))
        classes=[]
        for k in range(t.shape[1]):
            m=y==k
            classes.append(dict(class_index=k,count=int(m.sum()),B_correct=int((m&bc).sum()),consensus_correct=int((m&cc).sum()),entry_correct=int((m&tc).sum()),rescue_B=int((m&~bc&tc).sum()),harm_B=int((m&bc&~tc).sum()),rescue_consensus=int((m&~cc&tc).sum()),harm_consensus=int((m&cc&~tc).sum())))
        record=dict(dataset=n,query=q,old=old,metrics=metrics,retention=rr,paired={k:paired(v,t,y,old) for k,v in outputs.items() if k!='entry'},classes=classes,gate=dict(veto=int(veto.sum()),eligible=int((~veto).sum()),eligible_retains_B=int((~veto&(t.argmax(1)==b.argmax(1))).sum())))
        if n=='ISIC' and q=='Qfit':
            # Audit only, after predictions: no labels or target identity enter the rule.
            target=(~bc)&((df.argmax(1)==y)|(vf.argmax(1)==y))&~cc&~veto
            assert int(target.sum())==2
            record['target_audit']=[dict(class_index=int(y[i]),consensus_correct=False,entry_correct=bool(tc[i])) for i in torch.where(target)[0].tolist()]
        rows=[]
        changed=t.argmax(1)!=c['scores'].argmax(1)
        for i in torch.where(changed)[0].tolist():
            top=t[i].topk(2).values
            rows.append(dict(row_index=i,label=int(y[i]),B_pred=int(b[i].argmax()),density_pred=int(df[i].argmax()),distance_pred=int(vf[i].argmax()),consensus_pred=int(c['scores'][i].argmax()),entry_pred=int(t[i].argmax()),entry_top2_gap=float(top[0]-top[1])))
        write_json(root/(n+'_'+q+'.changes.private.json'),rows)
        torch.save(dict(B=b,scores=t,labels=y),root/(n+'_'+q+'.scores.private.pt'));records.append(record)
    write_json(root/'results.json',records)
    write_json(root/'status.json',dict(state='COMPLETE',conditions=4,wall_seconds=time.time()-start,model_fits=0,image_forwards=0,scales_refitted=False,parameter_scan=False,test_access=False));print('COMPLETE',flush=True)


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.json',dict(error=traceback.format_exc()));raise
