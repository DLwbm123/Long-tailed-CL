"""Synthetic construction/native/report integration, with no real image access."""
import json,os,tempfile,time
from pathlib import Path
import torch
from . import evidence as E
from . import roots,construction,native,report


def run(output):
    start=time.time();torch.set_num_threads(1);torch.manual_seed(112)
    rows=[{'original_label':str(i//24),'identity_component':f'unit_{i}'} for i in range(23*24)]
    z=torch.nn.functional.normalize(torch.randn(len(rows),6,512),dim=-1)
    full=E.mem.from_views(z,torch.arange(len(rows)),torch.arange(len(rows))//24)
    cfg={'stages':[list(range(5)),list(range(5,10)),list(range(10,15)),list(range(15,19)),list(range(19,23))],'train_counts':[24]*23,'deadline':time.time()+3600,'setting':'synthetic'}
    previous=None;reference=None;records=[]
    for stage,cs in enumerate(cfg['stages']):
        current=full.take(torch.where(torch.isin(full.y,torch.tensor(cs)))[0])
        previous,reference,rec=construction.step(previous,reference,E.born(current,rows,stage),construction.reference_birth(current,rows,stage),cfg,stage,74003,rows)
        records.extend(rec);previous.validate();assert len(reference.bank)<=256
    assert len(records)==16 and all(r['full_oldtail_roles'] for r in records if r['required_oldtail_classes'])
    labels=torch.cat([torch.arange(20),torch.full((45-20,),20)])
    pool=native.fixture(z[:30],45,0,labels);future=[native.fixture(z[:30],8,10000+1000*h,torch.full((8,),21+h),h+1) for h in range(2)]
    m=roots.learner.Learner(112);m.arrive(pool.role(0),21);o=roots.learner.optimizer(m)
    m.forbidden={int(i) for e in [pool]+future for r in [1,2] for i in e.role(r).parents}
    candidates,ledger=roots.generate(pool,None,m,o,0,112,{},construction=True);counter={}
    run=roots.rollout(pool,future,m,o,None,candidates[0],31001,counter)
    assert [x['C'] for x in run['bytes']]==[21,22,23]
    scores=roots.score([run,run],set(range(7)))
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)
        for name in ['a','b']:
            out=p/name/'science';out.mkdir(parents=True);rs=[]
            for seed in [74003,74004]:
                for stage in [0,2,3,4]:
                    for index in range(4):
                        rs.append({'seed':seed,'stage':stage,'scenario':index,'metadata':{'real_oldtail_shadow':list(range(7)),'S_root_tokens':len(pool.role(0)),'S_capacity':214,'shadow_to_real':list(range(23))},
                                   'candidates':[{'family':'anchor','anchor':'E-ZERO'},{'family':'anchor','anchor':'E-BAL'}],'tapes':[scores,scores]})
            (out/'RESULTS.json').write_text(json.dumps({'stream':name,'roots':rs,'science_classifier_updates':0}))
        report.run(p);result=json.loads((p/'P1C_RESULTS.json').read_text());assert result['decision']=='C_no_consistent_local_advantage'
        assert len(result['blocks'])==16
    result={'passed':True,'synthetic_only':True,'checks':['five-stage new/reference construction','all 16 fixed slots retained','21-to-23 native shape','full 700-update three-horizon branch','four-rule report over 64 synthetic records'],'counts':counter,'real_encoder_rows':0,'real_classifier_updates':0,'wall_seconds':time.time()-start}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run(os.environ['Q_OUTPUT'])
