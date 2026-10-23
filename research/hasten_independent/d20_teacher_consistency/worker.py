"""Fixed TRAIN label-consistency counts; no validation or model training."""
import ast,os,json,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from teacher import teacher_prob
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
def read(p):return json.loads(Path(p).read_text())
def write(f,v):(out/f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def summarize(x,y,head,n,source):
    q=teacher_prob(x,head);agree=q.argmax(1)==y
    eligible=agree&(q[torch.arange(len(y)),y]>1/n+1e-12)
    entropy=-(q*q.clamp_min(1e-12).log()).sum(1)
    return [dict(class_index=int(c),source=source,count=int((y==c).sum()),agree=int(agree[y==c].sum()),eligible=int(eligible[y==c].sum()),mean_label_probability=float(q[y==c,c].mean()),mean_entropy=float(entropy[y==c].mean())) for c in y.unique()]
try:
    results={}
    for di,ds in enumerate(['NIH','ISIC','HK']):
        spec=read(cfg['configs'][ds]);seal=read(Path(cfg['d15'])/ds/'SPLIT_SEAL.json');ns=dict(CFG=spec,ORDER=seal['original_order'],Path=Path,torch=torch)
        tree=ast.parse(Path(spec['base_runner']).read_text());tree.body=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='read_rows'];exec(compile(tree,'<pinned-reader>','exec'),ns)
        rows,tr,te=ns['read_rows']();stages=seal['stages'];records=[]
        for stage,classes in enumerate(stages):
            write('STATUS.json',dict(phase='TRAIN_CONSISTENCY_AUDIT',dataset=ds,stage=stage,time=time.time()))
            n=sum(map(len,stages[:stage+1]));old=n-len(classes)
            cache=Path(spec['cache_root'] if ds=='NIH' else spec['shared_cache'])/f'stage_{stage}.private.pt';rec=torch.load(cache,map_location='cpu',weights_only=False)
            ids=[i for i in tr if rows[i]['label'] in classes];assert rec['ids']==ids and not set(ids)&set(te)
            y=torch.tensor([rows[i]['label'] for i in ids]);head=torch.load(Path(cfg['d15'])/ds/'heads'/f'Biomed_{stage}.private.pt',map_location='cpu',weights_only=False);assert head['classes']==n
            current=summarize(F.normalize(rec['x'],dim=1),y,head['ridge'],n,'current_real_TRAIN')
            assert sum(a['count'] for a in current)==sum(seal['train_counts'][c] for c in classes)
            replay=[]
            if old:
                source=Path(cfg['h0_sources'][ds]);state=torch.load(source/f'trajectories/74002/H-unit/stage{stage}.private.pt',map_location='cpu',weights_only=False)
                rng=torch.Generator().manual_seed(20261023+100*di+stage)
                for c in range(old):
                    mean=state['prototype'][c].double();cov=state['cov'][c].double();chol=torch.linalg.cholesky(cov)
                    sample=(torch.randn(128,len(mean),generator=rng,dtype=torch.double)@chol.T+mean).float()
                    replay+=summarize(sample,torch.full((128,),c),head['ridge'],n,'Gaussian_generating_class')
                assert len(replay)==old and all(a['count']==128 for a in replay)
                del state
            records.append(dict(stage=stage,seen=n,current=current,replay=replay))
        allrows=[r for stage in records for r in stage['current']+stage['replay']]
        eligible=sum(r['eligible'] for r in allrows);total=sum(r['count'] for r in allrows)
        results[ds]=dict(stages=records,eligible=eligible,total=total,passed=0<eligible<total)
        write('RESULTS.partial.json',results)
    write('RESULTS.json',results);write('STATUS.json',dict(phase='COMPLETE',time=time.time(),optimizer_steps=0))
except BaseException:
    (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
