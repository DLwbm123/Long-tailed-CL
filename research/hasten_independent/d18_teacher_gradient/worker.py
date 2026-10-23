"""Restore fixed H0 TRAIN states and inspect one prospective auxiliary loss."""
import os,json,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from teacher import teacher_prob,class_balanced_kl,bounded_anchor,selfcheck
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ds=cfg['audit_dataset']
# Execute definitions and fixed initialization only, never the runner's __main__/run.
ns=dict(__name__='pinned_train_probe',__file__=cfg['base_runner'])
exec(compile(Path(cfg['base_runner']).read_text(),'<pinned-runner-definitions>','exec'),ns)
GEO=ns['GEO'];UP=ns['UP'];rows,tr,te=ns['read_rows']();assert len(tr)>0
seal=json.loads((Path(cfg['h0_source'])/'SPLIT_SEAL.json').read_text());yall=torch.tensor([r['label'] for r in rows]);stages=seal['stages']
assert torch.bincount(yall[tr]).tolist()==seal['train_counts']
def write(name,v):(out/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def flat(grads,params):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten() for g,p in zip(grads,params)])
report=[];write('SYNTHETIC_ACCEPTANCE.json',selfcheck())
for stage in [0,len(stages)-1]:
    n=sum(map(len,stages[:stage+1]));xs=[];ys=[];probe_ids=[];bank=None
    for s in range(stage+1):
        cache=Path(cfg['cache_root'] if ds=='NIH' else cfg['shared_cache'])/f'stage_{s}.private.pt'
        rec=torch.load(cache,map_location='cpu',weights_only=False)
        expected=[i for i in tr if rows[i]['label'] in stages[s]];assert rec['ids']==expected
        for c in stages[s]:
            pick=[j for j,i in enumerate(rec['ids']) if rows[i]['label']==c][:2];assert len(pick)==2
            xs.append(rec['x'][pick]);ys.extend([c]*2);probe_ids.extend(rec['ids'][j] for j in pick)
        if s==stage:bank=rec
    x=F.normalize(torch.cat(xs).cuda(),dim=1);y=torch.tensor(ys,device='cuda');assert set(probe_ids).issubset(tr) and not set(probe_ids)&set(te)
    bank['text']=F.normalize(bank['text'],dim=1)
    head=torch.load(Path(cfg['d15'])/ds/'heads'/f'Biomed_{stage}.private.pt',map_location='cpu',weights_only=False);assert head['classes']==n
    q=teacher_prob(x,head['ridge']);nonuniform=float((q-1/n).abs().max());assert nonuniform>1e-10
    for seed in [74002,74003,74004]:
        ns['status'](phase='GRADIENT_PROBE',seed=seed,stage=stage,validation_rows=0)
        state=torch.load(Path(cfg['h0_source'])/f'trajectories/{seed}/H-unit/stage{stage}.private.pt',map_location='cuda',weights_only=False)
        learner=ns['restore_new'](ns['make_args'](seed,'H-unit'),state,bank['text'],bank['prompts'],bank['logit_scale']);net=learner._network;net.eval()
        before={k:v.detach().clone() for k,v in net.state_dict().items()}
        text=net.model.encode_text(net.tokenizer([ns['LEAF_TEMPLATE'].format(v) for v in ns['NAMES'][:n]]).cuda())
        im=net.trans_hyper(net.Image_encode(x));tx=net.trans_hyper(net.Text_encode(text));curv=net.get_curv();scale=net.logit_scale.exp()
        logits=-GEO.pairwise_dist(im,tx,curv);pair=-GEO.pairwise_dist(im,tx[y],curv);tgt=torch.arange(len(y),device=y.device)
        contrast=.5*(F.cross_entropy(scale*pair,tgt)+F.cross_entropy(scale*pair.T,tgt))
        entail=UP.compute_stable_entailment_loss(tx[y],im,curv,min_radius=.1)
        protos=learner.hierarchy_manager.get_all_prototypes(tx,torch.arange(n,device=y.device),curv)
        hier=learner.hierarchy_manager.compute_hierarchy_loss(protos,curv,parent_child_margin=.2)
        with torch.no_grad():target=GEO.exp_map0(net.Image_encode(x),curv=curv)
        total=contrast+.1*entail+.5*sum(hier.values())+.1*(im-target.detach()).square().sum(-1).mean()
        kd=class_balanced_kl(scale*logits,q,y);anchor,audit=bounded_anchor(contrast,kd,im)
        named=[(name,p) for name,p in net.named_parameters() if p.requires_grad];params=[p for name,p in named]
        grads={key:torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) for key,loss in [('contrast',contrast),('original',total),('teacher',kd),('added',anchor)]}
        vectors={key:flat(gs,params) for key,gs in grads.items()}
        groups={}
        for group,indices in [('mapping',[i for i,(name,p) in enumerate(named) if 'learned_lorentz_exp' in name]),('adapters',[i for i,(name,p) in enumerate(named) if 'Adapter' in name])]:
            groups[group]={key:float(torch.cat([(torch.zeros_like(params[i]) if gs[i] is None else gs[i]).detach().double().flatten() for i in indices]).norm()) if indices else 0. for key,gs in grads.items()}
        cosine=float(F.cosine_similarity(vectors['original'],vectors['teacher'],dim=0))
        projected={}
        for key,gs in grads.items():
            parts=[]
            for (name,p),g in zip(named,gs):
                g=torch.zeros_like(p) if g is None else g
                if name=='learned_lorentz_exp.exp_map.0.weight' and stage>0:g=net.learned_lorentz_exp.project_gradient(g)
                parts.append(g.detach().double().flatten())
            projected[key]=float(torch.cat(parts).norm())
        for v in vectors.values():assert torch.isfinite(v).all()
        assert groups['mapping']['teacher']>0 and groups['adapters']['teacher']>0 and vectors['added'].norm()>0
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items()),'NETWORK_STATE_CHANGED'
        assert torch.cuda.max_memory_reserved()<8*2**30
        recall=lambda pred:float(torch.stack([(pred[y==c]==c).double().mean() for c in range(n)]).mean()*100)
        report.append(dict(seed=seed,stage=stage,classes=n,probe_rows=len(y),validation_rows=0,teacher_TRAIN_BA=recall(q.argmax(1)),student_TRAIN_BA=recall(logits.argmax(1)),teacher_nonuniform_max=nonuniform,teacher_entropy=float(-(q*q.clamp_min(1e-12).log()).sum(1).mean()),teacher_mean_top_probability=float(q.max(1).values.mean()),original_KD_parameter_cosine=cosine,parameter_gradient_norms={k:float(v.norm()) for k,v in vectors.items()},group_gradient_norms=groups,post_mapping_projection_norms=projected,network_unchanged=True,losses=dict(original=float(total.detach()),contrast=float(contrast.detach()),KD=float(kd.detach())),**audit))
        write('RESULTS.partial.json',report);del learner,net,state,before,grads,vectors;torch.cuda.empty_cache()
write('RESULTS.json',dict(dataset=ds,cells=report,passed=len(report)==6,optimizer_steps=0,validation_rows=0));ns['status'](phase='COMPLETE')
