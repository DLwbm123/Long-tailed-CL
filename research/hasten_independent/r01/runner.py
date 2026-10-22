"""Support-only, stage-gated development using pinned upstream training mechanics."""
import os,sys,json,time,random,copy,math,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from compat import load_upstream, install_feature_model, selfcheck, apply_delta_projection

NAMES=['dyed resection margins','hemorroids','ileum','ulcerative colitis grade 3','ulcerative colitis grade 2','bbps 0 1','bbps 2 3','retroflex stomach','retroflex rectum','oesophagitis b d','normal z line','oesophagitis a','ulcerative colitis grade 1 2']
PARENTS=['an anatomical landmark','a pathological finding','bowel preparation quality','a therapeutic intervention','hemorrhoids','ulcerative colitis','esophagitis']
TYPE=[3,1,0,1,1,2,2,0,0,1,0,1,1];SUB=[-1,0,-1,1,1,-1,-1,-1,-1,2,-1,2,1]
STAGES=[list(range(5)),list(range(5,9)),list(range(9,13))]
ARMS=['H-full','H-no-tree','H-shuffle','H-no-proj'];SEEDS=[74510,74511,74512]
CFG=json.loads(Path(os.environ['RUN_CONFIG']).read_text());OUT=Path(CFG['output']);OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def write(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def status(**kw):write('STATUS.json',dict(time=time.time(),**kw))
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def finite(x):assert torch.isfinite(x).all()

class FeatureDataset(Dataset):
    def __init__(self,x,y,ids):self.x=x;self.y=y;self.ids=ids;self.labels=np.array([int(y[i]) for i in ids])
    def __len__(self):return len(self.ids)
    def __getitem__(self,j):i=self.ids[j];return j,self.x[i],self.y[i]

class DataManager:
    def __init__(self,x,y,tr,te):self.x=x;self.y=y;self.tr=tr;self.te=te;self._class_to_label=NAMES;self._data_to_prompt=['a photo of a {}.']
    def get_task_size(self,k):return len(STAGES[k])
    def get_dataset(self,classes,source,mode):
        ids=self.tr if source=='train' else self.te
        return FeatureDataset(self.x,self.y,[i for i in ids if int(self.y[i]) in set(classes.tolist())])

class EncoderCache:
    def __init__(self,rows):self.rows=rows;self.model=None
    def ensure(self,stage):
        target=OUT/f'cache/stage_{stage}.private.pt'
        if target.exists():return torch.load(target,weights_only=False,map_location='cpu')
        import open_clip
        from PIL import Image
        if self.model is None:
            self.model,_,self.preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',precision='fp32',device='cuda',cache_dir=CFG['clip_cache'])
            self.model.eval().requires_grad_(False);self.tokenizer=open_clip.get_tokenizer('ViT-B-32')
        ids=[i for i,r in enumerate(self.rows) if r['label'] in STAGES[stage]];pieces=[]
        with torch.no_grad():
            for j in range(0,len(ids),32):
                batch=[]
                for i in ids[j:j+32]:
                    with Image.open(Path(CFG['image_root'])/self.rows[i]['relative_path']) as im:batch.append(self.preprocess(im.convert('RGB')))
                z=self.model.encode_image(torch.stack(batch).cuda()).float().cpu();finite(z);pieces.append(z)
            seen=sum(STAGES[:stage+1],[])
            # Only ancestors of currently arrived leaves become available.
            pn=set()
            for c in seen:
                pn.add(PARENTS[TYPE[c]])
                if SUB[c]>=0:pn.add(PARENTS[4+SUB[c]])
            prompts=['a photo of a '+NAMES[c]+'.' for c in seen]+['an endoscopic image of '+n+'.' for n in sorted(pn)]
            bank=self.model.encode_text(self.tokenizer(prompts).cuda()).float().cpu();finite(bank)
        rec=dict(ids=ids,x=torch.cat(pieces),prompts=prompts,text=bank,logit_scale=float(self.model.logit_scale),stage=stage)
        target.parent.mkdir(exist_ok=True);torch.save(rec,target)
        write(f'cache/stage_{stage}_receipt.json',dict(stage=stage,image_rows=len(ids),text_rows=len(prompts),raw_norm_range=[float(rec['x'].norm(dim=1).min()),float(rec['x'].norm(dim=1).max())],normalized=False,arrived_classes=seen))
        return rec
    def release(self):self.model=None;torch.cuda.empty_cache()

def hierarchy_files(seed,arm):
    root=OUT/f'trees/{seed}/{arm}';root.mkdir(parents=True,exist_ok=True)
    for stage in range(3):
        seen=sum(STAGES[:stage+1],[]);perm=seen[:]
        if arm=='H-shuffle':
            # Shuffle within arrival cohorts so prior attachments never change.
            perm=[]
            for k,part in enumerate(STAGES[:stage+1]):
                shift=random.Random(seed+100+k).randrange(1,len(part));perm+=part[shift:]+part[:shift]
        graph={}
        for c,attached in zip(seen,perm):
            parent=PARENTS[4+SUB[c]] if SUB[c]>=0 else PARENTS[TYPE[c]]
            graph.setdefault(parent,[]).append(NAMES[attached])
            if SUB[c]>=0:
                children=graph.setdefault(PARENTS[TYPE[c]],[])
                if parent not in children:children.append(parent)
        assert {v for vs in graph.values() for v in vs if v in NAMES}=={NAMES[c] for c in seen}
        (root/f'task_{stage}.json').write_text(json.dumps(graph))
    return root

def make_args(seed,arm):
    return dict(dataset='hk',device=[torch.device('cuda:0')],convnet_type='cached',init_cls=5,increment=4,
                tuned_epoch=10,init_lr=.001,batch_size=64,weight_decay=.001,min_lr=0.,optimizer='adam',stat=.5,sample_num=0,
                curvature=1.,hierarchy_weight=0. if arm=='H-no-tree' else .5,entail_weight=.1,reg_weight=.1,margin=.2,kcone=.1,mapping_mode='mlp',
                project_grad=arm!='H-no-proj',nullspace_energy_threshold=.95,hierarchy_dir=str(hierarchy_files(seed,arm)))

INC,UP,GEO=load_upstream(CFG['official'])

class CheckedLearner(UP.Learner):
    def start_audit(self):
        self.audit=dict(steps=0,replay=Counter(),loss_first=None,loss_last=None,loss_max=0.,max_projection_residual=0.,max_projection_delta=0.,gradient_components={},fixed_parent_max_change=0.)
        self.fixed_parents={}
        if self.hierarchy_manager is not None:
            hm=self.hierarchy_manager
            self.fixed_parents={v:hm.all_prototypes[hm.node_to_idx[v]].detach().clone() for v in hm.virtual_nodes if hm.has_proto[hm.node_to_idx[v]]}
        self.old_adapters={n:p.detach().clone() for n,p in self._network.named_parameters() if ('Adapter' in n)}
    def _compute_accuracy(self,*a,**kw):return 0. # No validation/model selection inside training.
    def _record_replay(self,classes):self.audit['replay'].update(classes)
    def _inspect_loss(self,total,contrast,anchor,hier):
        finite(total);a=self.audit;a['loss_first']=float(total.detach()) if a['loss_first'] is None else a['loss_first'];a['loss_last']=float(total.detach());a['loss_max']=max(a['loss_max'],float(total.detach()))
        hm=self.hierarchy_manager
        for name in hm.virtual_nodes:
            i=hm.node_to_idx[name]
            if hm.has_proto[i]:
                current=hm.all_prototypes[i].detach()
                if name in self.fixed_parents:
                    change=float((current-self.fixed_parents[name]).abs().max());a['fixed_parent_max_change']=max(a['fixed_parent_max_change'],change);assert change==0
                else:self.fixed_parents[name]=current.clone()
        if a['steps']==0:
            params=[p for p in self._network.parameters() if p.requires_grad]
            for name,val in [('contrast',contrast),('anchor',anchor),*[(k,v) for k,v in hier.items()]]:
                if val.requires_grad:
                    grads=torch.autograd.grad(val,params,allow_unused=True,retain_graph=True)
                    a['gradient_components'][name]=sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5
                else:a['gradient_components'][name]=0.
    def _before_step(self):
        self._step_before=self._network.learned_lorentz_exp.exp_map[0].weight.detach().clone()
        for p in self._network.parameters():
            if p.grad is not None:finite(p.grad)
    def _after_step(self):
        lle=self._network.learned_lorentz_exp;w=lle.exp_map[0].weight
        if self._cur_task>0 and self.args['project_grad']:
            r=apply_delta_projection(w,self._step_before,lle.null_space_basis)
            self.audit['max_projection_residual']=max(self.audit['max_projection_residual'],r['off_subspace_norm'])
            self.audit['max_projection_delta']=max(self.audit['max_projection_delta'],r['delta_norm'])
            assert r['off_subspace_norm']<=r['delta_norm']*.002+float(w.norm())*3e-6
        for n,p in self._network.named_parameters():
            finite(p)
            if n in self.old_adapters:assert torch.equal(p.detach(),self.old_adapters[n])
        self.audit['steps']+=1
        if self.audit['steps']%50==0:status(phase='TRAINING',arm=self.arm,seed=self.seed,stage=self._cur_task,steps=self.audit['steps'],loss=self.audit['loss_last'])

@torch.no_grad()
def predictions(learner,x):
    net=learner._network;net.eval();n=learner._total_classes
    text=net.model.encode_text(net.tokenizer(['a photo of a '+v+'.' for v in NAMES[:n]]).cuda())
    th=net.trans_hyper(net.Text_encode(text))
    hyp=[];proto=[];mix=[]
    for batch in x.split(128):
        b=batch.cuda();h=-GEO.pairwise_dist(net.trans_hyper(net.Image_encode(b)),th,net.get_curv())
        p=net.fc(torch.nn.functional.normalize(b,dim=-1))['logits'];m=learner.get_ens(h,p)
        finite(h);finite(p);finite(m);hyp.append(h.cpu());proto.append(p.cpu());mix.append(m.cpu())
    return dict(hyper=torch.cat(hyp),prototype=torch.cat(proto),final=torch.cat(mix))

def scores(logits,y,n,old):
    pred=logits.argmax(1);counts=torch.bincount(y,minlength=n);correct=torch.bincount(y[pred==y],minlength=n)
    rec=[None if counts[c]==0 else float(correct[c]/counts[c]) for c in range(n)]
    def avg(cs):
        vals=[rec[c] for c in cs if rec[c] is not None];return sum(vals)/len(vals) if vals else None
    result=dict(BA=avg(range(n)),old_BA=avg(range(old)),new_BA=avg(range(old,n)),tail_BA=avg([c for c in [1,2,12] if c<n]),counts=counts.tolist(),correct=correct.tolist(),recall=rec)
    if old:
        mask=y<old;p=logits[mask,:old].argmax(1);yy=y[mask];r=[]
        for c in range(old):
            cc=yy==c
            if cc.any():r.append(float((p[cc]==yy[cc]).double().mean()))
        result['old_candidates_BA']=sum(r)/len(r)
    return result

@torch.no_grad()
def drift(old,x,learner):
    xo,wo=old;xn=learner._network.Image_encode(x.cuda()).double();wn=learner._network.learned_lorentz_exp.exp_map[0].weight.double()
    a=xo@(wn-wo).T;b=(xn-xo)@wn.T;total=xn@wn.T-xo@wo.T
    return dict(mapping_RMS=float(a.square().mean().sqrt()),adapter_RMS=float(b.square().mean().sqrt()),total_RMS=float(total.square().mean().sqrt()),decomposition_max_error=float((total-a-b).abs().max()))

def checkpoint(learner,path):
    hm=learner.hierarchy_manager
    state=dict(network=learner._network.state_dict(),hierarchy=hm.state_dict(),prototype=learner.prototype,cov=learner.cov,task=learner._cur_task,total=learner._total_classes,known=learner._known_classes,accumulated_cov=learner._network.learned_lorentz_exp.accumulated_cov,null_space_basis=learner._network.learned_lorentz_exp.null_space_basis,n_total_samples=learner._network.learned_lorentz_exp.n_total_samples,
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(state,path)
    return torch.load(path,map_location='cuda',weights_only=False)

def restore_new(args,state,bank,prompts,scale):
    install_feature_model(INC,bank,prompts,scale);other=CheckedLearner(args)
    for k in range(state['task']+1):
        total=sum(len(v) for v in STAGES[:k+1]);other._network.update_fc(total);other._network.update_task()
    other._network.to('cuda');other._network.load_state_dict(state['network'])
    other._cur_task=state['task'];other._total_classes=state['total'];other._known_classes=state['known'];other.prototype=state['prototype'];other.cov=state['cov']
    hm=UP.HierarchyManager(torch.device('cuda'),other._network,args=args);hm.load_growth_hierarchy(str(Path(args['hierarchy_dir'])/f"task_{state['task']}.json"),NAMES[:state['total']]);hm.load_state_dict(state['hierarchy']);other.hierarchy_manager=hm
    lle=other._network.learned_lorentz_exp;lle.accumulated_cov=state['accumulated_cov'];lle.null_space_basis=state['null_space_basis'];lle.n_total_samples=state['n_total_samples']
    return other

def run():
    status(phase='SYNTHETIC_ACCEPTANCE');accept=selfcheck()
    # Exercise actual upstream virtual cache + gradient lifecycle, not a fixture implementation.
    seed_all(99);fixture=torch.randn(20,512);prompts=['a photo of a '+v+'.' for v in NAMES[:5]]+['an endoscopic image of '+v+'.' for v in PARENTS]
    install_feature_model(INC,fixture[:len(prompts)],prompts,2.);args=make_args(99,'H-full');probe=CheckedLearner(args)
    probe._network.update_task();probe._network.to('cuda');hm=UP.HierarchyManager(torch.device('cuda'),probe._network,args=args)
    hm.load_growth_hierarchy(str(Path(args['hierarchy_dir'])/'task_0.json'),NAMES[:5]);hm.set_task_classes(torch.arange(5,device='cuda'))
    leaf=probe._network.encode_text(probe._network.tokenizer(prompts[:5]).cuda());z=hm.get_all_prototypes(leaf,torch.arange(5,device='cuda'),probe._network.get_curv())
    parent_ids=[hm.node_to_idx[v] for v in hm.virtual_nodes];saved=z[parent_ids].detach().clone()
    losses=hm.compute_hierarchy_loss(z,probe._network.get_curv());sum(losses.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in probe._network.Text_Adapter[-1].parameters())
    with torch.no_grad():
        for p in probe._network.Text_Adapter[-1].parameters():p.add_(.001)
    leaf2=probe._network.encode_text(probe._network.tokenizer(prompts[:5]).cuda());z2=hm.get_all_prototypes(leaf2,torch.arange(5,device='cuda'),probe._network.get_curv())
    assert torch.equal(z2[parent_ids],saved) and not torch.equal(leaf2,leaf)
    accept.update(official_virtual_cache_fixed=True,current_leaf_gradient_nonzero=True,current_leaf_plastic=True)
    write('ACCEPTANCE_SYNTHETIC.json',accept);del probe,hm,z,z2,leaf,leaf2;torch.cuda.empty_cache()
    plan=json.loads(Path(CFG['support_plan']).read_text());rows=plan['rows'];tr=plan['folds'][0]['train'];te=plan['folds'][0]['test'];y=torch.tensor([r['label'] for r in rows]);x=torch.zeros(len(rows),512)
    assert len(rows)==3140 and not ({rows[i]['identity_component'] for i in tr}&{rows[i]['identity_component'] for i in te})
    public=dict(train_counts=torch.bincount(y[tr],minlength=13).tolist(),eval_counts=torch.bincount(y[te],minlength=13).tolist(),split='prior outer fold0 seed86420',patient_independence='UNKNOWN',stages=STAGES,query_rows=0)
    write('SPLIT_SEAL.json',public)
    encoder=EncoderCache(rows);all_results=[]
    for seed in SEEDS:
      for arm in ARMS:
        seed_all(seed);learner=None;args=make_args(seed,arm);trajectory=[]
        for stage in range(3):
            status(phase='STAGE_PREPARATION',seed=seed,arm=arm,stage=stage)
            cache=encoder.ensure(stage);x[cache['ids']]=cache['x'];encoder.release()
            bank,prompts,scale=cache['text'],cache['prompts'],cache['logit_scale']
            if learner is None:
                seed_all(seed);install_feature_model(INC,bank,prompts,scale);learner=CheckedLearner(args);learner.seed=seed;learner.arm=arm
            else:
                learner._network.model.text_bank=bank.cuda();learner._network.tokenizer=type(learner._network.tokenizer)(prompts)
            old=learner._known_classes;n=sum(len(v) for v in STAGES[:stage+1]);ids=[i for i in te if int(y[i])<n];oldids=[i for i in ids if int(y[i])<old]
            oldstate=None
            if old:
                with torch.no_grad():oldstate=(learner._network.Image_encode(x[oldids].cuda()).double(),learner._network.learned_lorentz_exp.exp_map[0].weight.detach().double().clone())
            # Reset deterministic stage RNG, matched across arms; encoding consumes no training RNG.
            seed_all(seed+stage*1000)
            learner.start_audit();dm=DataManager(x,y,tr,te)
            begin=time.time();learner.incremental_train(dm);wall=time.time()-begin
            # New tasks freeze the old adapter after start_audit: assert again against stage entry copies.
            finite(learner._network.learned_lorentz_exp.accumulated_cov)
            pred=predictions(learner,x[ids]);result=dict(seed=seed,arm=arm,stage=stage,old=old,seen=n,wall_seconds=wall,metrics={k:scores(v,y[ids],n,old) for k,v in pred.items()},audit=dict(learner.audit),projection_rank=int(learner._network.learned_lorentz_exp.null_space_basis.shape[1]))
            result['audit']['replay']=dict(result['audit']['replay'])
            if old:
                expected={c:learner.audit['steps'] for c in range(old)};assert dict(learner.audit['replay'])==expected
                result['drift']=drift(oldstate,x[oldids],learner)
            hm=learner.hierarchy_manager;virtual=[hm.node_to_idx[v] for v in hm.virtual_nodes];r=hm.all_prototypes[virtual].norm(dim=1)
            result['parent_radius_range']=[float(r.min()),float(r.max())];result['parent_cap_count']=int((r<=.2).sum())
            learner.after_task()
            folder=OUT/f'trajectories/{seed}/{arm}';folder.mkdir(parents=True,exist_ok=True)
            state=checkpoint(learner,folder/f'stage{stage}.private.pt');restored=restore_new(args,state,bank,prompts,scale)
            rp=predictions(restored,x[ids]);assert all(torch.equal(rp[k],pred[k]) for k in pred)
            result['checkpoint_restore_predictions_exact']=True
            # Exercise the restored hierarchy and projection state at the next task.
            learner=restored;learner.seed=seed;learner.arm=arm
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
            torch.save(dict(ids=ids,labels=y[ids],scores=pred),folder/f'stage{stage}_responses.private.pt')
            write(f'trajectories/{seed}/{arm}/stage{stage}.json',result);trajectory.append(result);all_results.append(result)
            write('aggregate_progress.json',all_results)
            if seed==74510 and arm=='H-full' and stage==1:
                write('TWO_STAGE_ACCEPTANCE.json',dict(passed=True,classification_not_a_gate=True,steps=sum(v['audit']['steps'] for v in trajectory),completed_stages=2,replay_complete=True,checkpoint_restore_exact=True,continue_to_three_stages=True))
        del learner;torch.cuda.empty_cache()
    write('aggregate.json',dict(results=all_results,phase='COMPLETE',stage_trainings=len(all_results),updates=sum(r['audit']['steps'] for r in all_results),query_rows=0,independent_confirmation=False))
    status(phase='COMPLETE',stage_trainings=len(all_results))

if __name__=='__main__':
    try:run()
    except BaseException:
        (OUT/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
