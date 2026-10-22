"""Support-only, stage-gated development using pinned upstream training mechanics."""
import os,sys,json,time,random,copy,math,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from compat import load_upstream, install_feature_model, selfcheck, apply_delta_projection, class_statistics

ORDER=[13,19,6,14,17,1,9,4,15,8,18,2,0,12,16,3,7,11,5,10]
ORIGINAL_NAMES=['atelectasis','cardiomegaly','consolidation','edema','effusion','emphysema','fibrosis','hernia','infiltration','mass','nodule','pleural thickening','pneumonia','pneumothorax','pneumoperitoneum','pneumomediastinum','subcutaneous emphysema','tortuous aorta','calcification of the aorta','no finding']
NAMES=[ORIGINAL_NAMES[c] for c in ORDER]
# Anatomical finding groups; no severity or causal inclusion claims.
PARENTS=['a pulmonary parenchymal finding','a pleural finding','a cardiovascular finding','an extrapulmonary finding','a chest radiographic observation']
ORIGINAL_TYPE=[0,2,0,0,1,0,0,3,0,0,0,1,0,1,3,3,3,2,2,4]
TYPE=[ORIGINAL_TYPE[c] for c in ORDER]
STAGES=[list(range(10))]+[list(range(i,i+2)) for i in range(10,20,2)]
ARMS=['H-full','H-no-proj'];SEEDS=[74002]
LEAF_TEMPLATE='a chest X-ray showing {}.'
PARENT_TEMPLATE='a chest X-ray showing {}.'
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
    def __init__(self,x,y,tr,te):self.x=x;self.y=y;self.tr=tr;self.te=te;self._class_to_label=NAMES;self._data_to_prompt=[LEAF_TEMPLATE]
    def get_task_size(self,k):return len(STAGES[k])
    def get_dataset(self,classes,source,mode):
        ids=self.tr if source=='train' else self.te
        return FeatureDataset(self.x,self.y,[i for i in ids if int(self.y[i]) in set(classes.tolist())])

class EncoderCache:
    def __init__(self,rows):self.rows=rows;self.model=None
    def load_model(self):
        if self.model is None:
            import open_clip,io,requests
            from open_clip.factory import _MODEL_CONFIGS
            assets=Path(CFG['biomed_assets']);spec=json.loads((assets/'open_clip_config.json').read_text());mc=copy.deepcopy(spec['model_cfg'])
            mc['text_cfg']['hf_model_name']=str(assets/'text');mc['text_cfg']['hf_tokenizer_name']=str(assets);mc['text_cfg']['hf_model_pretrained']=False
            _MODEL_CONFIGS['fixed_biomedical']=mc
            target=OUT/'biomed_checkpoint.private.bin'
            if not target.exists():
                status(phase='DOWNLOADING_FIXED_BIOMED_CHECKPOINT')
                proxy=os.environ['HTTPS_PROXY'];assert proxy.startswith('http://127.0.0.1:')
                session=requests.Session();session.trust_env=False;session.proxies={'http':proxy,'https':proxy}
                url='https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/9f341de24bfb00180f1b847274256e9b65a3a32e/open_clip_pytorch_model.bin'
                with session.get(url,stream=True,timeout=(30,120)) as response:
                    response.raise_for_status();buffer=io.BytesIO()
                    for chunk in response.iter_content(8*1024*1024):buffer.write(chunk)
                assert buffer.tell()==783705670
                tmp=target.with_suffix('.partial');tmp.write_bytes(buffer.getbuffer());tmp.replace(target);buffer.close()
            status(phase='LOADING_BIOMED_BACKBONE')
            self.model,_,self.preprocess=open_clip.create_model_and_transforms('fixed_biomedical',pretrained=None,pretrained_hf=False,precision='fp32',device='cuda',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()})
            state=torch.load(target,map_location='cpu',weights_only=False)
            if 'state_dict' in state:state=state['state_dict']
            key='text.transformer.embeddings.position_ids'
            if key in state:
                assert torch.equal(state[key],self.model.text.transformer.embeddings.position_ids.cpu());del state[key]
            self.model.load_state_dict(state,strict=True);del state
            self.model.eval().requires_grad_(False);self.tokenizer=open_clip.get_tokenizer('fixed_biomedical',context_length=256)
            write('BACKBONE_RECEIPT.json',dict(model='BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',revision='9f341de24bfb00180f1b847274256e9b65a3a32e',strict_learned_weight_load=True,pretrained_hf=False,dimension=512,preprocess=spec['preprocess_cfg'],open_clip_version=open_clip.__version__))
    @torch.no_grad()
    def encode(self,ids,phase):
        from PIL import Image
        self.load_model();pieces=[];begin=time.time()
        for j in range(0,len(ids),32):
            batch=[]
            for i in ids[j:j+32]:
                with Image.open(Path(CFG['image_root'])/self.rows[i]['relative_path']) as im:batch.append(self.preprocess(im.convert('RGB')))
            z=self.model.encode_image(torch.stack(batch).cuda()).float().cpu();finite(z);pieces.append(z)
            assert torch.cuda.max_memory_reserved()<8*2**30
            if j%1024==0:status(phase=phase,encoded=j+len(batch),total=len(ids),elapsed=time.time()-begin)
        return torch.cat(pieces)
    def ensure(self,stage):
        target=OUT/f'cache/stage_{stage}.private.pt'
        if target.exists():return torch.load(target,weights_only=False,map_location='cpu')
        ids=[i for i,r in enumerate(self.rows) if r['split']=='train' and r['label'] in STAGES[stage]]
        features=self.encode(ids,'ENCODE_TRAIN_STAGE_'+str(stage));seen=sum(STAGES[:stage+1],[])
        pn=sorted({PARENTS[TYPE[c]] for c in seen})
        prompts=[LEAF_TEMPLATE.format(NAMES[c]) for c in seen]+[PARENT_TEMPLATE.format(n) for n in pn]
        with torch.no_grad():bank=self.model.encode_text(self.tokenizer(prompts).cuda()).float().cpu();finite(bank)
        rec=dict(ids=ids,x=features,prompts=prompts,text=bank,logit_scale=float(self.model.logit_scale),stage=stage)
        target.parent.mkdir(exist_ok=True);torch.save(rec,target)
        write(f'cache/stage_{stage}_receipt.json',dict(stage=stage,training_rows=len(ids),validation_rows=0,text_rows=len(prompts),normalized=False,arrived_original_classes=[ORDER[c] for c in seen]))
        return rec
    def release(self):self.model=None;torch.cuda.empty_cache()

def hierarchy_files(seed,arm):
    root=OUT/f'trees/{seed}/{arm}';root.mkdir(parents=True,exist_ok=True)
    for stage in range(len(STAGES)):
        seen=sum(STAGES[:stage+1],[]);graph={}
        for c in seen:graph.setdefault(PARENTS[TYPE[c]],[]).append(NAMES[c])
        assert {v for vs in graph.values() for v in vs}=={NAMES[c] for c in seen}
        (root/f'task_{stage}.json').write_text(json.dumps(graph))
    return root

def make_args(seed,arm):
    return dict(dataset='nih',device=[torch.device('cuda:0')],convnet_type='cached',init_cls=10,increment=2,
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
    text=net.model.encode_text(net.tokenizer([LEAF_TEMPLATE.format(v) for v in NAMES[:n]]).cuda())
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
    result=dict(BA=avg(range(n)),old_BA=avg(range(old)),new_BA=avg(range(old,n)),tail_BA=avg([c for c in TAIL if c<n]),counts=counts.tolist(),correct=correct.tolist(),recall=rec)
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

def read_rows():
    import csv
    rows=[]
    for split in ['train','val']:
        with open(Path(CFG['manifest'])/(split+'.csv')) as f:part=list(csv.DictReader(f))
        with open(Path(CFG['author_labels'])/(split+'.csv')) as f:source={r['id']:r for r in csv.DictReader(f)}
        for r in part:
            name=Path(r['relative_path']).name;src=source[name];c=int(r['original_label'])
            assert r['split']==split and not Path(r['relative_path']).is_absolute() and '..' not in Path(r['relative_path']).parts
            columns=[k for k in src if k not in ['id','subject_id']]
            assert len(columns)==20 and int(src[columns[c]])==1 and sum(int(src[k]) for k in columns)==1
            r['group']=str(int(src['subject_id']));assert r['group']==str(int(name.split('_')[0]))
            r['label']=ORDER.index(c)
        assert len({r['relative_path'] for r in part})==len(part)
        rows+=part
    tr=[i for i,r in enumerate(rows) if r['split']=='train'];te=[i for i,r in enumerate(rows) if r['split']=='val']
    assert len(tr)==68058 and len(te)==300 and not ({rows[i]['group'] for i in tr}&{rows[i]['group'] for i in te})
    return rows,tr,te

def run():
    global TAIL
    status(phase='SYNTHETIC_ACCEPTANCE');write('ACCEPTANCE_SYNTHETIC.json',selfcheck())
    rows,tr,te=read_rows();y=torch.tensor([r['label'] for r in rows]);x=torch.zeros(len(rows),512)
    counts=torch.bincount(y[tr],minlength=20);TAIL=[c for c in range(20) if counts[c]<250]
    assert len(TAIL)==7 and torch.bincount(y[te],minlength=20).tolist()==[15]*20
    write('SPLIT_SEAL.json',dict(train_counts=counts.tolist(),eval_counts=[15]*20,original_order=ORDER,stages=STAGES,train_rows=len(tr),val_rows=len(te),tail_original_ids=[ORDER[c] for c in TAIL],patient_overlap=0,test_access=False))
    encoder=EncoderCache(rows);all_results=[]
    for seed in SEEDS:
      for arm in ARMS:
        seed_all(seed);learner=None;args=make_args(seed,arm)
        for stage in range(len(STAGES)):
            status(phase='STAGE_PREPARATION',seed=seed,arm=arm,stage=stage)
            cache=encoder.ensure(stage);x[cache['ids']]=cache['x'];encoder.release()
            bank,prompts,scale=cache['text'],cache['prompts'],cache['logit_scale']
            folder=OUT/f'trajectories/{seed}/{arm}'
            completed=folder/f'stage{stage}.json'
            if completed.exists():
                result=json.loads(completed.read_text());assert result['checkpoint_restore_predictions_exact']
                state=torch.load(folder/f'stage{stage}.private.pt',map_location='cuda',weights_only=False)
                restored=restore_new(args,state,bank,prompts,scale);learner=restored
                # Reconstruct only raw replay statistics in FP64 from sealed TRAIN caches.
                # No optimizer update or validation access; original checkpoint stays intact.
                errors=[]
                for c in range(state['total']):
                    ids=[i for i in tr if int(y[i])==c]
                    mean,cov=class_statistics(x[ids].cuda());torch.linalg.cholesky(cov)
                    errors.append(float((cov-state['cov'][c].double()).abs().max()))
                    assert torch.allclose(mean,learner.prototype[c],atol=2e-5,rtol=2e-5)
                    learner.cov[c]=cov
                write(f'trajectories/{seed}/{arm}/stage{stage}_resume_covariance.json',dict(no_retraining=True,validation_rows=0,covariance_dtype='float64',diagonal_regularizer=1e-5,max_abs_differences=errors))
                all_results.append(result)
                continue
            if learner is None:
                seed_all(seed);install_feature_model(INC,bank,prompts,scale);learner=CheckedLearner(args)
            else:
                learner._network.model.text_bank=bank.cuda();learner._network.tokenizer=type(learner._network.tokenizer)(prompts)
            learner.seed=seed;learner.arm=arm;old=learner._known_classes;n=sum(len(v) for v in STAGES[:stage+1])
            seed_all(seed+stage*1000);learner.start_audit()
            begin=time.time();learner.incremental_train(DataManager(x,y,tr,[]));wall=time.time()-begin
            finite(learner._network.learned_lorentz_exp.accumulated_cov)
            assert torch.cuda.max_memory_reserved()<8*2**30
            if old:assert dict(learner.audit['replay'])=={c:learner.audit['steps'] for c in range(old)}
            result=dict(seed=seed,arm=arm,stage=stage,old=old,seen=n,wall_seconds=wall,audit=dict(learner.audit),projection_rank=int(learner._network.learned_lorentz_exp.null_space_basis.shape[1]),validation_rows_read=0)
            result['audit']['replay']=dict(result['audit']['replay'])
            learner.after_task();folder=OUT/f'trajectories/{seed}/{arm}';folder.mkdir(parents=True,exist_ok=True)
            # Restore parity uses current training features, never held-out images.
            probe=x[cache['ids'][:32]];pred=predictions(learner,probe)
            state=checkpoint(learner,folder/f'stage{stage}.private.pt');restored=restore_new(args,state,bank,prompts,scale)
            rp=predictions(restored,probe);assert all(torch.equal(rp[k],pred[k]) for k in pred)
            result['checkpoint_restore_predictions_exact']=True
            learner=restored
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
            write(f'trajectories/{seed}/{arm}/stage{stage}.json',result);all_results.append(result);write('aggregate_progress.json',all_results)
            if arm=='H-full' and stage==1:write('TWO_STAGE_ACCEPTANCE.json',dict(passed=True,classification_not_a_gate=True,validation_rows=0,replay_complete=True,checkpoint_restore_exact=True))
        del learner,state,restored;torch.cuda.empty_cache()
    write('ALL_HEADS_SEALED.json',dict(stages=len(all_results),validation_rows_read=0,time=time.time()))
    x[te]=encoder.encode(te,'ENCODE_VAL_AFTER_ALL_HEADS_SEALED');encoder.release()
    for result in all_results:
        seed,arm,stage=result['seed'],result['arm'],result['stage'];n=result['seen'];old=result['old']
        cache=encoder.ensure(stage);args=make_args(seed,arm);folder=OUT/f'trajectories/{seed}/{arm}'
        state=torch.load(folder/f'stage{stage}.private.pt',map_location='cuda',weights_only=False)
        learner=restore_new(args,state,cache['text'],cache['prompts'],cache['logit_scale'])
        ids=[i for i in te if int(y[i])<n];pred=predictions(learner,x[ids])
        result['metrics']={k:scores(v,y[ids],n,old) for k,v in pred.items()}
        torch.save(dict(ids=ids,labels=y[ids],scores=pred),folder/f'stage{stage}_responses.private.pt')
        write(f'trajectories/{seed}/{arm}/stage{stage}.json',result)
        del learner;torch.cuda.empty_cache()
    write('aggregate.json',dict(results=all_results,phase='COMPLETE',stage_trainings=len(all_results),updates=sum(r['audit']['steps'] for r in all_results),independent_confirmation=False))
    report=['# NIH HASTEN数值比较','同一68058/300划分及10+2×5顺序；HASTEN为BiomedCLIP B16，旧参照为ImageNet ViT B16，训练与预处理不同。单seed74002开发结果，不是机制单因素对照。','| 路线 | 六阶段平均BA | 最终BA | 最终正确数 |','|---|---:|---:|---:|','| CLS＋patch | 34.0053% | 27.3333% | 82/300 |','| Spatial p3 | 33.2341% | 28.0000% | 84/300 |']
    for arm in ARMS:
        rr=[r for r in all_results if r['arm']==arm];m=[r['metrics']['final']['BA'] for r in rr]
        report.append(f"| {arm} | {100*sum(m)/6:.4f}% | {100*m[-1]:.4f}% | {sum(rr[-1]['metrics']['final']['correct'])}/300 |")
    (OUT/'COMPARISON_ZH.md').write_text('\n'.join(report)+'\n')
    status(phase='COMPLETE',stage_trainings=len(all_results))

if __name__=='__main__':
    try:run()
    except BaseException:
        (OUT/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
