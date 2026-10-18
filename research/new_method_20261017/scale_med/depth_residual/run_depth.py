"""Curator, stage-local fitting and sealed evaluation run in separate processes."""
import json,os,sys,time,resource,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from core_attribution import holdout
from core_depth import *

def write(path,obj):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False));p.replace(path)
def load(path):return torch.load(path,weights_only=False)
def rows(path):return json.loads(path.read_text())
def metrics(s,y,seen):
    p=s.argmax(1);recalls=[float((p[y==c]==c).double().mean()) for c in range(seen)]
    return dict(BA=sum(recalls)/seen,recall=recalls,counts=[int((y==c).sum()) for c in range(seen)],correct=[int(((y==c)&(p==c)).sum()) for c in range(seen)])

def main(cfg):
    start=time.time();torch.set_num_threads(4);root=Path(cfg['output']);phase=os.environ['PHASE']
    bc=rows(Path(cfg['benchmark_config']));src=rows(Path(bc['transfer_config']));prior=Path(bc['prior_rows']).resolve();testroot=Path(bc['output']).resolve();inputs=root/'inputs'
    active_bundle=None;allowed_images=set();retired=set();image_roots=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
    costs={};current_stage=-1
    def status(step,**kw):write(root/'status.json',dict(state='RUNNING',phase=phase,step=step,**kw))
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if phase=='fit':
            assert p not in retired,'Retired input accessed'
            if p.is_relative_to(prior) or p.is_relative_to(testroot):raise AssertionError('Learner cannot open historical caches/test')
            if p.is_relative_to(inputs):assert p==active_bundle,'Future or retired stage bundle accessed'
            if any(p.is_relative_to(d) for d in image_roots):assert p in allowed_images,'Image outside current stage'
        if phase=='evaluate':assert not p.is_relative_to(inputs) and not p.is_relative_to(prior),'Evaluation cannot read support'
    sys.addaudithook(guard);status('start')
    if phase=='prepare':
        audits={};bundles=[]
        for role in ['S_old','S_new']:
            rr=rows(prior/('ISIC.'+role+'.rows.private.json'));cache=load(prior/('ISIC.'+role+'.features.private.pt'))
            y=torch.tensor([r['label'] for r in rr]);groups=[r['identity_component'] for r in rr]
            assert torch.equal(cache['y'],y) and cache['components']==groups and cache['x'].shape==(len(y),1536)
            stageids=[0] if role=='S_old' else [1,2]
            for stage in stageids:
                mask=y<4 if stage==0 else y==stage+3;idx=torch.where(mask)[0].tolist();rs=[rr[i] for i in idx];yy=y[mask];gg=[groups[i] for i in idx]
                expected=list(range(4)) if stage==0 else [stage+3];assert sorted(set(yy.tolist()))==expected
                assert all(int((yy==c).sum())==src['support_counts']['ISIC'][c] for c in expected)
                valid,missing=holdout(yy,gg);assert not missing
                torch.save(dict(g=F.normalize(cache['x'][mask].double(),dim=1),y=yy,groups=gg,rows=rs,valid=valid),inputs/(str(stage)+'.private.pt'))
                bundles.append(rs);audits[str(stage)]=dict(images=len(rs),validation_images=int(valid.sum()),training_images=int((~valid).sum()),classes=expected,groups=len(set(gg)))
        tt=rows(testroot/'ISIC.test.rows.private.json');assert len(tt)==586;bundles.append(tt)
        for key in ['identity_component','relative_path']:
            sets=[{r[key] for r in rr} for rr in bundles];assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
        write(root/'DATA_AUDIT.json',dict(stages=audits,test_images=586,test_groups=len({r['identity_component'] for r in tt}),group_path_overlap=0,content_audit_reused=True,patient_isolation='UNKNOWN',test_historically_exposed=True))
        write(root/'BUDGET.json',dict(cap_bytes=BUDGET,width=WIDTH,final_bytes={a:planned(a,WIDTH[a]) for a in ARMS},precision='float64',head_count_in_budget=1,common_model_excluded=True))
    else:
        import timm
        from safetensors.torch import load_file
        from run import Images
        from torch.utils.data import DataLoader
        torch.manual_seed(74002);model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
        model.load_state_dict({k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')},strict=True);model.eval().requires_grad_(False).cuda()
        model_bytes=sum(p.numel()*p.element_size() for p in model.parameters());parity_done=False
        def block6(x):
            x=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
            for b in model.blocks[:6]:x=b(x)
            return x[:,0].float()
        def extract(rr,image_root):
            nonlocal parity_done
            tick=time.time();values=[]
            for x,idx in DataLoader(Images(rr,dict(images=image_root)),batch_size=64,num_workers=0,shuffle=False):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    v=block6(x.cuda())
                    if not parity_done:
                        captured=[];hook=model.blocks[5].register_forward_hook(lambda m,a,o:captured.append(o[:,0].float()))
                        model.forward_features(x.cuda());hook.remove();err=float((v-captured[0]).abs().max());assert err<1e-5
                        write(root/(phase+'.FEATURE_PARITY.json'),dict(block=6,samples=len(x),manual_hook_max_error=err));parity_done=True
                assert torch.isfinite(v).all();values.append(v.cpu().double());status('block6_extraction',stage=current_stage,processed=int(idx[-1])+1,total=len(rr))
            return torch.cat(values),time.time()-tick
        if phase=='fit':
            selfcheck();write(root/'SELFCHECK.json',dict(pass_all=True));state={};heads={};trans=None;alpha=None;retirement={};costs['stages']={}
            def fit_stage(stage):
                nonlocal trans,alpha
                tick=time.time();obj=load(active_bundle);g,y,groups,valid=obj['g'],obj['y'],obj['groups'],obj['valid'];tr=~valid;rr=obj['rows']
                assert not {groups[i] for i in torch.where(valid)[0]}&{groups[i] for i in torch.where(tr)[0]}
                h,extract_s=extract(rr,bc['datasets']['ISIC']['development_images']);trgroups=[groups[i] for i in torch.where(tr)[0].tolist()];learn_s=0;selection=None
                if stage==0:
                    t=time.time();trans,selection=transforms(g[tr],h[tr],y[tr],trgroups);learn_s=time.time()-t
                    torch.save(trans,root/'transforms.private.pt');write(root/'SELECTOR.json',selection)
                stageheads={};baseval={};budget={};reparam={};fit_s=0
                seen=4+stage
                for arm in ARMS:
                    x=mapped(g,h,trans[arm]);part=stats(x[tr],y[tr],trgroups,6)
                    if stage==0:state[arm]=part
                    else:state[arm]=(state[arm][0]+part[0],state[arm][1]+part[1])
                    t=time.time();stageheads[arm]={str(a):solve(state[arm][0],state[arm][1][:,:seen],a,seen) for a in ALPHAS};fit_s+=time.time()-t
                    if stage==0:baseval[arm]={a:ba(x[valid]@w,y[valid]) for a,w in stageheads[arm].items()}
                    if arm=='C':reparam={a:coordinate_check(g[valid],h[valid],trans[arm],w) for a,w in stageheads[arm].items()}
                    # Reserve final six-class H and one six-class deployed head at every stage.
                    expected=planned(arm,WIDTH[arm]);actual=nbytes(state[arm])+nbytes(trans[arm])+(x.shape[1]*6*8)+64
                    assert actual==expected and actual<=BUDGET,(arm,actual,expected)
                    budget[arm]=dict(reserved_six_class_state_bytes=actual,diagnostic_three_current_heads_bytes=nbytes(stageheads[arm]))
                if stage==0:
                    alpha=max(ALPHAS,key=lambda a:(baseval['G'][str(a)],a));write(root/'BASE_SELECTION.json',dict(common_alpha=alpha,selection_arm='G',all_arm_validation_BA=baseval))
                    write(root/'TRANSFORM_SEAL.json',dict(block=6,rank=128,common_alpha=alpha,all_transforms_fixed=True,sealed_at=time.time(),future_training_used=False,source_commit=cfg['source_commit']))
                heads[str(stage)]=stageheads;torch.save(stageheads,root/(str(stage)+'.heads.private.pt'));torch.save(state,root/(str(stage)+'.statistics.private.pt'))
                write(root/(str(stage)+'.REPARAMETERIZATION.json'),dict(max_abs_logit_errors=reparam,direct_penalty_equivalent=True))
                costs['stages'][str(stage)]=dict(wall_seconds=time.time()-tick,extraction_seconds=extract_s,transform_and_selection_seconds=learn_s,ridge_seconds=fit_s,budget=budget,feature_rows=len(y))
                return [Path(bc['datasets']['ISIC']['development_images'])/r['relative_path'] for r in rr]
            for stage in range(3):
                current_stage=stage;active_bundle=(inputs/(str(stage)+'.private.pt')).resolve()
                # Only paths from this stage are admitted before opening its images.
                stage_obj=load(active_bundle);allowed_images={Path(bc['datasets']['ISIC']['development_images'],r['relative_path']).resolve() for r in stage_obj['rows']};del stage_obj
                status('stage_fit',stage=stage);paths=fit_stage(stage)
                retired.update(p.resolve() for p in paths);retired.add(active_bundle);allowed_images.clear()
                rejected=False
                try:open(active_bundle,'rb')
                except AssertionError:rejected=True
                assert rejected;retirement[str(stage)]=dict(retired_input_guard_pass=True,retained='aggregate statistics and frozen transforms only');write(root/'RETIREMENT.json',retirement)
            write(root/'MODEL_SEAL.json',dict(stages=3,arms=5,penalties=ALPHAS,common_alpha=alpha,source_commit=cfg['source_commit'],all_old_inputs_retired=True,test_used=False,sealed_at=time.time()))
            costs['snapshot_heads_bytes']=nbytes(heads)
        elif phase=='evaluate':
            seal=rows(root/'MODEL_SEAL.json');assert seal['stages']==3 and (root/'TRANSFORM_SEAL.json').exists()
            trans=load(root/'transforms.private.pt');rr=rows(testroot/'ISIC.test.rows.private.json');cache=load(testroot/'ISIC.test.features.private.pt');y=torch.tensor([r['label'] for r in rr]);groups=[r['identity_component'] for r in rr]
            assert torch.equal(cache['y'],y) and cache['components']==groups
            g=F.normalize(cache['x'].double(),dim=1);h,extract_s=extract(rr,bc['datasets']['ISIC']['images']);outputs={};summary={};parity={}
            for stage in range(3):
                hh=load(root/(str(stage)+'.heads.private.pt'));seen=4+stage;mask=y<seen;outputs[str(stage)]={};summary[str(stage)]={}
                for arm in ARMS:
                    x=mapped(g,h,trans[arm]);outputs[str(stage)][arm]={a:x@w for a,w in hh[arm].items()}
                    summary[str(stage)][arm]={a:metrics(s[mask],y[mask],seen) for a,s in outputs[str(stage)][arm].items()}
                    if arm=='C':parity[str(stage)]={a:coordinate_check(g,h,trans[arm],w) for a,w in hh[arm].items()}
            torch.save(dict(scores=outputs,y=y,groups=groups),root/'predictions.private.pt');write(root/'TEST_REPARAMETERIZATION.json',parity)
            status('paired_bootstrap');import numpy as np
            correct=np.array([[[ (outputs[str(stage)][arm][str(a)].argmax(1)==y).numpy() for arm in ARMS] for a in ALPHAS] for stage in [1,2]])
            rng=np.random.default_rng(74003);sampled=np.zeros((2,3,5,10000,6));yy=y.numpy()
            for c in range(6):
                ids=np.where(yy==c)[0];keys=sorted({groups[i] for i in ids});members=[[i for i in ids if groups[i]==k] for k in keys];sizes=np.array([len(v) for v in members]);hits=np.stack([correct[:,:,:,m].sum(-1) for m in members],-1)
                ix=rng.integers(0,len(keys),size=(10000,len(keys)));den=sizes[ix].sum(1)
                for st in range(2):
                    for ai in range(3):
                        for ar in range(5):sampled[st,ai,ar,:,c]=hits[st,ai,ar][ix].sum(1)/den
            avg=(sampled[0,:,:,:,:5].mean(-1)+sampled[1].mean(-1))/2;ai=ALPHAS.index(seal['common_alpha']);ci=ARMS.index('C');contrasts={};main_points=[];main_samples=[];future=True;retention=True;sensitive=True
            def average(arm,a):return (summary['1'][arm][str(a)]['BA']+summary['2'][arm][str(a)]['BA'])/2
            for arm in ['G','R','P','S']:
                j=ARMS.index(arm);delta=100*(avg[:,ci]-avg[:,j]);point=[100*(average('C',a)-average(arm,a)) for a in ALPHAS]
                birth=100*(summary['1']['C'][str(seal['common_alpha'])]['recall'][4]-summary['1'][arm][str(seal['common_alpha'])]['recall'][4]);old=100*(summary['2']['C'][str(seal['common_alpha'])]['recall'][4]-summary['2'][arm][str(seal['common_alpha'])]['recall'][4]);new=100*(summary['2']['C'][str(seal['common_alpha'])]['recall'][5]-summary['2'][arm][str(seal['common_alpha'])]['recall'][5])
                future &= birth>0 and new>0;retention &= old>0 and old-birth>=-1;sensitive &= sum(v>0 for v in point)>=2 and point[ai]>0 and min(point)>=-1
                paired={}
                for stage in [1,2]:
                    pc=outputs[str(stage)]['C'][str(seal['common_alpha'])].argmax(1);pr=outputs[str(stage)][arm][str(seal['common_alpha'])].argmax(1)
                    paired[str(stage)]=[dict(class_index=c,rescue=int(((y==c)&(pr!=y)&(pc==y)).sum()),harm=int(((y==c)&(pr==y)&(pc!=y)).sum())) for c in range(4+stage)]
                class_intervals={}
                for name,st,c in [('class4_birth',0,4),('class4_retired',1,4),('class5_birth',1,5)]:
                    ds=100*(sampled[st,ai,ci,:,c]-sampled[st,ai,j,:,c]);class_intervals[name]=np.quantile(ds,[.025,.975]).tolist()
                main_points.append(point[ai]);main_samples.append(delta[ai]);contrasts[arm]=dict(mean_BA_gain_pp=point[ai],CI95_pp=np.quantile(delta[ai],[.025,.975]).tolist(),all_penalty_mean_gain_pp=dict(zip(map(str,ALPHAS),point)),class4_birth_gain_pp=birth,class4_retired_gain_pp=old,class5_birth_gain_pp=new,class4_advantage_change_pp=old-birth,class_gain_CI95_pp=class_intervals,paired_rescue_harm=paired)
            joint_lower=float(np.quantile(np.stack(main_samples).min(0),.025));primary=min(main_points)>=1 and joint_lower>0
            gates=dict(budget=True,primary_mean_BA=bool(primary),future_classes=bool(future),new_to_old=bool(retention),penalty_sensitivity=bool(sensitive))
            passed=all(gates.values());result=dict(dataset='ISIC',stages=[list(range(4)),list(range(5)),list(range(6))],common_alpha=seal['common_alpha'],metrics=summary,contrasts_C_minus_control=contrasts,minimum_control_gain_lower95_pp=joint_lower,gates=gates,decision='EXPLORATORY_PASS_NO_AUTOMATIC_EXTENSION' if passed else 'STOP_NO_EXTENSION',bootstrap=dict(draws=10000,seed=74003,paired_across_stages=True,unit='known_identity_group_within_class',patient_isolation='UNKNOWN'),test_historically_exposed=True)
            write(root/'results.json',result);costs['test_extraction_seconds']=extract_s;write(root/'status.json',dict(state='COMPLETE',decision=result['decision'],additional_experiments=False))
        else:raise ValueError(phase)
        costs['common_encoder_bytes']=model_bytes
    costs.update(wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,historical_global_cache_creation_excluded=True)
    write(root/(phase+'.costs.json'),costs);print('COMPLETE',phase,flush=True)

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write(Path(cfg['output'])/'failure.private.json',dict(phase=os.environ.get('PHASE'),traceback=traceback.format_exc()));write(Path(cfg['output'])/'status.json',dict(state='FAILED'));raise
