"""Read sealed scores only; scientific comparison code inherited unchanged."""
import json,os,sys,time,resource,traceback
from pathlib import Path
import numpy as np
import torch
ARMS=['G','P','C']

def json_scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError('unsupported JSON value: '+type(x).__name__)

def write(p,x):
    encoded=json.dumps(x,indent=2,allow_nan=False,default=json_scalar)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(encoded);q.replace(p)

def metrics(s,y,groups,k):
    pred=s.argmax(1).numpy();yy=y.numpy();rec=[float((pred[yy==c]==c).mean()) for c in range(k)]
    grec=[]
    for c in range(k):
        gs=sorted({groups[i] for i in np.where(yy==c)[0]});grec.append(float(np.mean([np.mean(pred[[i for i in np.where(yy==c)[0] if groups[i]==g]]==c) for g in gs])))
    return dict(BA=sum(rec)/k,recall=rec,correct=[int(((pred==c)&(yy==c)).sum()) for c in range(k)],counts=[int((yy==c).sum()) for c in range(k)],group_equal_BA=sum(grec)/k,group_recall=grec)

def aggregate(scores,ty,gg,modelseal,out):
    summary={};harm=[]
    assert len(scores)==18 and len(ty)==len(gg) and set(ty.tolist())==set(range(6))
    for st in range(3):
        k=st+4;mask=ty<k;vgs=[gg[i] for i in torch.where(mask)[0].tolist()]
        for a in ARMS:
            for mode in ['ridge','CE']:
                s=scores[a+'_'+mode,st]
                assert s.device.type=='cpu' and s.shape==(len(ty),k) and bool(torch.isfinite(s).all())
                summary[a+'_'+mode,st]=metrics(s[mask],ty[mask],vgs,k)
            rp=scores[a+'_ridge',st][mask].argmax(1);cp=scores[a+'_CE',st][mask].argmax(1);truth=ty[mask]
            harm.append(dict(arm=a,stage=st,rescue=int(((rp!=truth)&(cp==truth)).sum()),harm=int(((rp==truth)&(cp!=truth)).sum())))
    summary_json={n:{str(s):summary[n,s] for s in range(3)} for n in [a+'_'+m for a in ARMS for m in ['ridge','CE']]}
    write(out/'RISK_TABLE.json',dict(summary=summary_json,rescue_harm=harm))
    def check(phase):pass
    names=[a+'_ridge' for a in ARMS]+[a+'_CE' for a in modelseal['complete_arms']];truth=ty.numpy();rng=np.random.default_rng(74150);rec=np.zeros((len(names),2,10000,6));check('paired_bootstrap')
    for c in range(6):
        ids=np.where(truth==c)[0];keys=sorted({gg[i] for i in ids});members=[[i for i in ids if gg[i]==g] for g in keys];sizes=np.array([len(v) for v in members]);draw=rng.integers(0,len(keys),(10000,len(keys)));den=sizes[draw].sum(1)
        for ni,name in enumerate(names):
            for si,st in enumerate([1,2]):
                if c>=st+4:continue
                ok=(scores[name,st].argmax(1).numpy()==truth);hits=np.array([ok[ix].sum() for ix in members]);rec[ni,si,:,c]=hits[draw].sum(1)/den
    jboot=(rec[:,0,:,:5].mean(-1)+rec[:,1].mean(-1))/2*100
    J={n:100*(summary[n,1]['BA']+summary[n,2]['BA'])/2 for n in names};comparisons={};gates={}
    for a in modelseal['complete_arms']:
        n=a+'_CE';diffs={}
        for r in ARMS:
            b=jboot[names.index(n)]-jboot[names.index(r+'_ridge')];diffs[r]=dict(J_gain_pp=J[n]-J[r+'_ridge'],CI95_pp=np.quantile(b,[.025,.975]).tolist(),CI9_adjusted_pp=np.quantile(b,[.05/18,1-.05/18]).tolist())
        own=a+'_ridge';stagegains=[100*(summary[n,s]['BA']-summary[own,s]['BA']) for s in [1,2]]
        birth=[100*(summary[n,1]['recall'][4]-summary[own,1]['recall'][4]),100*(summary[n,2]['recall'][5]-summary[own,2]['recall'][5]),100*(summary[n,2]['recall'][4]-summary[own,2]['recall'][4])]
        oldg=[100*np.mean(np.array(summary[n,s]['recall'][:4])-summary[own,s]['recall'][:4]) for s in [1,2]]+[100*np.mean(np.array(summary[n,2]['recall'][:5])-summary[own,2]['recall'][:5])]
        gate=dict(J_vs_G_at_least1pp=diffs['G']['J_gain_pp']>=1,J_vs_P_C_positive=all(diffs[r]['J_gain_pp']>0 for r in ['P','C']),all3_adjusted_lower_positive=all(z['CI9_adjusted_pp'][0]>0 for z in diffs.values()),both_stage_gain_nonnegative=min(stagegains)>=-1e-12,future_birth_and_retirement_nonnegative=min(birth)>=-1e-12,old_macro_cost_within1pp=min(oldg)>=-1-1e-12)
        comparisons[a]=dict(vs_ridge=diffs,stage_gain_own_pp=stagegains,class4_birth_class5_birth_class4_retired_gain_own_pp=birth,old_macro_gains_pp=oldg);gates[a]=dict(**gate,pass_all=all(gate.values()))
    passed=[a for a,v in gates.items() if v['pass_all']];decision='TARGET_HEADROOM_EXPLORATORY' if passed else ('STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION' if len(modelseal['complete_arms'])==3 else 'UNRESOLVED_OPTIMIZATION_NO_EXTENSION')
    result=dict(status='COMPLETE',summary=summary_json,rescue_harm=harm,J_pp=J,comparisons=comparisons,gates=gates,passing_arms=passed,decision=decision,method_success=False,bootstrap=dict(draws=10000,seed=74150,primary_family=9),historical_test_exposed=True,patient_isolation='UNKNOWN',new_parameter_search_after_base=False)
    write(out/'COMPARISONS.json',dict(comparisons=comparisons,gates=gates,decision=decision))
    write(out/'aggregate.json',result)
    return result

def guarded(action,out):
    out.mkdir(exist_ok=True,parents=True);start=time.monotonic();code=1
    try:
        result=action();code=0;return result
    except BaseException as error:
        write(out/'FAILURE.json',dict(error=type(error).__name__,decision='UNRESOLVED_ENGINEERING_NO_EXTENSION',no_retry=True))
        (out/'FAILURE.private.txt').write_text(traceback.format_exc());return None
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        write(out/'COST_RECEIPT.json',dict(wall_seconds=time.monotonic()-start,process_CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
        write(out/'EXIT.json',dict(exit_code=code))

def preflight(out):
    ty=torch.arange(6).repeat_interleave(4);gg=['s'+str(i//2) for i in range(24)];seal=dict(complete_arms=ARMS)
    def scores(perfect_ce):
        d={}
        for st in range(3):
            k=st+4
            for a in ARMS:
                for mode in ['ridge','CE']:
                    target=ty%k if (mode=='CE' and perfect_ce) else (ty+1)%k
                    d[a+'_'+mode,st]=torch.nn.functional.one_hot(target,k).double()
        return d
    checks={}
    for name,perfect in [('all_pass',True),('all_fail',False)]:
        sub=out/name;sub.mkdir();r=aggregate(scores(perfect),ty,gg,seal,sub)
        assert len(r['comparisons'])==3 and sum(len(x['vs_ridge']) for x in r['comparisons'].values())==9
        assert all(bool(g['pass_all'])==perfect for g in r['gates'].values())
        disk=json.loads((sub/'aggregate.json').read_text());assert all(type(g['old_macro_cost_within1pp']) is bool for g in disk['gates'].values())
        checks[name]=True
    f=out/'scalar.json';write(f,dict(b=np.bool_(True),i=np.int64(3),f=np.float64(.25)));assert json.loads(f.read_text())==dict(b=True,i=3,f=.25)
    for invalid in [dict(v=float('nan')),dict(v=object())]:
        old=f.read_text()
        try:write(f,invalid)
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid type was accepted')
        assert f.read_text()==old
    injected=out/'injected_failure'
    def fail():
        write(injected/'checkpoint.json',dict(saved=True));write(injected/'bad.json',dict(v=object()))
    assert guarded(fail,injected) is None
    assert json.loads((injected/'EXIT.json').read_text())['exit_code']==1
    assert json.loads((injected/'FAILURE.json').read_text())['error']=='TypeError'
    assert (injected/'COST_RECEIPT.json').exists() and json.loads((injected/'checkpoint.json').read_text())['saved']
    result=dict(state='PASSED',synthetic_only=True,complete_three_arm_pass_and_fail=True,nine_comparison_family=True,NumPy_scalar_roundtrip=True,NaN_and_unknown_types_rejected=True,atomic_failure_preserves_prior_file=True,failed_action_receipts_and_checkpoint_preserved=True,GPU_initialized=torch.cuda.is_initialized())
    assert not result['GPU_initialized'];write(out/'PRECHECK.json',result);return result

def restore(cfg,out):
    source=Path(cfg['input']);partial=json.loads((source/'partial.json').read_text());seal=json.loads((source/'MODEL_SEAL.json').read_text());selection=json.loads((source/'SELECTION_SEAL.json').read_text())
    assert seal['fit_units']==seal['converged_units']==15 and seal['test_used'] is False and seal['complete_arms']==ARMS
    assert partial['chosen']==seal['chosen']==selection['chosen']=={a:.0001 for a in ARMS}
    allowed={source/(n+'.json') for n in ['partial','MODEL_SEAL','SELECTION_SEAL','REUSE_RECEIPT','EXIT','COORDINATOR_EXIT']}
    allowed.add(source/'predictions.private.pt')
    allowed|={source/f"head_{f['arm']}_{f['stage']}_{f['lambda_value']}.private.pt" for f in partial['fits']}
    seen=set()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        f=Path(os.fsdecode(args[0])).resolve()
        if f==out or f.is_relative_to(out):return
        if f.is_relative_to(source):
            assert f in allowed,'not a sealed input';seen.add(f.name);return
        assert f.suffix.lower() not in ['.pt','.pth','.safetensors','.jpg','.png','.jpeg','.npy','.npz'],'unapproved data access'
    sys.addaudithook(guard)
    shapes={'G':1536,'P':1504,'C':1447};head_count=0
    for f in partial['fits']:
        assert f['status']=='CONVERGED' and f['gradient_max_abs']<=1e-7
        w=torch.load(source/f"head_{f['arm']}_{f['stage']}_{f['lambda_value']}.private.pt",map_location='cpu',weights_only=False)
        assert w.dtype==torch.float64 and w.shape==(shapes[f['arm']],f['stage']+4) and bool(torch.isfinite(w).all());head_count+=1;del w
    write(out/'INPUT_SEAL.json',dict(source_commit=cfg['input_commit'],sealed_heads_checked=head_count,chosen=partial['chosen'],no_fit=True,no_head_forward=True,no_feature_extraction=True,model_seal_before_test=seal['test_used'] is False))
    obj=torch.load(source/'predictions.private.pt',map_location='cpu',weights_only=False);ty=obj['y'];gg=obj['groups'];scores=obj['scores']
    assert len(ty)==586 and len(set(gg))==500 and int((ty==1).sum())==100 and len({gg[i] for i in torch.where(ty==1)[0].tolist()})==70
    r=aggregate(scores,ty,gg,seal,out)
    r.update(source_commit=cfg['source_commit'],prediction_source_commit=cfg['input_commit'],fits=partial['fits'],chosen=partial['chosen'],counts=partial['counts'],model_seal=seal,parity=partial['parity'],restoration_only=True,no_training=True,no_feature_extraction=True,no_new_predictions=True,source_run_exit=1,source_serialization_failure_preserved=True)
    expected={'G_ridge':53.044186046511626,'P_ridge':53.25271317829457,'C_ridge':53.48682170542635}
    assert all(abs(r['J_pp'][k]-v)<1e-5 for k,v in expected.items())
    assert not torch.cuda.is_initialized()
    write(out/'INPUT_VERIFICATION.json',dict(sealed_head_count=15,score_cells=18,test_rows=586,known_groups=500,class1_rows=100,class1_groups=70,all_scores_finite=True,read_only_sealed_assets=True,no_training=True,no_head_forward=True,no_feature_extraction=True,no_GPU=True,nine_comparisons=True,original_ridge_J_matches=True,only_source_assets_opened=True,opened_input_names=sorted(seen)))
    write(out/'aggregate.json',r);return r

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['cpu_budget'],cfg['cpu_budget']));torch.set_num_threads(4)
    result=guarded(lambda:preflight(out) if cfg['mode']=='PREFLIGHT' else restore(cfg,out),out)
    sys.exit(0 if result is not None else 1)
