"""CPU statistics on trusted sealed data dictionaries; no model imports or forwards."""
import time,resource,os,json,csv,math,argparse
from pathlib import Path
BOOT_WALL=time.time();BOOT_CPU=time.process_time()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
torch.set_num_threads(2)

PATTERNS=[f'{i:03b}' for i in range(8)]
SETS={**{p:[i] for i,p in enumerate(PATTERNS)},'STRUCTURAL':[0,7],'INFORMATIVE':[1,2,3,4,5,6],'RESCUE':[1,2,3],'POTENTIAL_HARM':[4,5,6],'ALL':list(range(8))}
MODES=['T','CENTER','MULTI','UNIFORM','SUP_CF','RAW_EXACT_CF','GROUP_EXACT_CF','GRPO_CF','GRPO_IN']
READS={'tensor_files':0,'json_files':0,'saved_table_rows':0,'saved_VAL_query_rows':0,'historical_VAL_metadata_rows_loaded':0,'saved_VAL_readout_rows':0,'sampled_query_group_occurrences':0}

def js(p):
    READS['json_files']+=1;return json.loads(Path(p).read_text())
def pt(p):
    p=Path(p);assert p.name.endswith('.private.pt') and not any(x in p.name.lower() for x in ['checkpoint','teacher','optimizer','step','features','test'])
    READS['tensor_files']+=1;return torch.load(p,map_location='cpu',weights_only=True)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def csvout(p,rows):
    assert rows,p
    with Path(p).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def arr(t):return t.detach().numpy() if torch.is_tensor(t) else np.asarray(t)
def groups(c,s,y,C):
    births=np.array([b for b,cl in enumerate(c['stages'][:s+1]) for _ in cl]);old=births<s;tail=np.array(c['train_counts'][:C])<250
    return {'all':np.ones(len(y),bool),'old':old[y],'new':~old[y],'head':~tail[y],'tail':tail[y],'old_head':(old&~tail)[y],'old_tail':(old&tail)[y],'new_head':(~old&~tail)[y],'new_tail':(~old&tail)[y]}
def cb(v,y,mask,C):
    n=np.bincount(y[mask],minlength=C);valid=n>0
    return float((np.bincount(y[mask],weights=np.asarray(v,dtype=float)[mask],minlength=C)[valid]/n[valid]).mean()) if valid.any() else None
def micro(v,mask):return float(np.asarray(v,dtype=float)[mask].mean()) if mask.any() else None
def uniq(comps,mask):return len(set(comps[mask].tolist()))
def frac(n,d):return float(n/d) if d else None
def weighted_counts(mask,y,g,C):return int((mask&g).sum()),micro(mask,g),cb(mask,y,g,C)
def initial_zero(cc):
    success=cc@np.array([.8,.1,.1]);return success**8+(1-success)**8
def synthetic():
    cc=np.array([[int(x) for x in p] for p in PATTERNS]);z=initial_zero(cc);struct=np.isin(np.arange(8),[0,7]);assert np.all(z[struct]==1)
    for weights in [np.ones(8)/8,np.arange(1,9)/36]:
        zs=float(weights@struct);zp=float(weights@((~struct)*z));eff=float(weights@((~struct)*(1-z)));assert abs(zs+zp+eff-1)<1e-12
    assert len(PATTERNS)==len(set(PATTERNS))==8
    return {'passed':True,'scope':'new accounting identities only','optimizer_backward_model_encoder_image_VAL_scoring_TEST':0}

def run(source,out):
    source=Path(source);out=Path(out);out.mkdir(exist_ok=True)
    tables={k:[] for k in ['ASSET_AVAILABILITY','CANDIDATE_PATTERN_SUMMARY','CANDIDATE_PATTERN_CLASS','UNIFORM_VS_CANDIDATE_PATTERN','ZERO_GROUP_DECOMPOSITION','OOF_FULL_UTILITY_TRANSITIONS','OOF_FULL_ACTION_ORDER','OOF_FULL_CLASS_UTILITY','CLASS_GAIN_CONTRIBUTIONS','GAIN_CONCENTRATION','ROUTING_DECISION_QUALITY','PREDICTION_AGREEMENT']}
    unique={};stage_metrics=[];zero_covered=0;limits=[]
    write(out/'SYNTHETIC_CHECK.json',synthetic())
    for stream in ['a','b']:
        c=js(source/stream/'RUN_CONFIG.private.json');manifest=Path(c['manifest']);train=list(csv.DictReader((manifest/'train.csv').open()));val=list(csv.DictReader((manifest/'val.csv').open()));binding=js(c['source_binding'])
        import io
        assert train==list(csv.DictReader(io.StringIO(binding['train_manifest'])))
        sy=np.array([c['order'].index(int(v['original_label'])) for v in train]);vy=np.array([c['order'].index(int(v['original_label'])) for v in val]);A=set(binding['roles']['S']);B=set(binding['roles']['B']+binding['roles']['F']);assert not A&B and A|B==set(range(6821))
        for role in ['TRAIN_OOF','TRAIN_FULL','VAL_FULL']:
            unique[(stream,role)]={'images':set(),'components':set(),'informative_images':set(),'informative_components':set()}
        for s in range(5):
            C=sum(map(len,c['stages'][:s+1]));folder=source/stream/f's{s}';oo=pt(folder/'OOF.private.pt');fu=pt(folder/'IN.private.pt');vp=pt(source/stream/f'VAL_predictions_s{s}.private.pt');plan=arr(pt(folder/'QUERY_PLAN.private.pt'))
            ids=np.array(oo['ids']);assert np.array_equal(ids,fu['ids']) and np.array_equal(ids,np.flatnonzero(sy<C));y=sy[ids]
            vc=np.array(vp['ids']);assert np.array_equal(vc,np.flatnonzero(vy<C)) and np.array_equal(arr(vp['labels']),vy[vc])
            # Historical metadata only: do not substitute its predictions or teacher.
            oldmeta=pt(Path(c['reuse_root'])/f'VAL_predictions_s{s}.private.pt');READS['historical_VAL_metadata_rows_loaded']+=len(vc);assert oldmeta['indices']==vc.tolist() and oldmeta['components']==[val[i]['identity_component'] for i in vc]
            for role,data in [('TRAIN_OOF',oo),('TRAIN_FULL',fu)]:
                ss=arr(data['scores']);cc=arr(data['correct']).astype(bool);assert ss.shape==(len(ids),3,C) and np.array_equal(ss.argmax(-1)==y[:,None],cc)
            roles=[('TRAIN_OOF',oo,y,ids,train),('TRAIN_FULL',fu,y,ids,train),('VAL_FULL',vp,vy[vc],vc,val)]
            for role,data,yy,ii,meta in roles:
                isval=role=='VAL_FULL';pred=np.stack([arr(data['predictions'][a]) for a in ['T','CENTER','MULTI']],1) if isval else arr(data['scores']).argmax(-1)
                cc=pred==yy[:,None];assert np.array_equal(cc,arr(data['candidate_correct'] if isval else data['correct']))
                patterns=cc.astype(int)@np.array([4,2,1]);comps=np.array([meta[i]['identity_component'] for i in ii]);sample=np.array([meta[i]['sample_id'] for i in ii]);gg=groups(c,s,yy,C)
                READS['saved_table_rows']+=len(yy)
                if isval:READS['saved_VAL_query_rows']+=len(yy);READS['saved_VAL_readout_rows']+=9*len(yy)
                info=np.isin(patterns,SETS['INFORMATIVE']);u=unique[(stream,role)]
                for key,v in [('images',sample),('components',comps),('informative_images',sample[info]),('informative_components',comps[info])]:u[key].update(v.tolist())
                for field,precision,available,reason in [('stable_query_identity','endpoint_only',True,'index joined to frozen manifest; VAL components crosschecked historical saved metadata'),('candidate_logits','endpoint_only' if not isval else 'unavailable',not isval,'TRAIN cached scores; VAL candidate correctness/predictions only'),('candidate_correctness','endpoint_only',True,'joint row patterns directly available'),('learned_router_predictions_actions','endpoint_only' if isval else 'aggregate_only',isval,'VAL actions saved; TRAIN class aggregates only'),('learned_router_probabilities','unavailable',False,'weights and mean probabilities do not provide row outputs'),('rollout_query_plan','rollout_exact' if not isval else 'unavailable',not isval,'shared128x128 query plan; no individual sampled actions'),('rollout_old_probabilities_actions','unavailable',False,'not persisted by frozen source'),('identity_component','endpoint_only',True,'component metadata, not verified patients')]:
                    tables['ASSET_AVAILABILITY'].append(dict(stream=stream,stage=s,role=role,field=field,precision=precision,available=available,limitation=reason))
                if not isval:
                    bank=np.array(['FIT_B' if i in A else 'FIT_A' for i in ii])
                    assert all(i in (A if b=='FIT_B' else B) for i,b in zip(ii,bank))
                    # All queries are excluded from the named bank's component fit set.
                    for b,fit in [('FIT_A',A),('FIT_B',B)]:assert not set(comps[bank==b]) & {train[i]['identity_component'] for i in fit}
                for group,g in gg.items():
                    n=int(g.sum());Cg=len(set(yy[g].tolist()))
                    for name,codes in SETS.items():
                        mask=np.isin(patterns,codes);tables['CANDIDATE_PATTERN_SUMMARY'].append(dict(stream=stream,stage=s,role=role,group=group,pattern=name,query_denominator=n,class_denominator=Cg,count=int((g&mask).sum()),micro_fraction=micro(mask,g),class_balanced_fraction=cb(mask,yy,g,C),unique_components=uniq(comps,g&mask),total_unique_components=uniq(comps,g)))
                for cl in range(C):
                    g=yy==cl;n=int(g.sum())
                    for name,codes in SETS.items():
                        mask=g&np.isin(patterns,codes);tables['CANDIDATE_PATTERN_CLASS'].append(dict(stream=stream,stage=s,role=role,classifier_index=cl,legacy_index=c['legacy_indices'][cl],original_label=c['order'][cl],pattern=name,count=int(mask.sum()),class_query_denominator=n,class_fraction=frac(mask.sum(),n),global_class_balanced_contribution=frac(mask.sum(),C*n),unique_components=uniq(comps,mask)))
                if isval:uniform=arr(data['predictions']['UNIFORM']);provenance='saved_prediction_only_no_VAL_logits'
                else:
                    score=arr(data['scores']).astype(np.float64);ex=np.exp(score-score.max(-1,keepdims=True));p=ex/ex.sum(-1,keepdims=True);uniform=p.mean(1).argmax(-1);provenance='fixed_equal_probability_recompute_from_saved_logits'
                ugood=uniform==yy;oracle=cc.any(-1)
                for group,g in list(gg.items())+[(f'class_{cl}',yy==cl) for cl in range(C)]:
                    for pat in range(8):
                        mask=g&(patterns==pat);n=int(mask.sum());tables['UNIFORM_VS_CANDIDATE_PATTERN'].append(dict(stream=stream,stage=s,role=role,group=group,pattern=PATTERNS[pat],count=n,UNIFORM_correct=int((mask&ugood).sum()),UNIFORM_correct_fraction=micro(ugood,mask),UNIFORM_loses_existing_correct_candidate=int((mask&oracle&~ugood).sum()),UNIFORM_rescues_without_correct_candidate=int((mask&~oracle&ugood).sum()),UNIFORM_rescue_components=uniq(comps,mask&~oracle&ugood),source_precision=provenance))
                # The initial q is known exactly from protocol; not a trained endpoint.
                if not isval:
                    z=initial_zero(cc);struct=~info
                    for group,g in gg.items():
                        for weighting,mean in [('natural',lambda v:micro(v,g)),('class_balanced',lambda v:cb(v,yy,g,C))]:
                            zs=mean(struct);zp=mean(info*z);eff=mean(info*(1-z));inf=mean(info);total=zs+zp if zs is not None else None
                            tables['ZERO_GROUP_DECOMPOSITION'].append(dict(stream=stream,stage=s,role=role,group=group,arm='known_initial_qref',precision='known_initial_reference_expected_not_training_history',weighting=weighting,group_denominator=int(g.sum()),structural_fraction=zs,sampling_zero_fraction=zp,total_zero_fraction=total,effective_fraction=eff,informative_fraction=inf,conditional_zero_given_informative=frac(zp,inf) if zp is not None else None,structural_share_of_zero=frac(zs,total) if zs is not None else None,sampling_share_of_zero=frac(zp,total) if zp is not None else None,old_policy_expected_history_available=False))
            # Exact observed aggregate decomposition, no reconstruction of old policy.
            assert plan.shape==(128,128) and plan.min()>=0 and plan.max()<len(ids)
            for arm,data in [('GRPO_CF',oo),('GRPO_IN',fu)]:
                log=js(folder/f'{arm}_TRAIN_DIAGNOSTICS.json');records=log['records'];assert len(records)==512
                cc=arr(data['correct']).astype(bool);struct=cc.all(-1)|(~cc).all(-1);zs=0;zt=0
                for outer in range(128):
                    rr=[v for v in records if v['outer']==outer];assert sorted(v['inner'] for v in rr)==[0,1,2,3]
                    zero=[v['sampled_zero_group_fraction'] for v in rr];assert all(v==zero[0] for v in zero);nzero=zero[0]*128;assert abs(nzero-round(nzero))<1e-9
                    ns=int(struct[plan[outer]].sum());assert ns<=round(nzero)<=128;zs+=ns;zt+=round(nzero)
                N=128*128;zp=zt-zs;eff=N-zt;inf=N-zs;zero_covered+=N;READS['sampled_query_group_occurrences']+=N
                tables['ZERO_GROUP_DECOMPOSITION'].append(dict(stream=stream,stage=s,role='TRAIN_FULL' if arm=='GRPO_IN' else 'TRAIN_OOF',group='all',arm=arm,precision='derived_rollout_aggregate_exact_observed_not_old_policy_expected',weighting='actual_sampled_query_occurrences',group_denominator=N,structural_fraction=zs/N,sampling_zero_fraction=zp/N,total_zero_fraction=zt/N,effective_fraction=eff/N,informative_fraction=inf/N,conditional_zero_given_informative=frac(zp,inf),structural_share_of_zero=frac(zs,zt),sampling_share_of_zero=frac(zp,zt),old_policy_expected_history_available=False))
            tables['ASSET_AVAILABILITY'].append(dict(stream=stream,stage=s,role='TRAIN_OOF_and_FULL',field='observed_rollout_zero_group_totals',precision='derived_rollout_aggregate_exact',available=True,limitation='query plan plus128 exact outer aggregate zero counts;4inner deduplicated; no per-query zero identity or old-policy expectation'))
            # Pair relative utilities in identical TRAIN query rows.
            a=arr(oo['correct']).astype(int);b=arr(fu['correct']).astype(int);comps=np.array([train[i]['identity_component'] for i in ids]);gg=groups(c,s,y,C)
            for action,name in [(1,'CENTER'),(2,'MULTI')]:
                ra=a[:,action]-a[:,0];rb=b[:,action]-b[:,0]
                for group,g in gg.items():
                    for x in [-1,0,1]:
                        for z in [-1,0,1]:
                            mask=(ra==x)&(rb==z);kind='sign_reversal' if x*z==-1 else 'disappeared' if x!=0 and z==0 else 'appeared' if x==0 and z!=0 else 'unchanged'
                            n,mi,ma=weighted_counts(mask,y,g,C)
                            tables['OOF_FULL_UTILITY_TRANSITIONS'].append(dict(stream=stream,stage=s,action=name,group=group,r_OOF=x,r_FULL=z,change_type=kind,count=n,micro_fraction=mi,class_balanced_fraction=ma,teacher_wrong_to_correct=int((g&mask&(a[:,0]==0)&(b[:,0]==1)).sum()),teacher_correct_to_wrong=int((g&mask&(a[:,0]==1)&(b[:,0]==0)).sum()),candidate_wrong_to_correct=int((g&mask&(a[:,action]==0)&(b[:,action]==1)).sum()),candidate_correct_to_wrong=int((g&mask&(a[:,action]==1)&(b[:,action]==0)).sum()),unique_components=uniq(comps,g&mask)))
                for group,g in list(gg.items())+[(f'class_{cl}',y==cl) for cl in range(C)]:
                    for bank,cc in [('OOF',a),('FULL',b)]:
                        rescue=(cc[:,0]==0)&(cc[:,action]==1);harm=(cc[:,0]==1)&(cc[:,action]==0)
                        tables['OOF_FULL_CLASS_UTILITY'].append(dict(stream=stream,stage=s,action=name,group=group,bank=bank,count=int(g.sum()),rescue=int((g&rescue).sum()),harm=int((g&harm).sum()),net_correct=int((g&(cc[:,action]==1)).sum()-(g&(cc[:,0]==1)).sum()),micro_net_gain_pp=micro(cc[:,action]-cc[:,0],g)*100 if g.any() else None,class_macro_net_gain_pp=cb(cc[:,action]-cc[:,0],y,g,C)*100 if g.any() else None,teacher_correct=int((g&(cc[:,0]==1)).sum()),candidate_correct=int((g&(cc[:,action]==1)).sum()),rescue_components=uniq(comps,g&rescue),harm_components=uniq(comps,g&harm)))
            order_a=np.sign(a[:,1]-a[:,2]);order_b=np.sign(b[:,1]-b[:,2])
            for group,g in gg.items():
                for x in [-1,0,1]:
                    for z in [-1,0,1]:
                        mask=(order_a==x)&(order_b==z);n,mi,ma=weighted_counts(mask,y,g,C);tables['OOF_FULL_ACTION_ORDER'].append(dict(stream=stream,stage=s,group=group,CENTER_minus_MULTI_order_OOF=x,CENTER_minus_MULTI_order_FULL=z,count=n,micro_fraction=mi,class_balanced_fraction=ma,unique_components=uniq(comps,g&mask)))
            # All90 saved readouts, original class-denominator contributions and routes.
            yy=arr(vp['labels']);den=np.bincount(yy,minlength=C);tail=np.array(c['train_counts'][:C])<250;birth=np.array([b for b,cl in enumerate(c['stages'][:s+1]) for _ in cl]);old=birth<s;gg=groups(c,s,yy,C);comps=np.array([val[i]['identity_component'] for i in vc]);tp=arr(vp['predictions']['T']);goodT=tp==yy;candidate=np.stack([arr(vp['predictions'][x]) for x in ['T','CENTER','MULTI']],1);cc=candidate==yy[:,None]
            for mode in MODES:
                pred=arr(vp['predictions'][mode]);good=pred==yy;correct=np.bincount(yy[good],minlength=C);rec=correct/den;contrib=100/C*(correct-np.bincount(yy[goodT],minlength=C))/den
                stage_metrics.append(dict(stream=stream,stage=s,mode=mode,micro=100*good.mean(),macro=100*rec.mean(),tail=100*rec[tail].mean() if tail.any() else None,old_tail=100*rec[old&tail].mean() if (old&tail).any() else None))
                for cl in range(C):
                    g=yy==cl;h=g&goodT&~good;re=g&~goodT&good;tcorrect=int((g&goodT).sum());tables['CLASS_GAIN_CONTRIBUTIONS'].append(dict(stream=stream,stage=s,mode=mode,classifier_index=cl,legacy_index=c['legacy_indices'][cl],original_label=c['order'][cl],tiny=cl in c['tiny_indices'],old=bool(old[cl]),tail=bool(tail[cl]),count=int(den[cl]),teacher_correct=tcorrect,method_correct=int(correct[cl]),teacher_recall=100*tcorrect/den[cl],method_recall=100*rec[cl],net_correct=int(correct[cl])-tcorrect,rescue=int(re.sum()),harm=int(h.sum()),rescue_components=uniq(comps,re),harm_components=uniq(comps,h),class_components=uniq(comps,g),macro_contribution_pp=float(contrib[cl]),tail_contribution_pp=float((correct[cl]-tcorrect)/den[cl]*100/tail.sum()) if tail[cl] else (0. if tail.any() else None),old_tail_contribution_pp=float((correct[cl]-tcorrect)/den[cl]*100/(old&tail).sum()) if old[cl]&tail[cl] else (0. if (old&tail).any() else None)))
                positive=np.maximum(contrib,0);negative=np.minimum(contrib,0);top=np.sort(positive)[::-1];net=float(contrib.sum());c1=int(c['legacy_indices'].index(1));specified=[c['legacy_indices'].index(cl) for cl in [13,16,20] if c['legacy_indices'].index(cl)<C]
                tables['GAIN_CONCENTRATION'].append(dict(stream=stream,stage=s,mode=mode,positive_sum_pp=float(positive.sum()),negative_sum_pp=float(negative.sum()),net_macro_pp=net,top1_share_positive=frac(top[0],positive.sum()),top3_share_positive=frac(top[:3].sum(),positive.sum()),positive_classes=int((contrib>0).sum()),negative_classes=int((contrib<0).sum()),legacyC1_contribution_pp=float(contrib[c1]) if c1<C else None,legacyC13_C16_C20_contribution_pp=float(contrib[specified].sum()),specified_share_net=frac(contrib[specified].sum(),net),specified_share_positive=frac(positive[specified].sum(),positive.sum()),total_rescue=int((~goodT&good).sum()),total_harm=int((goodT&~good).sum()),unique_rescue_components=uniq(comps,~goodT&good),unique_harm_components=uniq(comps,goodT&~good)))
                if mode in vp['actions']:
                    action=arr(vp['actions'][mode]);assert np.array_equal(candidate[np.arange(len(yy)),action],pred)
                    local=action>0;changed=pred!=tp;rescue=~goodT&good;harm=goodT&~good;opportunity=~goodT&cc[:,1:].any(-1);risk=goodT&(~cc[:,1:]).any(-1)
                    for group,g in gg.items():
                        tables['ROUTING_DECISION_QUALITY'].append(dict(stream=stream,stage=s,role='VAL_FULL',mode=mode,group=group,count=int(g.sum()),action0=int((g&(action==0)).sum()),action1=int((g&(action==1)).sum()),action2=int((g&(action==2)).sum()),local_selected=int((g&local).sum()),local_prediction_changed=int((g&local&changed).sum()),local_prediction_unchanged=int((g&local&~changed).sum()),rescue=int((g&rescue).sum()),harm=int((g&harm).sum()),both_correct=int((g&goodT&good).sum()),both_wrong=int((g&~goodT&~good).sum()),both_wrong_prediction_changed=int((g&~goodT&~good&changed).sum()),available_rescues=int((g&opportunity).sum()),available_harm_risks=int((g&risk).sum()),rescue_utilization=frac((g&rescue).sum(),(g&opportunity).sum()),harm_incidence_given_risk=frac((g&harm).sum(),(g&risk).sum()),macro_net_pp=cb(good.astype(int)-goodT.astype(int),yy,g,C)*100 if g.any() else None,row_policy_probabilities_available=False))
            dif=arr(vp['predictions']['GRPO_CF'])!=arr(vp['predictions']['GROUP_EXACT_CF']);tables['PREDICTION_AGREEMENT'].append(dict(stream=stream,stage=s,comparison='GRPO_CF_vs_GROUP_EXACT_CF',different_predictions=int(dif.sum()),query_count=len(yy),parameter_probability_trajectory_equality_claim=False))
            del oo,fu,vp,oldmeta
    assert zero_covered==327680
    for name,rows in tables.items():csvout(out/(name+'.csv'),rows)
    csvout(out/'RECOMPUTED_STAGE_METRICS.csv',stage_metrics)
    coverage=[dict(stream=k[0],role=k[1],**{field:len(v) for field,v in values.items()}) for k,values in unique.items()];csvout(out/'UNIQUE_EVIDENCE_COVERAGE.csv',coverage)
    write(out/'ANALYSIS_COUNTS.json',{'new_optimizer_steps':0,'new_backwards':0,'new_model_forwards':0,'new_image_reads':0,'new_encoder_forwards':0,'new_VAL_model_scoring_rows':0,'TEST_reads':0,'saved_reads':READS,'observed_sampled_groups_deduplicated':zero_covered,'old_policy_probabilities_and_sampled_actions_saved':False,'row_endpoint_policy_probabilities_saved':False})
    write(out/'ANALYSIS_COSTS.json',{'analysis_CPU_seconds':time.process_time()-BOOT_CPU,'analysis_wall_seconds':time.time()-BOOT_WALL,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'new_output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'CPU_threads':2,'historical_training_costs_added':False,'preparation_SSH_document_publication_CPU':'nonzero not fully sampled'})
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<512*2**20
    print(json.dumps({'status':'AVAILABLE_ANALYSIS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION','counts':READS,'observed_groups':zero_covered}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();run(args.source,args.output)
