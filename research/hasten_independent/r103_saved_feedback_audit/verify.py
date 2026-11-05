"""Independent saved-data accounting checks; no learning code/model imports."""
import os,time,json,csv,resource,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
START=time.time();CPU=time.process_time()
from pathlib import Path
import torch
torch.set_num_threads(2)

def read(root,name):
    with (Path(root)/(name+'.csv')).open() as f:return list(csv.DictReader(f))
def near(a,b):
    if b is None:assert a=='',a
    else:assert abs(float(a)-float(b))<1e-9,(a,b)
def filt(rows,**keys):return [r for r in rows if all(str(r[k])==str(v) for k,v in keys.items())]
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def check(source,out):
    source=Path(source);out=Path(out);names=['CANDIDATE_PATTERN_SUMMARY','CANDIDATE_PATTERN_CLASS','UNIFORM_VS_CANDIDATE_PATTERN','ZERO_GROUP_DECOMPOSITION','OOF_FULL_UTILITY_TRANSITIONS','OOF_FULL_ACTION_ORDER','CLASS_GAIN_CONTRIBUTIONS','ROUTING_DECISION_QUALITY','GAIN_CONCENTRATION','RECOMPUTED_STAGE_METRICS'];data={n:read(out,n) for n in names};metric_count=0;history_groups=0;tensor_reads=0;saved_VAL_rows=0
    for z in data['ZERO_GROUP_DECOMPOSITION']:
        if z['structural_fraction']=='':assert int(z['group_denominator'])==0;continue
        st,sa,to,ef,inf=[float(z[k]) for k in ['structural_fraction','sampling_zero_fraction','total_zero_fraction','effective_fraction','informative_fraction']]
        assert abs(st+sa-to)<1e-12 and abs(to+ef-1)<1e-12 and abs(st+inf-1)<1e-12
        if inf:near(z['conditional_zero_given_informative'],sa/inf)
        else:assert z['conditional_zero_given_informative']==''
    for stream in ['a','b']:
        c=json.loads((source/stream/'RUN_CONFIG.private.json').read_text());metadata=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()));all_y=torch.tensor([c['order'].index(int(row['original_label'])) for row in metadata]);original=json.loads((source/stream/'STAGE_METRICS.json').read_text());summary=json.loads((source/stream/'RESULT_SUMMARY.json').read_text())
        for stage in range(5):
            C=sum(map(len,c['stages'][:stage+1]));folder=source/stream/f's{stage}';oo=load(folder/'OOF.private.pt');fu=load(folder/'IN.private.pt');vp=load(source/stream/f'VAL_predictions_s{stage}.private.pt');plan=load(folder/'QUERY_PLAN.private.pt');tensor_reads+=4
            assert oo['ids']==fu['ids'];yy=all_y[oo['ids']];birth=torch.tensor([s for s,cl in enumerate(c['stages'][:stage+1]) for _ in cl]);old=birth<stage;tail=torch.tensor(c['train_counts'][:C])<250
            classgroups={'all':torch.ones(C,dtype=torch.bool),'old':old,'new':~old,'head':~tail,'tail':tail,'old_head':old&~tail,'old_tail':old&tail,'new_head':~old&~tail,'new_tail':~old&tail}
            for role,tab,labels in [('TRAIN_OOF',oo,yy),('TRAIN_FULL',fu,yy),('VAL_FULL',vp,vp['labels'])]:
                correct=tab['candidate_correct'].bool() if role=='VAL_FULL' else tab['correct'].bool();codes=4*correct[:,0].long()+2*correct[:,1].long()+correct[:,2].long()
                if role=='VAL_FULL':uniform=tab['predictions']['UNIFORM'];saved_VAL_rows+=len(labels)
                else:uniform=tab['scores'].double().softmax(-1).mean(1).argmax(-1)
                for group,cls in classgroups.items():
                    mask=cls[labels];n=int(mask.sum());rows=filt(data['CANDIDATE_PATTERN_SUMMARY'],stream=stream,stage=stage,role=role,group=group);base=[r for r in rows if len(r['pattern'])==3 and set(r['pattern'])<={'0','1'}];assert len(base)==8 and sum(int(r['count']) for r in base)==n
                    for r in base:
                        good=codes==int(r['pattern'],2);assert int(r['count'])==int((good&mask).sum())
                        near(r['micro_fraction'],float(good[mask].double().mean()) if n else None)
                        parts=[good[(labels==cl)].double().mean() for cl in torch.where(cls)[0].tolist()];near(r['class_balanced_fraction'],float(torch.stack(parts).mean()) if parts else None)
                    if n:near(sum(float(r['class_balanced_fraction']) for r in base),1.)
                for cl in range(C):
                    rows=filt(data['CANDIDATE_PATTERN_CLASS'],stream=stream,stage=stage,role=role,classifier_index=cl);base=[r for r in rows if len(r['pattern'])==3 and set(r['pattern'])<={'0','1'}];assert sum(int(r['count']) for r in base)==int((labels==cl).sum())
                un=filt(data['UNIFORM_VS_CANDIDATE_PATTERN'],stream=stream,stage=stage,role=role,group='all')
                for row in un:
                    mask=codes==int(row['pattern'],2);assert int(row['UNIFORM_correct'])==int((mask&(uniform==labels)).sum())
                    if row['pattern']=='111':assert int(row['UNIFORM_correct'])==int(row['count'])
            for arm,tab in [('GRPO_CF',oo),('GRPO_IN',fu)]:
                struct=tab['correct'].all(-1)|(~tab['correct']).all(-1);zs=int(struct[plan].sum());log=json.loads((folder/f'{arm}_TRAIN_DIAGNOSTICS.json').read_text())['records'];zt=0
                for outer in range(128):
                    rec=[v for v in log if v['outer']==outer];assert len(rec)==4;vals={v['sampled_zero_group_fraction'] for v in rec};assert len(vals)==1;zt+=round(vals.pop()*128)
                result=next(v for v in filt(data['ZERO_GROUP_DECOMPOSITION'],stream=stream,stage=stage,arm=arm) if v['precision'].startswith('derived_rollout'));near(result['structural_fraction'],zs/16384);near(result['total_zero_fraction'],zt/16384);near(result['sampling_zero_fraction'],(zt-zs)/16384);assert int(result['group_denominator'])==16384;history_groups+=16384
            a=oo['correct'].long();b=fu['correct'].long()
            for action,name in [(1,'CENTER'),(2,'MULTI')]:
                ra=a[:,action]-a[:,0];rb=b[:,action]-b[:,0]
                for group,cls in classgroups.items():
                    mask=cls[yy];rows=filt(data['OOF_FULL_UTILITY_TRANSITIONS'],stream=stream,stage=stage,action=name,group=group);assert len(rows)==9
                    for r in rows:
                        event=(ra==int(r['r_OOF']))&(rb==int(r['r_FULL']));assert int(r['count'])==int((mask&event).sum())
                        near(r['micro_fraction'],float(event[mask].double().mean()) if mask.any() else None)
                        parts=[event[yy==cl].double().mean() for cl in torch.where(cls)[0].tolist()];near(r['class_balanced_fraction'],float(torch.stack(parts).mean()) if parts else None)
                    for val in [-1,0,1]:
                        assert sum(int(r['count']) for r in rows if int(r['r_OOF'])==val)==int((mask&(ra==val)).sum())
                        assert sum(int(r['count']) for r in rows if int(r['r_FULL'])==val)==int((mask&(rb==val)).sum())
            order_a=torch.sign(a[:,1]-a[:,2]);order_b=torch.sign(b[:,1]-b[:,2])
            for group,cls in classgroups.items():
                mask=cls[yy]
                for r in filt(data['OOF_FULL_ACTION_ORDER'],stream=stream,stage=stage,group=group):
                    event=(order_a==int(r['CENTER_minus_MULTI_order_OOF']))&(order_b==int(r['CENTER_minus_MULTI_order_FULL']));assert int(r['count'])==int((mask&event).sum())
                    near(r['micro_fraction'],float(event[mask].double().mean()) if mask.any() else None)
                    parts=[event[yy==cl].double().mean() for cl in torch.where(cls)[0].tolist()];near(r['class_balanced_fraction'],float(torch.stack(parts).mean()) if parts else None)
            labels=vp['labels'];den=torch.bincount(labels,minlength=C);T=vp['predictions']['T']==labels
            for mode,pred in vp['predictions'].items():
                good=pred==labels;cc=torch.bincount(labels[good],minlength=C);tc=torch.bincount(labels[T],minlength=C);rec=cc.double()/den;metrics={'micro':float(good.double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None,'old_tail':float(rec[old&tail].mean()*100) if (old&tail).any() else None};row=filt(data['RECOMPUTED_STAGE_METRICS'],stream=stream,stage=stage,mode=mode)[0]
                for key,value in metrics.items():
                    near(row[key],value);target=original[stage]['modes'][mode][key];assert target is None if value is None else abs(target-value)<1e-10
                contrib=filt(data['CLASS_GAIN_CONTRIBUTIONS'],stream=stream,stage=stage,mode=mode);assert len(contrib)==C
                for v in contrib:
                    cl=int(v['classifier_index']);mask=labels==cl;res=int((mask&~T&good).sum());harm=int((mask&T&~good).sum());assert int(v['rescue'])==res and int(v['harm'])==harm and int(v['net_correct'])==res-harm==int(cc[cl]-tc[cl]);near(v['macro_contribution_pp'],100/C*(res-harm)/int(den[cl]));assert int(v['original_label'])==c['order'][cl]
                near(sum(float(v['macro_contribution_pp']) for v in contrib),metrics['macro']-original[stage]['modes']['T']['macro'])
                for key,subset in [('tail',tail),('old_tail',old&tail)]:
                    field=key+'_contribution_pp'
                    if subset.any():near(sum(float(v[field]) for v in contrib),metrics[key]-original[stage]['modes']['T'][key])
                    else:assert all(v[field]=='' for v in contrib)
                concentration=filt(data['GAIN_CONCENTRATION'],stream=stream,stage=stage,mode=mode)[0];near(concentration['positive_sum_pp'],sum(max(0,float(v['macro_contribution_pp'])) for v in contrib));near(concentration['negative_sum_pp'],sum(min(0,float(v['macro_contribution_pp'])) for v in contrib))
                if mode in vp['actions']:
                    action=vp['actions'][mode]
                    for row in filt(data['ROUTING_DECISION_QUALITY'],stream=stream,stage=stage,mode=mode):
                        mask=classgroups[row['group']][labels];n=int(mask.sum());assert int(row['count'])==n and sum(int(row[f'action{j}']) for j in range(3))==n
                        assert int(row['local_prediction_changed'])+int(row['local_prediction_unchanged'])==int(row['local_selected'])
                        assert sum(int(row[k]) for k in ['rescue','harm','both_correct','both_wrong'])==n
                        assert int(row['local_selected'])==int((mask&(action>0)).sum())
                        cand=vp['candidate_correct'];opportunity=(~T)&cand[:,1:].any(-1);risk=T&(~cand[:,1:]).any(-1)
                        assert int(row['available_rescues'])==int((mask&opportunity).sum()) and int(row['available_harm_risks'])==int((mask&risk).sum())
                        assert int(row['rescue'])==int((mask&~T&good).sum()) and int(row['harm'])==int((mask&T&~good).sum())
                metric_count+=1
        for mode in summary['mean']:
            rows=filt(data['RECOMPUTED_STAGE_METRICS'],stream=stream,mode=mode)
            for key in ['micro','macro','tail','old_tail']:
                values=[float(v[key]) for v in rows if v[key]!=''];assert len(values)==summary['mean_valid_stages'][mode][key];assert abs(sum(values)/len(values)-summary['mean'][mode][key])<1e-10
    assert metric_count==90 and history_groups==327680
    result={'status':'COMPLETE_AVAILABLE_ANALYSES_WITH_PARTIAL_UNSAVED_POLICY_COVERAGE','passed_available_checks':True,'full_requested_information_available':False,'readouts_recomputed':metric_count,'observed_rollout_groups_deduplicated':history_groups,'patterns_exhaustive_and_class_micro_consistent':True,'zero_decomposition_identities':True,'utility_transition_marginals':True,'rescue_harm_contribution_identities':True,'UNIFORM_fixed_rule_checked_where_logits_saved':True,'all111_UNIFORM_correct_verified':True,'empty_groups_and_original_mean_denominators':True,'missing_fields':['per-row trained router probabilities','true rollout old probabilities','sampled actions/per-query zero flags'],'expected_training_history_zero_decomposition_available':False,'observed_aggregate_history_zero_decomposition_available':True,'new_optimizer_backward_model_image_encoder_VAL_scoring_TEST':0,'verification_tensor_files_read':tensor_reads,'verification_saved_VAL_query_rows':saved_VAL_rows,'CPU_seconds':time.process_time()-CPU,'wall_seconds':time.time()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    (out/'VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();check(a.source,a.output)
