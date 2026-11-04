"""Frozen local joint enumeration, followed only on admission by a fit handoff."""
import os,json,time,traceback,resource,csv,itertools,subprocess
from pathlib import Path
import torch
from common import write,save,guard
from check import bank,splits,blocks,scores,metrics,select,selfcheck

c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output'])
started=time.time();cpu=time.process_time()
counts={'optimizer':0,'image_encoder':0,'VAL':0,'TEST':0,'B_plan_query_rows':0,'F_plan_query_rows':0,'candidate_partitions':0,'joint_plans':0,'contexts':0,'feature_cache_files':0}
records=[];plans=list(itertools.product(range(3),repeat=2))

def scalar(x):return float(x)

try:
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(74002)
    scpu=time.process_time();swall=time.time();software=selfcheck();software.update(CPU_seconds=time.process_time()-scpu,wall_seconds=time.time()-swall,included_in_worker=True)
    write(r/'PRECHECK.json',software)
    assert torch.cuda.is_available()
    own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    torch.zeros(1,device='cuda');own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert any(line.split(',')[0].strip()==str(os.getpid()) and c['GPU_UUID'] in line for line in own.splitlines()),'DEVICE_IDENTITY'
    probe=r/'probe.private.tmp';probe.write_bytes(b'structural-check');assert probe.read_bytes()==b'structural-check';probe.unlink()
    src=json.loads(Path(c['source_binding']).read_text())
    rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()))
    import io
    assert rows==list(csv.DictReader(io.StringIO(src['train_manifest']))) and len(rows)==6821
    y=torch.tensor([c['order'].index(int(v['original_label'])) for v in rows])
    assert torch.bincount(y,minlength=23).tolist()==c['train_counts']
    roles={k:src['roles'][v] for k,v in [('S','S'),('B','W'),('F','F')]}
    assert [len(roles[k]) for k in ['S','B','F']]==[3405,1713,1703]
    comp=[{rows[i]['identity_component'] for i in roles[k]} for k in roles]
    assert all(not a&b for i,a in enumerate(comp) for b in comp[i+1:])
    write(r/'ASSET_CONTRACT.json',{'passed':True,'role_counts':{k:len(v) for k,v in roles.items()},'component_disjoint':True,'original_teacher_full_TRAIN_history':True,'F_historical_development':True,'no_new_encoder':True,'no_VAL_TEST':True})
    write(r/'STARTED_CHECK.private.json',{'start':c['start'],'deadline':c['deadline'],'worker':os.getpid(),'physical_GPU':c['physical_gpu'],'GPU_UUID':c['GPU_UUID']})
    for stage in range(6):
        guard(c);write(r/'STATUS.json',{'phase':'STRUCTURAL_CHECK','stage':stage,'counts':counts})
        pack=torch.load(Path(c['cache_source'])/f'teacher_s{stage}.private.pt',map_location='cpu',weights_only=False);counts['feature_cache_files']+=1
        C=sum(map(len,c['stages'][:stage+1]));ids=pack['ids'];assert ids==[i for i,v in enumerate(y.tolist()) if v<C]
        assert torch.equal(pack['labels'],y) and pack['filled'][ids].all() and pack['stage']==stage
        h=torch.load(Path(c['strong_heads'])/f'head{stage}.private.pt',map_location='cpu',weights_only=False)
        base=h['prototypes'].cuda();assert base.shape==(C,512) and torch.isfinite(base).all()
        candidates=[];bank_sizes=[]
        for cl in range(C):
            si=[i for i in roles['S'] if int(y[i])==cl];z=pack['z'][si].cuda()
            assert torch.isfinite(z).all() and torch.allclose(z.norm(dim=1),torch.ones(len(z),device='cuda'),atol=2e-5)
            b=bank(z,[rows[i]['identity_component'] for i in si]);bank_sizes.append(len(b));candidates.append(splits(b));counts['candidate_partitions']+=len(candidates[-1])
        pairs=blocks(base,candidates)
        save(r/f'CONSTRUCTION_s{stage}.private.pt',{'base':base.cpu(),'candidates':[[p.cpu() for p in v] for v in candidates],'banksizes':bank_sizes,'blocks':pairs,'source':'S only; fixed teacher baseline'})
        assert len(pairs)==4,'NOT_ADMITTED_STRUCTURE_COVERAGE'
        tail=torch.tensor(c['train_counts'][:C],device='cuda')<250
        old=torch.arange(C,device='cuda')<(sum(map(len,c['stages'][:stage])) if stage else 0)
        if stage==0:
            bi=sorted(i for i in roles['B'] if int(y[i])<C)
            bq=pack['z'][bi].cuda();by=y[bi].cuda();profile=[]
            for _ in range(2):
                torch.cuda.synchronize();t=time.time()
                trial=scores(bq,base,candidates,pairs[0]);metrics(trial,by,tail,old)
                torch.cuda.synchronize();profile.append(time.time()-t)
            projected=profile[1]*2*24*2*(1713/len(bi))*(23/C)+300+2*(time.time()-started)
            write(r/'BUDGET.json',{'passed':projected<1800 and c['deadline']-time.time()>projected+7200,'precheck_projection_seconds':projected,'second_pass_seconds':profile[1],'safety':2,'construction_IO_reserve_seconds':300,'conditional_fit_not_yet_admitted':True,'engineering_B_plan_query_rows':len(bi)*9*2,'engineering_F_VAL_TEST':0,'engineering_optimizer_encoder':0,'full_plan_original_clock_unchanged':True})
            assert projected<1800 and c['deadline']-time.time()>projected+7200,'NOT_ADMITTED_BUDGET'
            write(r/'NATIVE.json',{'passed':True,'own_device_UUID_verified':True,'physical_GPU':c['physical_gpu'],'exact_NOOP':torch.equal(trial[0],(20*bq@base.T).double()),'new_optimizer_encoder':0,'reference_frozen':True,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        qs={}
        for role in ['B','F']:
            ri=sorted(i for i in roles[role] if int(y[i])<C);q=pack['z'][ri].cuda();yy=y[ri].cuda()
            assert torch.isfinite(q).all() and torch.allclose(q.norm(dim=1),torch.ones(len(q),device='cuda'),atol=2e-5)
            qs[role]=(q,yy,ri)
        for block,pair in enumerate(pairs):
            guard(c);out={};private={}
            for role,(q,yy,ri) in qs.items():
                sc=scores(q,base,candidates,pair);m,pred=metrics(sc,yy,tail,old)
                assert torch.equal(sc[0],(20*q@base.T).double()),'NOOP_NOT_EXACT'
                out[role]=m;private[role]={'predictions':pred.cpu(),'labels':yy.cpu(),'indices':ri}
                counts[role+'_plan_query_rows']+=len(q)*9
            chosen=select(out['B']['reward']);oracle=select(out['F']['reward']);counts['contexts']+=1;counts['joint_plans']+=9
            row={'stage':stage,'block':block,'B_queries':len(qs['B'][0]),'F_queries':len(qs['F'][0]),'B_selected_plan':chosen,'F_oracle_plan_diagnostic_only':oracle,'joint_plans':9}
            for role,m in out.items():
                for name,j in [('selected',chosen),('fixed',4),('oracle',select(m['reward']))]:
                    for metric in ['macro','tail','old']:
                        row[f'{role}_{name}_{metric}_delta_pp']=scalar((m[metric][j]-m[metric][0])*100)
                    for metric in ['reward','harm','rescue','tiny_harm','changes']:
                        row[f'{role}_{name}_{metric}']=scalar(m[metric][j])
                for metric in ['macro','tail']:
                    row[f'{role}_uniform_{metric}_delta_pp']=scalar((m[metric].mean()-m[metric][0])*100)
            records.append(row)
            save(r/f'CONTEXT_s{stage}_b{block}.private.pt',{'pair':pair,'plans':plans,'results':private,'selected_on_B':chosen})
            write(r/'ACTIVE_COUNTS.private.json',{'stage':stage,'block_complete':block+1,'counts':counts})
        del pack,h,qs,candidates
    import statistics
    stage_rows=[]
    for s in range(6):
        v=[x for x in records if x['stage']==s]
        stage_rows.append({'stage':s,**{k:statistics.mean(x[k] for x in v) for k in records[0] if k not in ['stage','block','B_selected_plan','F_oracle_plan_diagnostic_only']}})
    mean=lambda k:statistics.mean(x[k] for x in stage_rows)
    both=lambda role:sum(x[f'{role}_selected_macro_delta_pp']>=-1e-8 and x[f'{role}_selected_tail_delta_pp']>=-1e-8 for x in stage_rows)
    A={'macro_ge_0_5':mean('B_selected_macro_delta_pp')>=.5-1e-8,'tail_nondecrease':mean('B_selected_tail_delta_pp')>=-1e-8,'stage_coverage_4of6':both('B')>=4,'genuine_rescue':sum(x['B_selected_rescue'] for x in records)>0}
    B={'macro_ge_0_3':mean('F_selected_macro_delta_pp')>=.3-1e-8,'tail_nondecrease':mean('F_selected_tail_delta_pp')>=-1e-8,'stage_coverage_4of6':both('F')>=4,'tiny_protection':sum(x['F_selected_tiny_harm'] for x in records)==0}
    for ref in ['fixed','uniform']:
        for metric in ['macro','tail']:
            B[f'vs_{ref}_{metric}_nonnegative']=mean(f'F_selected_{metric}_delta_pp')-mean(f'F_{ref}_{metric}_delta_pp')>=-1e-8
    passed=all(A.values()) and all(B.values())
    for name,v in [('CONTEXTS.csv',records),('STAGES.csv',stage_rows)]:
        with (r/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
    summary={'passed':passed,'gate_A':A,'gate_B':B,'mean':{k:mean(k) for k in stage_rows[0] if k!='stage'},'B_stages_both_nonnegative':both('B'),'F_stages_both_nonnegative':both('F'),'tiny_harm_exposures':sum(x['F_selected_tiny_harm'] for x in records),'counts':counts,'scope':'24 fixed local joint contexts,9 complete plans each; finite candidate reachability only; no independent patients or stage repeats'}
    write(r/'RESULT_SUMMARY.json',summary)
    # Saved-only scalar reconstruction; no new cache/query forwards or optimizer.
    reread=list(csv.DictReader((r/'CONTEXTS.csv').open()));assert len(reread)==24
    for role in ['B','F']:
        for metric in ['macro','tail']:
            assert abs(statistics.mean(float(x[f'{role}_selected_{metric}_delta_pp']) for x in reread)-mean(f'{role}_selected_{metric}_delta_pp'))<1e-10
    write(r/'VERIFICATION.json',{'passed':True,'contexts':24,'stages':6,'joint_plans':216,'exact_NOOP_all_contexts':True,'CSV_mean_reconstruction':True,'new_optimizer_encoder_VAL_TEST':0})
    phase='PRECHECK_PASSED_FIT_CONFIG_PENDING' if passed else 'COMPLETE_STRUCTURAL_PRECHECK_NEGATIVE'
    write(r/'STATUS.json',{'phase':phase,'checks_passed':passed,'counts':counts,'RL_started':False})
    write(r/'WORKER_COMPLETE.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-started,'counts':counts})
except BaseException:
    (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'STATUS.json',{'phase':'FAILED','counts':counts});raise
finally:
    write(r/'COSTS.json',{'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-started,'worker_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'counts':counts,'software_native_included_in_worker':True,'borrowed_teacher_feature_costs_historical_not_free':True,'preparation_SSH_document_publication_CPU':'nonzero not fully sampled'})
