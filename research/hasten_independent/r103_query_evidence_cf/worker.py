"""One stream, three legal banks, fixed full protocol and global VAL barrier."""
import os,json,time,traceback,gc,csv
from pathlib import Path
import torch
from common import write,save,guard

counts={k:0 for k in ['fold_teacher_optimizer','fold_query_rows','fold_support_rows','fold_observation_rows','fold_prototype_rows','fold_micro_backwards','TRAIN_generic_view_rows','TRAIN_teacher_view_rows','VAL_generic_view_rows','VAL_teacher_view_rows','fusion_optimizer','fusion_fit_rows','router_optimizer','sampled_query_groups','sampled_actions','engineering_prefix_rows','engineering_teacher_optimizer','engineering_teacher_query_rows','engineering_fusion_optimizer','engineering_fusion_fit_rows','engineering_router_optimizer']}
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);wall=time.time();cpu=time.process_time()
try:
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from data_contract import bind,seed
    from fold_teacher import Store,stage
    from teacher_base import new_tail
    from teacher_vision import load_delta
    import fusion,build_feedback,train_routers,preflight,check
    from evidence_views import encode_views
    from router import ARMS
    rows,y,roles=bind(c);assert json.loads((Path(c['root_output'])/'SOFTWARE_CHECK.json').read_text())['status']=='PASS';torch.set_num_threads(2)
    store=Store(c,rows,y);store.arrive(c['stages'][0]);write(r/'STATUS.json',{'phase':'NATIVE_FULL_PLAN_PROFILE','counts':counts})
    assert preflight.run(c,rows,y,store,counts),'FULL_PLAN_BUDGET_NOT_ADMITTED'
    gc.collect();torch.cuda.empty_cache()
    folds={k:new_tail(store.bundle,c['seed']) for k in ['FIT_A','FIT_B']}
    write(r/'FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'scope':'20 new fold teachers,60 fusion,50 router endpoints across both streams','original_clock_not_reset':True})
    lineage=[];fallback=[];fit_records=[];feedback_records=[]
    for s,classes in enumerate(c['stages']):
        guard(c)
        if s:store.arrive(classes)
        C=sum(map(len,c['stages'][:s+1]));ids=sorted(store.position);yy=y[ids].cuda();birth=torch.tensor([j for j,cl in enumerate(c['stages'][:s+1]) for _ in cl],device='cuda');stagefolder=r/f's{s}';stagefolder.mkdir()
        heads={'FULL':torch.load(Path(c['reuse_root'])/'strong_heads'/f'head{s}.private.pt',map_location='cpu',weights_only=False)}
        tails={};full,_=new_tail(store.bundle,c['seed']);load_delta(full,heads['FULL']['delta']);full.eval().requires_grad_(False);tails['FULL']=full
        for bank in ['FIT_A','FIT_B']:
            write(r/'STATUS.json',{'phase':'FOLD_TEACHER','stage':s,'bank':bank,'counts':counts});folder=r/bank/f's{s}';folder.parent.mkdir(exist_ok=True)
            m,params=folds[bank];head=stage(c,rows,y,store,m,params,roles[bank],s,folder,counts);heads[bank]=head;tails[bank]=m
            fit=sorted(i for i in roles[bank] if y[i]<C);feedback=[i for i in ids if i not in set(fit)];fg={rows[i]['identity_component'] for i in fit};bg={rows[i]['identity_component'] for i in feedback};assert not fg&bg
            lineage.append({'stage':s,'bank':bank,'fit_rows':len(fit),'feedback_rows':len(feedback),'history_fit_components_intersect_feedback':False,'optimizer_fit_rows_only':True})
            fallback += [dict(bank=bank,**x) for x in json.loads((folder/'INNER_SUPPORT_FALLBACK.json').read_text())]
        write(r/'STATUS.json',{'phase':'TRAIN_SIX_VIEWS','stage':s,'counts':counts})
        feat=encode_views(c,rows,ids,store,tails,counts,'TRAIN')
        old=torch.load(Path(c['reuse_root'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False)
        error=float((feat['FULL'][:,0]-old['z'][ids]).abs().max());assert error<=2e-5
        P=heads['FULL']['prototypes'];assert torch.equal((20*feat['FULL'][:,0]@P.T).argmax(-1),(20*old['z'][ids]@P.T).argmax(-1));del old
        OOF=None;IN=None;banks={}
        for bank in ['FULL','FIT_A','FIT_B']:
            folder=stagefolder/bank;folder.mkdir();fit=ids if bank=='FULL' else [i for i in ids if i in set(roles[bank])];fitpos=torch.tensor([ids.index(i) for i in fit]);P=heads[bank]['prototypes'];nc=heads[bank]['train_counts'];assert torch.equal(nc,torch.bincount(y[fit],minlength=C))
            save(folder/'features.private.pt',{'ids':ids,'views':feat[bank],'fit_ids':fit,'stage':s,'bank':bank})
            write(r/'STATUS.json',{'phase':'FUSION','stage':s,'bank':bank,'counts':counts})
            models=fusion.train(c,folder,feat[bank][fitpos],y[fit],P,seed(c,s,bank,'fusion'),counts)
            tab=build_feedback.table(feat[bank],P,models,nc,birth,s,yy);banks[bank]=tab
            build_feedback.persist(folder/'table.private.pt',tab,ids,bank)
            if bank=='FULL':IN=tab
            del models
        OOF={key:torch.empty_like(value) for key,value in IN.items()};owner=torch.zeros(len(ids),dtype=torch.long)
        for bank in ['FIT_A','FIT_B']:
            feedback_pos=torch.tensor([j for j,i in enumerate(ids) if i not in set(roles[bank])],device='cuda');owner[feedback_pos.cpu()]+=1
            for key in OOF:OOF[key][feedback_pos]=banks[bank][key][feedback_pos]
        assert (owner==1).all();build_feedback.persist(stagefolder/'OOF.private.pt',OOF,ids,'opposite_outer_fit');build_feedback.persist(stagefolder/'IN.private.pt',IN,ids,'FULL')
        write(r/'STATUS.json',{'phase':'ROUTERS','stage':s,'counts':counts})
        routers=train_routers.train(c,stagefolder,OOF,IN,yy,birth,s,seed(c,s,'shared','router'),counts)
        with torch.no_grad():
            protected=(birth<s)&(torch.tensor(c['train_counts'][:C],device='cuda')<250)
            for bank,tab in [('OOF',OOF),('IN',IN)]:
                for arm,model in routers.items():
                    p=model(tab['features']).double().softmax(-1);a=p.argmax(-1);correct=tab['correct'].double();h=correct[:,0,None]*(1-correct)
                    byclass=lambda v:[float(v[yy==cl].mean()) for cl in range(C)]
                    feedback_records.append({'stage':s,'bank':bank,'arm':arm,'candidate_class_accuracy':[byclass(correct[:,a]) for a in range(3)],'selector_class_accuracy':byclass(correct.gather(1,a[:,None]).squeeze(1)),'soft_expected_class_accuracy':byclass((p*correct).sum(-1)),'class_expected_teacher_harm':byclass((p*h).sum(-1)),'policy_mean':p.mean(0).tolist(),'action_fraction':torch.bincount(a,minlength=3).double().div(len(a)).tolist(),'confidence_mean':float(tab['scores'].double().softmax(-1).max(-1).values.mean()),'protected_class_count':int(protected.sum())})
            for arm,model in routers.items():
                po=model(OOF['features']).double().softmax(-1);pi=model(IN['features']).double().softmax(-1)
                fit_records.append({'stage':s,'arm':arm,'OOF_FULL_policy_L1_mean':float((po-pi).abs().sum(-1).mean()),'OOF_FULL_action_change_fraction':float((po.argmax(-1)!=pi.argmax(-1)).double().mean()),'OOF_policy_mean':po.mean(0).tolist(),'FULL_policy_mean':pi.mean(0).tolist()})
        write(stagefolder/'STAGE_SEAL.json',{'stage':s,'fusion':6,'routers':5,'teacher_full_V0_parity_maxabs':error,'OOF_each_query_once':True,'VAL_TEST_images':0})
        del feat,banks,OOF,IN,tab,routers,tails,full;gc.collect();torch.cuda.empty_cache()
    write(r/'LINEAGE_RUNTIME_AUDIT.json',{'passed':True,'records':lineage});write(r/'INNER_SUPPORT_FALLBACK.json',fallback);write(r/'TRAIN_FEEDBACK_DIAGNOSTICS.json',feedback_records);write(r/'FOLD_TO_FULL_SHIFT.json',fit_records)
    assert counts['fusion_optimizer']==15360 and counts['router_optimizer']==12800 and counts['sampled_query_groups']==163840 and counts['sampled_actions']==1310720
    write(r/'LEARNING_SEAL.json',{'new_fold_teacher_endpoints':10,'fusion_endpoints':30,'router_endpoints':25,'fusion_updates':15360,'router_updates':12800,'counts':counts,'VAL_TEST_images':0})
    del folds;store.cache.clear();gc.collect();torch.cuda.empty_cache();write(r/'STATUS.json',{'phase':'WAIT_GLOBAL_SEAL','counts':counts})
    while not (Path(c['root_output'])/'GLOBAL_LEARNING_SEAL.json').exists():guard(c);time.sleep(2)
    import evaluate
    evaluate.run(c,rows,y,store,counts)
    write(r/'WORKER_COMPLETE.json',{'counts':counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall});write(r/'STATUS.json',{'phase':'COMPLETE','counts':counts})
except BaseException:
    (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'FAILURE_COUNTS.json',counts);write(r/'STATUS.json',{'phase':'FAILED','counts':counts});raise
finally:
    write(r/'COSTS.json',{'counts':counts,'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-wall,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'historical_FULL_teachers':'R102 costs charged separately, no R102 module arms rerun','subitems_included_in_worker':True,'preparation_binding_SSH_document_publication_CPU':'nonzero not fully sampled'})
