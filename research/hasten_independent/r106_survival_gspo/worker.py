"""Seven true bounded-history learners; current-arrival pseudo futures and sealed VAL."""
import os,json,time,copy,traceback,gc,csv
from pathlib import Path
import torch
from common import write,save,guard
from memory import from_views,join,BUDGET_BYTES
from learner import Learner,optimizer,learn,frozen
from manager import ValueNet,Policy,Choice
from encoder import Encoder
from meta import Meta
from preflight import run as preflight,ARMS
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);cpu=time.process_time();wall=time.time();counter={};records=[]
try:
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((Path(c['root_output'])/'SOFTWARE_CHECK.json').read_text())['passed']
    rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()));assert len(rows)==6821;y=torch.tensor([c['order'].index(int(x['original_label'])) for x in rows]);assert torch.bincount(y,minlength=23).tolist()==c['train_counts']
    first=[i for i in range(len(rows)) if int(y[i])<5];encoder=Encoder(c);write(r/'STATUS.json',{'phase':'NATIVE_MAX_PLAN_PROFILE','counts':counter})
    assert preflight(c,encoder,rows,y,first,counter),'FULL_PLAN_BUDGET_NOT_ADMITTED'
    write(r/'NATIVE_ADMISSION.json',{'passed':True,'counts':counter})
    while not (Path(c['root_output'])/'GLOBAL_NATIVE_ADMISSION.json').exists():guard(c);time.sleep(2)
    gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(c['seed']);value=ValueNet().cuda();more=copy.deepcopy(value);gspo=Policy().cuda();pg=copy.deepcopy(gspo);meta=Meta(c,value,more,gspo,pg)
    models={a:Learner(c['seed']).cuda() for a in ARMS};opts={a:optimizer(m) for a,m in models.items()};memories={a:None for a in ARMS};observed=[];birth=[]
    write(r/'FORMAL_STARTED.private.json',{'original_start':c['start'],'deadline':c['deadline'],'arms':ARMS,'upstream_generic_only':True,'memory_budget_bytes':BUDGET_BYTES,'max_shadow_groups':80})
    for t,classes in enumerate(c['stages']):
        guard(c);folder=r/f's{t}';folder.mkdir();lo=len(observed);C=lo+len(classes);assert classes==list(range(lo,C));ids=[i for i in range(len(rows)) if lo<=int(y[i])<C]
        write(r/'STATUS.json',{'phase':'ENCODE_CURRENT_ONLY','stage':t,'counts':counter});z=encoder.encode(c,rows,ids,counter,'TRAIN',ids);data=from_views(z,torch.tensor(ids,device='cuda'),y[ids].cuda());del z
        assert set(data.y.tolist())==set(classes);observed.extend([int((y==cl).sum()) for cl in classes]);birth.extend([t]*len(classes))
        write(r/'STATUS.json',{'phase':'META_CURRENT_ONLY','stage':t,'counts':counter});meta_records=meta.stage(data,rows,t,counter,folder)
        save(folder/'POLICY_AND_VALUE.private.pt',{'value':value.state_dict(),'more':more.state_dict(),'gspo':gspo.state_dict(),'pg':pg.state_dict(),'value_optimizer':meta.vo.state_dict(),'more_optimizer':meta.mo.state_dict(),'gspo_optimizer':meta.go.state_dict(),'pg_optimizer':meta.po.state_dict(),'stage':t,'current_only_meta':True})
        for arm in ARMS:
            guard(c);write(r/'ACTIVE_COUNTS.private.json',{'phase':'FORMAL_CLASSIFIER','stage':t,'arm':arm,'counts':counter});m=models[arm];o=opts[arm];teacher=frozen(m) if m.C else None;previous=memories[arm];pool=join(previous,data);assert set(pool.pid.tolist())<=set(ids)|(set(previous.pid.tolist()) if previous is not None else set())
            m.arrive(data,C);before=counter.get('formal_classifier_optimizer',0);learn(m,o,pool,200,c['seed']+t*100+1,counter,teacher,64,'formal')
            val=more if arm=='VALUE_GREEDY_MORE' else value;mode={'VALUE_GREEDY':'VALUE','VALUE_GREEDY_MORE':'VALUE','SURVIVAL_GSPO':'POLICY','SURVIVAL_PG':'POLICY'}.get(arm,arm);actor=gspo if arm=='SURVIVAL_GSPO' else pg
            choice=Choice(pool,m,observed,birth,t,256,val);memory,_,info=choice.construct(mode,actor=actor,generator=torch.Generator(device='cuda').manual_seed(c['seed']+70000+t),record=False);assert memory.assert_subset(pool)
            if arm=='GLOBAL':assert info['local_tokens']==0
            learn(m,o,memory,100,c['seed']+t*100+2,counter,teacher,64,'formal');assert counter['formal_classifier_optimizer']-before==300
            bytes_=memory.serialize(folder/f'{arm}_MEMORY.private.bin',observed,birth);assert bytes_<=BUDGET_BYTES;memories[arm]=memory
            save(folder/f'{arm}_MODEL.private.pt',{'model':m.state_dict(),'optimizer':o.state_dict(),'seen':C,'stage':t,'arm':arm,'observed_counts':observed.copy(),'birth':birth.copy(),'memory_bytes':bytes_,'formal_updates':300})
            restore=Learner(c['seed']).cuda();pack=torch.load(folder/f'{arm}_MODEL.private.pt',weights_only=False);restore.load_state_dict(pack['model']);restore.C=pack['seen'];oo=optimizer(restore);oo.load_state_dict(pack['optimizer']);xx,_,_=memory.bags()
            with torch.no_grad():assert torch.equal(m(xx[:2]),restore(xx[:2]))
            assert all(torch.isfinite(v).all() for v in restore.state_dict().values());assert len(memory.parents)>=C;assert not set(memory.pid.tolist())-set(pool.pid.tolist());assert len(observed)==C
            records.append({'stage':t,'arm':arm,'seen':C,'new_parents':len(ids),'retained_parents':len(memory.parents),'global_tokens':info['global_tokens'],'local_tokens':info['local_tokens'],'memory_bytes':bytes_,'formal_updates':300,'strict_restore':True,'old_views_from_own_previous_memory_only':True,'per_class_parent_coverage':torch.bincount(memory.y[memory.gidx],minlength=C).cpu().tolist()});write(r/'MEMORY_AND_ACCESS_AUDIT.json',records)
            del restore,oo,pack,xx,teacher,choice,pool
        write(folder/'STAGE_SEAL.json',{'stage':t,'classifier_endpoints':7,'policy_value_endpoint':1,'current_train_classes':classes,'VAL_TEST':0});del data;gc.collect();torch.cuda.empty_cache()
    assert counter['formal_classifier_optimizer']==10500;write(r/'LEARNING_SEAL.json',{'classifier_endpoints':35,'policy_value_endpoints':5,'formal_classifier_updates':10500,'counts':counter,'VAL_TEST_images':0})
    write(r/'STATUS.json',{'phase':'WAIT_GLOBAL_SEAL','counts':counter})
    while not (Path(c['root_output'])/'GLOBAL_LEARNING_SEAL.json').exists():guard(c);time.sleep(2)
    import evaluate
    evaluate.run(c,encoder,counter);write(r/'STATUS.json',{'phase':'COMPLETE','counts':counter});write(r/'WORKER_COMPLETE.json',{'counts':counter,'core_CPU_seconds':time.process_time()-cpu,'core_wall_seconds':time.time()-wall})
except BaseException:
    (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'FAILURE_COUNTS.json',counter);write(r/'STATUS.json',{'phase':'FAILED_PRESERVED','counts':counter});raise
finally:
    write(r/'COSTS.json',{'counts':counter,'core_CPU_seconds':time.process_time()-cpu,'core_wall_seconds':time.time()-wall,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'native_profile_included_once_in_worker':True,'generic_encoder_shared_physical_work_vs_method_logical_costs':'see protocol; no historical supervised model/feature reuse','preparation_import_SSH_document_delivery_CPU':'nonzero not fully sampled'})
