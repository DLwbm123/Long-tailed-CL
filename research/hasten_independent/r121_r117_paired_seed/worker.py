"""One fixed new-seed224 paired mechanism replication; no old task adapters."""
import csv
import gc
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from .core import write
from .rl_kernel import rl_corrected

ARMS=['SEED_TARS_FULL_IS','SEED_GRPO_FULL_IS','SEED_CE_FULL_IS']


def load_package(path,name):
    spec=importlib.util.spec_from_file_location(name,Path(path)/'__init__.py',submodule_search_locations=[str(path)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    return __import__(name+'.worker',fromlist=['Experiment'])


def set_weights(e,w):
    e.w=w.double().cpu();e.p=e.class_counts.double()/e.class_counts.sum()
    e.Z=float((e.p*e.w).sum());e.q=e.p*e.w/e.Z;e.rho=e.p/e.q
    assert torch.allclose(e.rho*e.w,torch.full_like(e.w,e.Z),atol=1e-12)


def coefficients(e,arm):
    return (e.rho.float().cuda() if arm==ARMS[1] else (e.rho*e.w).float().cuda()),e.rho.float().cuda()


def setup(cfg):
    for key,sha in [('base_source','5c0a106fe77f98b5ae15f04e953ad93cad8831e5')]:
        assert json.loads((Path(cfg[key])/'CONFIG.private.json').read_text())['source_commit']==sha
    assert json.loads((Path(cfg['base_source'])/'CONTROL_EXIT.json').read_text())['exit_code']==0
    base=load_package(cfg['base_source'],'q114_dependency')
    helper=load_package(cfg['support_source'],'q118_dependency')
    dep=load_package(cfg['allocation_source'],'q117_dependency')
    e=base.Experiment(cfg)  # Constructor creates generic LoRA; it never loads a task checkpoint.
    assert all(int(torch.count_nonzero(a.b))==0 for a in e.engine.adapters)
    assert cfg['seed']==74006 and cfg['sampler_seed']==11774006 and cfg['response_seed_base']==11481000
    torch.manual_seed(74006)
    generator=torch.Generator(device='cuda').manual_seed(74006)
    changed=[]
    with torch.no_grad():
        for adapter in e.engine.adapters:
            old=adapter.a.detach().clone()
            torch.nn.init.kaiming_uniform_(adapter.a,a=math.sqrt(5),generator=generator)
            adapter.b.zero_();changed.append(not torch.equal(old,adapter.a))
        expected=torch.empty_like(e.engine.adapters[0].a)
        torch.nn.init.kaiming_uniform_(expected,a=math.sqrt(5),generator=torch.Generator(device='cuda').manual_seed(74006))
        replay=torch.equal(expected,e.engine.adapters[0].a)
    assert replay and all(changed)
    e.sft=e.engine.snapshot();e.engine.freeze_reference()
    e.rl_corrected=types.MethodType(rl_corrected,e)
    step=e.step
    def measured_step(opt):
        before=[p.detach().clone() for p in e.engine.parameters];norm=step(opt)
        e.last_parameter_delta_L2=sum(float((p.detach()-old).double().square().sum()) for p,old in zip(e.engine.parameters,before))**.5
        return norm
    e.step=measured_step;set_weights(e,torch.ones(23,dtype=torch.float64))
    write(e.out/'GENERIC_INITIALIZATION.json',{'seed':74006,'all_task_adapter_B_zero':True,'old_task_checkpoints_loaded':False,'all_image_forwards_224':True,'all_adapter_A_changed_from_constructor74005':all(changed),'first_adapter_seed_replay_verified':replay,'initialization_generator':'CUDA74006','TRAIN_shuffle_seed':74006,'sampler_seed':11774006,'response_seed_base':11481000})
    return e,base,helper,dep


def native(e,helper,dep):
    e.status('NATIVE_GENERIC_SEED224_ENGINEERING');fixture=dep.objective_check()
    source=Path(e.cfg['base_source']);ids=[0,1]
    old=np.memmap(source/'TRAIN_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(6821,64,2048))
    start=time.time();features,raw=e.engine.encode(e.images(e.train,ids));torch.cuda.synchronize();encode_seconds=time.time()-start
    e.counts['TRAIN_encoder_rows']+=2
    reference=torch.from_numpy(np.array(old[:2],copy=True)).view(torch.bfloat16).cuda()
    feature_parity=float((features-reference).abs().max());assert feature_parity<=.05
    with torch.no_grad():
        full=e.engine.model(**raw,use_cache=False).logits[:,-1].float();e.counts['TRAIN_encoder_rows']+=2
        prompt=e.engine.prompt_ids.expand(2,-1);pos,_=e.engine.positions(prompt,torch.ones_like(prompt))
        hidden=e.engine.model.model(inputs_embeds=e.engine.embeddings(prompt,features),position_ids=pos,
                                    attention_mask=torch.ones_like(prompt),use_cache=False,return_dict=True).last_hidden_state
        parity=float((full-e.engine.model.lm_head(hidden[:,-1]).float()).abs().max())
    assert parity<=.05
    start=time.time();e.engine.generate(features);torch.cuda.synchronize();greedy_seconds=time.time()-start
    e.counts['generation_responses']+=2;timings={};details=[]
    for arm in ['BOOTSTRAP']+ARMS:
        e.engine.restore(e.sft);e.engine.freeze_reference();opt=e.optimizer(2e-4 if arm=='BOOTSTRAP' else 2e-5)
        start=time.time();pg,kl=coefficients(e,arm)
        e.counts['engineering_optimizer_updates']+=1
        if arm in ['BOOTSTRAP',ARMS[-1]]:
            loss,norm=e.ce(features,e.labels[ids],opt,pg);rec={'loss':loss,'gradient_norm':norm}
        else:rec=helper.audited_rl(e,features,e.labels[ids],opt,pg,kl,121000)
        rec['parameter_delta_L2']=e.last_parameter_delta_L2
        if rec['gradient_norm']==0:assert rec['parameter_delta_L2']==0
        torch.cuda.synchronize();timings[arm]=time.time()-start;details.append({'arm':arm,'result':rec});del opt
    e.engine.restore(e.sft);e.engine.freeze_reference()
    assert all(int(torch.count_nonzero(a.b))==0 and torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in e.engine.adapters)
    forecast=1.5*(13642/2*timings['BOOTSTRAP']+
                  6821/2*greedy_seconds+4*256*sum(timings[a] for a in ARMS)+4*1698/2*greedy_seconds)+1800
    passed=forecast<e.cfg['deadline']-time.time()
    write(e.out/'NATIVE_ADMISSION.json',{'passed':passed,'objective_fixture':fixture,'native_manual_max_logit_difference':parity,
          'reused_frozen_visual_feature_max_difference':feature_parity,'native_encoder_rows':4,'VAL_TEST':0,
          'two_image_encode_seconds':encode_seconds,'two_image_greedy_seconds':greedy_seconds,'two_query_seconds':timings,
          'paths':details,'generic_parameters_and_reference_restored':True,'native_unit_weights_engineering_only':True,
          'full_forecast_seconds_with_margin_and_reserve':forecast,'remaining_seconds':e.cfg['deadline']-time.time()})
    assert passed,'FULL_SEED224_MATRIX_NOT_TIME_ADMITTED'


def common_sft(e,data):
    opt=e.optimizer(2e-4);generator=torch.Generator().manual_seed(74006);ones=torch.ones(23,device='cuda')
    e.records=[]
    for epoch in range(2):
        indices=torch.randperm(6821,generator=generator)
        for lo in range(0,6821,16):
            k=e.counts['SFT_updates'];e.status('COMMON_SEED_SFT',training_epoch=epoch,update=k,total=854);start=time.time()
            lr=2e-4*.5*(1+math.cos(math.pi*k/854));opt.param_groups[0]['lr']=lr;ids=indices[lo:lo+16]
            loss,norm=e.ce(e.features(data,ids),e.labels[ids],opt,ones);e.counts['SFT_updates']+=1
            e.records.append({'phase':'SFT','arm':'SEED_SFT','epoch':epoch,'update':k,'queries':len(ids),'lr':lr,
                              'loss':loss,'gradient_norm':norm,'parameter_delta_L2':e.last_parameter_delta_L2,'seconds':time.time()-start})
            if k%32==0:write(e.out/'TRAINING_CURVES.json',e.records)
    e.sft=e.engine.snapshot();e.engine.freeze_reference();torch.save(e.sft,e.out/'SEED_SFT.private.pt');del opt


def science(e,base,helper):
    data=np.memmap(Path(e.cfg['base_source'])/'TRAIN_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(6821,64,2048))
    assert data.nbytes==1788084224
    write(e.out/'FEATURE_REUSE.json',{'TRAIN_rows':6821,'VAL_rows':1698,'TRAIN_bytes':1788084224,'VAL_bytes':445120512,'new_feature_disk_bytes':0,'mode':'read_only_no_copy'})
    common_sft(e,data)
    fit=e.evaluate(data,e.train,'SEED_SFT_TRAIN')
    w,_,median=base.tailness(fit['denominator'],fit['correct']);set_weights(e,w)
    write(e.out/'TARS_WEIGHTS.json',{'counts':fit['denominator'],'SFT_TRAIN_correct':fit['correct'],
          'class_names':e.names,'weights':e.w.tolist(),'raw_median':median,'Z':e.Z,'gamma':.75,
          'error_source':'new_seed74006_224SFT_TRAIN_greedy_once','new_VAL_TEST_so_far':0})
    generator=torch.Generator().manual_seed(11774006);classes=torch.multinomial(e.q,2048,replacement=True,generator=generator).reshape(256,8)
    pools={c:torch.where(e.labels==c)[0] for c in range(23)}
    tape=torch.stack([torch.stack([pools[int(c)][torch.randint(len(pools[int(c)]),(),generator=generator)] for c in row]) for row in classes])
    torch.save(tape,e.out/'QUERY_TAPE.private.pt')
    write(e.out/'ALLOCATION.json',{'natural_class_probability':e.p.tolist(),'query_class_probability':e.q.tolist(),
          'TARS_weights':e.w.tolist(),'importance_p_over_q':e.rho.tolist(),'Z':e.Z,
          'realized_queries_by_class':torch.bincount(classes.flatten(),minlength=23).tolist(),'same_tape_all_arms':True,
          'queries_per_arm':2048,'unique_TRAIN_components_by_class':[len(set(tape[classes==c].tolist())) for c in range(23)]})
    write(e.out/'SEED_SFT_SEALED.json',{'updates':854,'queries':13642,'epoch':time.time(),'VAL_TEST':0})
    for arm in ARMS:
        e.engine.restore(e.sft);assert all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in e.engine.adapters)
        opt=e.optimizer(2e-5);pg,kl=coefficients(e,arm)
        for k,ids in enumerate(tape):
            e.status('POSTTRAINING',arm=arm,update=k,total=256);start=time.time();f,y=e.features(data,ids),e.labels[ids]
            if arm==ARMS[-1]:
                loss,norm=e.ce(f,y,opt,pg);rec={'loss':loss,'gradient_norm':norm,'groups':0}
            else:rec=helper.audited_rl(e,f,y,opt,pg,kl,11481000+k)
            e.counts['posttraining_optimizer_updates']+=1
            rec.update(phase='POSTTRAINING',arm=arm,update=k,parameter_delta_L2=e.last_parameter_delta_L2,seconds=time.time()-start)
            e.records.append(rec)
            if k%8==0:write(e.out/'TRAINING_CURVES.json',e.records)
        torch.save(e.engine.snapshot(),e.out/(arm+'.private.pt'))
        write(e.out/(arm+'_SEALED.json'),{'updates':256,'common_initialization_reference_verified':True,'epoch':time.time(),'VAL_TEST':0})
        del opt;gc.collect();torch.cuda.empty_cache()
    write(e.out/'TRAINING_CURVES.json',e.records)
    write(e.out/'ALL_BRANCHES_SEALED.json',{'arms':['SEED_SFT']+ARMS,'epoch':time.time(),'new_VAL_TEST_so_far':0})
    del data;gc.collect()
    rows=list(csv.DictReader((Path(e.cfg['manifest'])/'val.csv').open()))
    assert len(rows)==1698 and not {r['identity_component'] for r in rows}&{r['identity_component'] for r in e.train}
    write(e.out/'VAL_READ_STARTED.json',{'epoch':time.time(),'all_endpoints_already_sealed':True})
    val=np.memmap(Path(e.cfg['base_source'])/'VAL_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(1698,64,2048))
    results=[]
    for arm in ['SEED_SFT']+ARMS:
        e.engine.restore(torch.load(e.out/(arm+'.private.pt'),weights_only=True,map_location='cpu'));results.append(e.evaluate(val,rows,arm))
    old=json.loads((Path(e.cfg['allocation_source'])/'FINAL_RESULTS.json').read_text());sft,main,ordinary,ce=results;original=old['historical_SFT']
    gates={'macro_over_seedSFT_ge_1pp':main['macro']-sft['macro']>=.01,'tail_over_seedSFT_ge_2pp':main['tail']-sft['tail']>=.02,
           'head_over_seedSFT_ge_minus1pp':main['head']-sft['head']>=-.01,'macro_over_GRPO_positive':main['macro']>ordinary['macro'],
           'tail_over_GRPO_positive':main['tail']>ordinary['tail'],'macro_over_CE_positive':main['macro']>ce['macro'],
           'tail_over_CE_positive':main['tail']>ce['tail']}
    directional={base+'_'+metric+'_positive':main[metric]>row[metric] for base,row in [('SFT',sft),('GRPO',ordinary),('CE',ce)] for metric in ['macro','tail']}
    write(e.out/'FINAL_RESULTS.json',{'results':results,'gates':gates,'all_gates_pass':all(gates.values()),'directional_positive_signal':directional,
          'all_six_positive_directions_repeat':all(directional.values()),'historical_original_SFT':original,'historical_R117':old['results'],
          'seed':74006,'TEST':0,'seed224_SFT_updates':854,'TRAIN_fit_greedy_queries':6821,'independent_confirmation':False,
          'patient_independence_verified':False,'scope':'single_newseed_paired_static_mechanism_replication_not_success_confirmation'})
    e.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=all(gates.values()))


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    e=None
    try:e,base,helper,dep=setup(cfg);native(e,helper,dep);science(e,base,helper)
    finally:
        if e is not None:write(e.out/'COSTS.json',{'counts':e.counts,'CPU_seconds':time.process_time()-e.cpu,'wall_seconds':time.time()-e.begin,
              'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
              'new_feature_disk_bytes':0,'TRAIN_RAM_feature_bytes':0,'reused_TRAIN_rows':6821,'reused_VAL_rows':1698,
              'historical_training_and_encoding_not_newly_executed':True})
