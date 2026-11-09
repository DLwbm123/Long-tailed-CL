"""Single fixed448 visual intervention, original SFT, matched RL/CE controls."""
import csv
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .core import write

ARMS = ['VISUAL_TARS_FULL_IS', 'VISUAL_GRPO_FULL_IS', 'VISUAL_CE_FULL_IS']


def setup(cfg):
    source = Path(cfg['support_source'])
    assert json.loads((source/'CONFIG.private.json').read_text())['source_commit'] == 'd14b7ed3f788f348151d80140764346ab72fbd0d'
    assert json.loads((source/'CONTROL_EXIT.json').read_text())['exit_code'] == 0
    name = 'q118_dependency'
    spec = importlib.util.spec_from_file_location(name, source/'__init__.py', submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec);sys.modules[name] = package;spec.loader.exec_module(package)
    helper = __import__(name+'.worker', fromlist=['dependency'])
    dep = helper.dependency(cfg);e = dep.make_experiment(cfg)
    from transformers import AutoProcessor
    engine = e.engine
    engine.processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True,
                                                     min_pixels=448**2, max_pixels=448**2)
    @torch.no_grad()
    def encode(images):
        inputs = engine.processor(text=[engine.text]*len(images), images=[im.resize((448,448)) for im in images],
                                  padding=True, return_tensors='pt').to('cuda')
        assert torch.equal(inputs.image_grid_thw, torch.tensor([[1,32,32]], device='cuda').expand(len(images),-1))
        assert all(torch.equal(row, inputs.input_ids[0]) for row in inputs.input_ids)
        engine.prompt_ids = inputs.input_ids[:1].detach();engine.grid = inputs.image_grid_thw[:1].detach()
        features = engine.model.visual(inputs.pixel_values.to(engine.model.visual.dtype), grid_thw=inputs.image_grid_thw)
        assert features.shape == (len(images)*256,2048)
        return features.reshape(len(images),256,2048), inputs
    def embeddings(ids, features):
        assert features.shape == (len(ids),256,2048)
        embedded = engine.model.get_input_embeddings()(ids);mask = ids == engine.model.config.image_token_id
        assert bool((mask.sum(-1) == 256).all())
        return embedded.masked_scatter(mask[:,:,None].expand_as(embedded), features.to(embedded.dtype))
    engine.encode, engine.embeddings = encode, embeddings
    step = e.step
    def measured_step(opt):
        before = [p.detach().clone() for p in engine.parameters];norm = step(opt)
        e.last_parameter_delta_L2 = sum(float((p.detach()-old).double().square().sum())
                                         for p,old in zip(engine.parameters,before)) ** .5
        return norm
    e.step = measured_step
    return e, helper, dep


def coefficients(e, arm):
    return e.coefficients('ALLOC_GRPO_FULL_IS' if arm == ARMS[1] else 'ALLOC_TARS_FULL_IS')


def native(e, helper, dep):
    e.status('NATIVE_448_ENGINEERING');fixture = dep.objective_check()
    start = time.time();features,raw = e.engine.encode(e.images(e.train,[0,1]));torch.cuda.synchronize()
    encode_seconds = time.time()-start;e.counts['TRAIN_encoder_rows'] += 2
    with torch.no_grad():
        logits = e.engine.model(**raw,use_cache=False).logits[:,-1].float()
        e.counts['TRAIN_encoder_rows'] += 2  # The native parity path repeats the visual forward.
        ids = e.engine.prompt_ids.expand(2,-1);pos,_ = e.engine.positions(ids,torch.ones_like(ids))
        hidden = e.engine.model.model(inputs_embeds=e.engine.embeddings(ids,features),position_ids=pos,
                                     attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
        manual = e.engine.model.lm_head(hidden[:,-1]).float();parity = float((logits-manual).abs().max())
    assert parity <= .05, ('NATIVE_448_CACHE_PARITY',parity)
    start = time.time();e.engine.generate(features);torch.cuda.synchronize();greedy_seconds = time.time()-start
    e.counts['generation_responses'] += 2
    timing = {};details = []
    for arm in ARMS:
        e.engine.restore(e.sft);e.engine.freeze_reference();opt = e.optimizer(2e-5);pg,kl = coefficients(e,arm);start=time.time()
        if arm == ARMS[-1]:
            loss,norm = e.ce(features,e.labels[:2],opt,pg);rec = {'loss':loss,'gradient_norm':norm}
        else:rec = helper.audited_rl(e,features,e.labels[:2],opt,pg,kl,119000)
        e.counts['engineering_optimizer_updates'] += 1;rec['parameter_delta_L2'] = e.last_parameter_delta_L2
        assert rec['parameter_delta_L2'] >= 0
        if rec['gradient_norm'] == 0:assert rec['parameter_delta_L2'] == 0
        torch.cuda.synchronize();timing[arm] = time.time()-start;details.append({'arm':arm,'result':rec});del opt
    e.engine.restore(e.sft);e.engine.freeze_reference()
    assert all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in e.engine.adapters)
    projected = 1.5*((1556+1698)/2*encode_seconds + 4*256*sum(timing.values()) + 4*1698/2*greedy_seconds)+1800
    passed = projected < e.cfg['deadline']-time.time()
    write(e.out/'NATIVE_ADMISSION.json',{'passed':passed,'objective_fixture':fixture,'actual_grid':[1,32,32],
          'visual_tokens':256,'native_manual_max_logit_difference':parity,'two_image_encode_seconds':encode_seconds,
          'two_image_greedy_seconds':greedy_seconds,'two_query_update_seconds':timing,'paths':details,
          'reference_original_parameters_restored':True,'TRAIN_encoder_rows':4,'VAL_TEST':0,
          'full_forecast_seconds_with_margin_and_reserve':projected,'remaining_seconds':e.cfg['deadline']-time.time()})
    assert passed,'FULL_448_MATRIX_NOT_TIME_ADMITTED'


def encode_cache(e, rows, ids, filename, split):
    result = np.memmap(e.out/filename,mode='w+',dtype=np.uint16,shape=(len(ids),256,2048))
    for lo in range(0,len(ids),2):
        e.status('ENCODE_'+split,rows_done=lo,rows_total=len(ids));chunk=ids[lo:lo+2]
        features,_ = e.engine.encode(e.images(rows,chunk))
        e.counts[split+'_encoder_rows'] += len(chunk)
        result[lo:lo+len(chunk)] = features.detach().cpu().view(torch.uint16).numpy()
    result.flush()
    return result


def gates_for(visual,main,ordinary,ce,original):
    return {'macro_over_visualSFT_ge_1pp':main['macro']-visual['macro']>=.01,
            'tail_over_visualSFT_ge_2pp':main['tail']-visual['tail']>=.02,
            'head_over_visualSFT_ge_minus1pp':main['head']-visual['head']>=-.01,
            'macro_over_GRPO_positive':main['macro']>ordinary['macro'],'tail_over_GRPO_positive':main['tail']>ordinary['tail'],
            'macro_over_CE_positive':main['macro']>ce['macro'],'tail_over_CE_positive':main['tail']>ce['tail'],
            'end_to_end_macro_over_original_ge_1pp':main['macro']-original['macro']>=.01,
            'end_to_end_tail_over_original_ge_2pp':main['tail']-original['tail']>=.02,
            'end_to_end_head_over_original_ge_minus1pp':main['head']-original['head']>=-.01}


def science(e,helper):
    root = Path(e.cfg['allocation_source']);tape = torch.load(root/'QUERY_TAPE.private.pt',weights_only=True,map_location='cpu')
    assert tape.shape == (256,8);unique,inverse = torch.unique(tape.flatten(),sorted=True,return_inverse=True)
    assert len(unique)==1556 and torch.equal(unique[inverse].reshape_as(tape),tape)
    compact = inverse.reshape_as(tape)
    torch.save({'global_TRAIN_rows':unique,'compact_query_tape':compact},e.out/'COMPACT_INDEX.private.pt')
    write(e.out/'ALLOCATION.json',json.loads((root/'ALLOCATION.json').read_text()))
    train = encode_cache(e,e.train,unique.tolist(),'TRAIN_FEATURES.private.bin','TRAIN')
    write(e.out/'TRAIN_CACHE_COMPLETE.json',{'rows':1556,'tokens':256,'dimension':2048,'bytes':1556*256*2048*2})
    e.records=[]
    write(e.out/'VISUAL_SFT_SEALED.json',{'parameter_updates':0,'fixed_original_SFT_parameters':True,'epoch':time.time(),'VAL_TEST':0})
    for arm in ARMS:
        e.engine.restore(e.sft)
        assert all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in e.engine.adapters)
        opt=e.optimizer(2e-5);pg,kl=coefficients(e,arm)
        for k,(global_ids,local_ids) in enumerate(zip(tape,compact)):
            e.status('POSTTRAINING',arm=arm,update=k,total=256);start=time.time()
            f,y=e.features(train,local_ids),e.labels[global_ids]
            if arm==ARMS[-1]:
                loss,norm=e.ce(f,y,opt,pg);rec={'loss':loss,'gradient_norm':norm,'groups':0}
            else:rec=helper.audited_rl(e,f,y,opt,pg,kl,11480000+k)
            e.counts['posttraining_optimizer_updates']+=1
            rec.update(phase='POSTTRAINING',arm=arm,update=k,parameter_delta_L2=e.last_parameter_delta_L2,seconds=time.time()-start)
            e.records.append(rec)
            if k%8==0:write(e.out/'TRAINING_CURVES.json',e.records)
        torch.save(e.engine.snapshot(),e.out/(arm+'.private.pt'))
        write(e.out/(arm+'_SEALED.json'),{'updates':256,'epoch':time.time(),'common_initialization_reference_verified':True,'VAL_TEST':0})
        del opt;gc.collect();torch.cuda.empty_cache()
    write(e.out/'TRAINING_CURVES.json',e.records)
    write(e.out/'ALL_BRANCHES_SEALED.json',{'arms':['VISUAL_SFT']+ARMS,'epoch':time.time(),'VAL_encoder_rows':0,'new_VAL_TEST_so_far':0})
    rows=list(csv.DictReader((Path(e.cfg['manifest'])/'val.csv').open()))
    assert len(rows)==1698 and not {r['identity_component'] for r in rows}&{r['identity_component'] for r in e.train}
    write(e.out/'VAL_CACHE_STARTED.json',{'epoch':time.time(),'all_endpoints_already_sealed':True})
    val=encode_cache(e,rows,list(range(1698)),'VAL_FEATURES.private.bin','VAL')
    write(e.out/'VAL_CACHE_COMPLETE.json',{'rows':1698,'tokens':256,'dimension':2048,'bytes':1698*256*2048*2})
    results=[]
    for arm in ['VISUAL_SFT']+ARMS:
        state=e.sft if arm=='VISUAL_SFT' else torch.load(e.out/(arm+'.private.pt'),weights_only=True,map_location='cpu')
        e.engine.restore(state);results.append(e.evaluate(val,rows,arm))
    old=json.loads((root/'FINAL_RESULTS.json').read_text());visual,main,ordinary,ce=results
    gates=gates_for(visual,main,ordinary,ce,old['historical_SFT'])
    write(e.out/'FINAL_RESULTS.json',{'results':results,'gates':gates,'all_gates_pass':all(gates.values()),
          'historical_original_SFT':old['historical_SFT'],'historical_R117_primary':old['results'][0],
          'seed':74005,'TEST':0,'new_SFT_updates':0,'resolution':[448,448],'independent_confirmation':False,
          'patient_independence_verified':False,'scope':'static_visual_information_and_incremental_RL_diagnostic'})
    e.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=all(gates.values()))


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    e=None
    try:e,helper,dep=setup(cfg);native(e,helper,dep);science(e,helper)
    finally:
        if e is not None:write(e.out/'COSTS.json',{'counts':e.counts,'CPU_seconds':time.process_time()-e.cpu,
              'wall_seconds':time.time()-e.begin,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_feature_logical_bytes':3412066304,
              'historical_training_not_newly_executed':True})
