"""Frozen native joint observations of already sealed crops; no policy learning."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from .core import METHODS, summarize, write


def dependency(path):
    spec=importlib.util.spec_from_file_location('q122_dependency',Path(path)/'__init__.py',submodule_search_locations=[str(path)])
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return __import__(spec.name+'.worker',fromlist=['load_base','crop'])


@torch.no_grad()
def recognize(e, images):
    start=time.time()
    content=[{'type':'image'} for _ in images]+[{'type':'text','text':e.cfg['prompt']}]
    text=e.engine.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    raw=e.engine.processor(text=[text],images=[im.resize((224,224)) for im in images],padding=True,return_tensors='pt').to('cuda')
    assert raw.image_grid_thw.tolist()==[[1,16,16]]*len(images)
    assert int((raw.input_ids==e.engine.model.config.image_token_id).sum())==64*len(images)
    e.counts['generation_attempts']+=1;e.counts['TRAIN_encoder_rows']+=len(images)
    tokens=e.engine.model.generate(**raw,do_sample=False,max_new_tokens=24,use_cache=True,pad_token_id=e.engine.pad,eos_token_id=e.engine.eos)
    response=tokens[0,raw.input_ids.shape[1]:];done=bool((response==e.engine.eos).any())
    answer=e.normalize(e.engine.tokenizer.decode(response,skip_special_tokens=True))
    lookup={e.normalize(n):c for c,n in e.names.items()}
    e.counts['generation_responses']+=1;torch.cuda.synchronize()
    return {'prediction':lookup.get(answer,-1) if done else -1,'truncated':not done,
            'input_tokens':int(raw.input_ids.numel()),'generated_tokens':int(response.numel()),
            'vision_encoder_rows':len(images),'seconds':time.time()-start}


def matrix(e, old, rank, crop):
    image=e.images(e.train,[old['global_TRAIN_row']])[0]
    boxes={m:old['results'][m]['proposal']['box'] for m in ['GLOBAL_REPEAT','CENTER_DETAIL']}
    views={'NATIVE_ROOT':[image],'DUPLICATE':[image,image],
           'FIXED_CENTER':[image,crop(image,[250,250,750,750])],
           'GLOBAL_BOX':[image,crop(image,boxes['GLOBAL_REPEAT'])],
           'DETAIL_BOX':[image,crop(image,boxes['CENTER_DETAIL'])]}
    joint=METHODS[1:];shift=rank%4;order=['NATIVE_ROOT']+joint[shift:]+joint[:shift]
    results={m:recognize(e,views[m]) for m in order}
    assert len({results[m]['input_tokens'] for m in joint})==1
    for old_name,new_name in [('GLOBAL_REPEAT','GLOBAL_BOX'),('CENTER_DETAIL','DETAIL_BOX')]:
        if not old['results'][old_name]['proposal']['valid']:
            assert results[new_name]['prediction']==results['DUPLICATE']['prediction']
    return {'global_TRAIN_row':old['global_TRAIN_row'],'selection_rank':rank,'class':old['class'],
            'split':old['split'],'boxes':boxes,'results':results}


def unchanged(e):
    assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(e.engine.snapshot(),e.instrument))
    assert all(int(torch.count_nonzero(a.ref_b))==0 and not a.reference for a in e.engine.adapters)
    return True


def native(e, old_native, crop):
    e.status('NATIVE_JOINT_OBSERVATION')
    hidden_diff=[];head_diff=[];raw_diff=[]
    with torch.no_grad():
        for row in [0,1]:
            features,raw=e.engine.encode(e.images(e.train,[row]));e.counts['TRAIN_encoder_rows']+=1
            full=e.engine.model(**raw,use_cache=False,output_hidden_states=True);e.counts['TRAIN_encoder_rows']+=1
            ids=e.engine.prompt_ids;pos,_=e.engine.positions(ids,torch.ones_like(ids))
            hidden=e.engine.model.model(inputs_embeds=e.engine.embeddings(ids,features),position_ids=pos,
                attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
            last=full.hidden_states[-1][:,-1:];manual=e.engine.model.lm_head(hidden[:,-1:])[:,0].float()
            hidden_diff.append(float((last-hidden[:,-1:]).abs().max()))
            head_diff.append(float((e.engine.model.lm_head(last)[:,0].float()-manual).abs().max()))
            raw_diff.append(float((full.logits[:,-1].float()-manual).abs().max()))
    assert max(hidden_diff)<=.05 and max(head_diff)<=.05
    start=time.time();records=[matrix(e,r,k,crop) for k,r in enumerate(old_native)];seconds=time.time()-start
    write(e.out/'NATIVE_RECORDS.private.json',records);assert unchanged(e)
    forecast=1.5*175/2*seconds+600;remaining=e.cfg['deadline']-time.time()
    proof={'passed':forecast<remaining,'single_image_hidden_parity':max(hidden_diff),
           'same_shape_last_head_parity':max(head_diff),'raw_full_head_vs_last_head_difference':max(raw_diff),
           'joint_native_HF_only_no_manual_multiimage_cache':True,'per_image_visual_tokens':64,'joint_visual_tokens':128,
           'joint_actual_input_tokens_equal':True,'native_responses':10,'native_TRAIN_encoder_rows':22,
           'two_component_matrix_seconds':seconds,'full_forecast_seconds':forecast,'remaining_seconds':remaining,
           'instrument_and_reference_immutable':True,'new_proposals_optimizer_VAL_TEST':0}
    write(e.out/'NATIVE_ADMISSION.json',proof)
    assert proof['passed'],'FULL_MATRIX_NOT_TIME_ADMITTED'


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    begin=time.time();cpu=time.process_time();e=None
    try:
        old=Path(cfg['previous_source']);assert json.loads((old/'FINAL_VERIFICATION.json').read_text())['passed']
        previous=dependency(old);base=previous.load_base(cfg['base_source']);e=base.Experiment(cfg);e.normalize=base.normalize
        e.counts={'generation_attempts':0,'generation_responses':0,'TRAIN_encoder_rows':0,'VAL_encoder_rows':0,'optimizer_updates':0,'new_proposals':0}
        assert all(int(torch.count_nonzero(a.b))==0 for a in e.engine.adapters)
        e.engine.freeze_reference();e.engine.restore(torch.load(Path(cfg['instrument_source'])/'SEED_SFT.private.pt',map_location='cpu',weights_only=True))
        e.engine.model.eval().requires_grad_(False);e.instrument=e.engine.snapshot();assert unchanged(e)
        native(e,json.loads((old/'NATIVE_RECORDS.private.json').read_text()),previous.crop)
        source=json.loads((old/'MEASUREMENTS.private.json').read_text());assert len(source)==175
        records=[]
        for k,row in enumerate(source):
            e.status('MEASURING_JOINT_TRAIN',components_completed=k,components_total=175)
            records.append(matrix(e,row,k,previous.crop))
            if k%8==0:write(e.out/'MEASUREMENTS.private.json',records)
        write(e.out/'MEASUREMENTS.private.json',records);assert unchanged(e)
        write(e.out/'MEASUREMENTS_SEALED.json',{'epoch':time.time(),'components':175,'instrument_and_reference_immutable':True,'new_proposals_optimizer_VAL_TEST':0})
        write(e.out/'CLASS_DEFINITIONS.json',json.loads((old/'CLASS_DEFINITIONS.json').read_text()))
        final=summarize(records,e.class_counts.tolist());write(e.out/'FINAL_RESULTS.json',final)
        e.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=final['all_gates_pass'])
    finally:
        if e is not None:write(e.out/'COSTS.json',{'counts':e.counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'new_feature_checkpoint_bytes':0,'historical_R122_proposals_and_R121_SFT_reused_not_free':True})
