"""Frozen native two-image proposal versus recognition measurement; no optimizer."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from .core import crop,fixtures,parse_box,summarize,write

PROMPT=('Image 1 is the full endoscopy image. Image 2 shows the region of Image 1 with normalized coordinates {roi}. '
        'Using only these views, propose one region of Image 1 to inspect for recognizing the endoscopic finding or anatomical landmark. '
        'Return only a JSON array [x1,y1,x2,y2] of four integers normalized from 0 to 1000. '
        'The box must lie inside Image 1 and each side must be at least 100 units. Do not name the category.')


def load_base(path):
    spec=importlib.util.spec_from_file_location('q114_dependency',Path(path)/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules['q114_dependency']=module;spec.loader.exec_module(module)
    return __import__('q114_dependency.worker',fromlist=['Experiment'])


@torch.no_grad()
def propose(e,image,method):
    begin=time.time();roi=[0,0,1000,1000] if method=='GLOBAL_REPEAT' else [250,250,750,750]
    detail=crop(image,roi);coords='('+','.join(f'{v/1000:.3f}' for v in roi)+')'
    prompt=PROMPT.format(roi=coords)
    messages=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}]}]
    text=e.engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    raw=e.engine.processor(text=[text],images=[image.resize((224,224)),detail.resize((224,224))],padding=True,return_tensors='pt').to('cuda')
    assert raw.image_grid_thw.tolist()==[[1,16,16],[1,16,16]]
    assert int((raw.input_ids==e.engine.model.config.image_token_id).sum())==128
    e.counts['proposal_attempts']+=1;e.counts['TRAIN_encoder_rows']+=2
    with e.engine.reference():
        tokens=e.engine.model.generate(**raw,do_sample=False,max_new_tokens=64,use_cache=True,pad_token_id=e.engine.pad,eos_token_id=e.engine.eos)
    response=tokens[0,raw.input_ids.shape[1]:];decoded=e.engine.tokenizer.decode(response,skip_special_tokens=True)
    box,valid=parse_box(decoded);done=bool((response==e.engine.eos).any());valid=valid and done
    if not valid:box=[0,0,1000,1000]
    e.counts['proposal_responses']+=1;e.counts['generation_responses']+=1
    torch.cuda.synchronize()
    return {'box':box,'valid':valid,'text':decoded,'truncated':not done,'input_tokens':int(raw.input_ids.numel()),
            'generated_tokens':int(response.numel()),'vision_encoder_rows':2,'seconds':time.time()-begin}


@torch.no_grad()
def recognize(e,image):
    begin=time.time();e.counts['recognition_attempts']+=1
    features,_=e.engine.encode([image]);e.counts['TRAIN_encoder_rows']+=1
    _,mask,_,text,done=e.engine.generate(features)
    lookup={e.normalize(name):c for c,name in e.names.items()};prediction=lookup.get(text[0],-1) if bool(done[0]) else -1
    e.counts['recognition_responses']+=1;e.counts['generation_responses']+=1;torch.cuda.synchronize()
    return {'prediction':prediction,'truncated':not bool(done[0]),'input_tokens':int(e.engine.prompt_ids.numel()),
            'generated_tokens':int(mask.sum()),'vision_encoder_rows':1,'seconds':time.time()-begin}


def measure(e,row,rank,split):
    image=e.images(e.train,[row])[0];results={}
    results['ROOT']={'recognition':recognize(e,image)}
    results['CENTER']={'recognition':recognize(e,crop(image,[250,250,750,750]))}
    methods=['GLOBAL_REPEAT','CENTER_DETAIL']
    if rank%2:methods.reverse()
    for method in methods:
        proposal=propose(e,image,method)
        answer=recognize(e,crop(image,proposal['box']))
        results[method]={'proposal':proposal,'recognition':answer}
    return {'global_TRAIN_row':int(row),'selection_rank':rank,'class':int(e.labels[row]),'split':split,'results':results}


def unchanged(e):
    assert all(torch.equal(a,old_a) and torch.equal(b,old_b) for (a,b),(old_a,old_b) in zip(e.engine.snapshot(),e.instrument))
    assert all(int(torch.count_nonzero(a.ref_b))==0 and not a.reference for a in e.engine.adapters)
    return True


def native(e):
    check=fixtures();e.status('NATIVE_INFORMATION_INTERFACE')
    with torch.no_grad():
        parity=[];hidden_parity=[];raw_head_parity=[]
        for row in [0,1]:
            features,raw=e.engine.encode(e.images(e.train,[row]));e.counts['TRAIN_encoder_rows']+=1
            full=e.engine.model(**raw,use_cache=False,output_hidden_states=True);e.counts['TRAIN_encoder_rows']+=1
            ids=e.engine.prompt_ids;pos,_=e.engine.positions(ids,torch.ones_like(ids))
            hidden=e.engine.model.model(inputs_embeds=e.engine.embeddings(ids,features),position_ids=pos,
                     attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
            native_last=full.hidden_states[-1][:,-1:]
            manual_last=e.engine.model.lm_head(hidden[:,-1:])[:,0].float()
            parity.append(float((e.engine.model.lm_head(native_last)[:,0].float()-manual_last).abs().max()))
            hidden_parity.append(float((native_last-hidden[:,-1:]).abs().max()))
            raw_head_parity.append(float((full.logits[:,-1].float()-manual_last).abs().max()))
        write(e.out/'NUMERICAL_PARITY.json',{'native_hidden_vs_manual_max_difference':max(hidden_parity),
              'same_shape_last_token_head_max_difference':max(parity),'raw_full_sequence_head_vs_manual_last_head_max_difference':max(raw_head_parity),
              'same_shape_threshold':.05,'native_TRAIN_encoder_rows':4,'VAL_TEST':0})
        assert max(parity)<=.05 and max(hidden_parity)<=.05
    start=time.time();records=[measure(e,row,rank,'NATIVE') for rank,row in enumerate([0,1])];elapsed=time.time()-start
    write(e.out/'NATIVE_RECORDS.private.json',records);assert unchanged(e)
    forecast=1.5*175/2*elapsed+600;remaining=e.cfg['deadline']-time.time()
    write(e.out/'NATIVE_ADMISSION.json',{'passed':forecast<remaining,'parser_fixtures':check,'recognition_native_hidden_same_head_max_difference':max(parity),
          'native_hidden_vs_manual_max_difference':max(hidden_parity),'raw_full_sequence_head_vs_manual_last_head_max_difference':max(raw_head_parity),
          'proposal_native_HF_only_no_manual_multiimage_cache':True,'generic_proposer_and_instrument_immutable':True,
          'native_responses':12,'native_TRAIN_encoder_rows':20,'VAL_TEST':0,'two_component_matrix_seconds':elapsed,
          'full_forecast_seconds_with_margin_and_reserve':forecast,'remaining_seconds':remaining,
          'native_valid_proposals':sum(r['results'][m]['proposal']['valid'] for r in records for m in ['GLOBAL_REPEAT','CENTER_DETAIL'])})
    assert forecast<remaining,'FULL_INFORMATION_MATRIX_NOT_TIME_ADMITTED'


def science(e):
    generator=torch.Generator().manual_seed(12274006);selection=[]
    for c in range(23):
        pool=torch.where(e.labels==c)[0];chosen=pool[torch.randperm(len(pool),generator=generator)[:min(8,len(pool))]]
        selection.extend({'row':int(row),'class':c,'split':'A' if k%2==0 else 'B'} for k,row in enumerate(chosen))
    assert len(selection)==len({r['row'] for r in selection})==175
    assert sum(r['split']=='A' for r in selection)==88
    assert len({e.train[r['row']]['identity_component'] for r in selection})==175
    write(e.out/'SELECTION.private.json',selection);records=[]
    for rank,item in enumerate(selection):
        e.status('MEASURING_TRAIN_INFORMATION',components_completed=rank,components_total=175)
        records.append(measure(e,item['row'],rank,item['split']))
        if rank%8==0:write(e.out/'MEASUREMENTS.private.json',records)
    write(e.out/'MEASUREMENTS.private.json',records);assert unchanged(e)
    write(e.out/'MEASUREMENTS_SEALED.json',{'epoch':time.time(),'components':175,'instrument_and_generic_reference_immutable':True,'VAL_TEST':0})
    write(e.out/'CLASS_DEFINITIONS.json',{'counts':e.class_counts.tolist(),'class_names':e.names,'cap_per_class':8})
    final=summarize(records,e.class_counts.tolist());write(e.out/'FINAL_RESULTS.json',final)
    e.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=final['all_gates_pass'])


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    e=None;cpu=time.process_time();begin=time.time()
    try:
        assert json.loads((Path(cfg['base_source'])/'CONFIG.private.json').read_text())['source_commit']=='5c0a106fe77f98b5ae15f04e953ad93cad8831e5'
        assert json.loads((Path(cfg['instrument_source'])/'CONFIG.private.json').read_text())['source_commit']=='c0d1994058e8b7a37d50a86df72e150767a138ae'
        base=load_base(cfg['base_source']);e=base.Experiment(cfg);e.normalize=base.normalize
        e.counts={'generation_responses':0,'proposal_attempts':0,'proposal_responses':0,'recognition_attempts':0,
                  'recognition_responses':0,'TRAIN_encoder_rows':0,'VAL_encoder_rows':0,'optimizer_updates':0}
        assert all(int(torch.count_nonzero(a.b))==0 for a in e.engine.adapters)
        e.engine.freeze_reference();e.engine.restore(torch.load(Path(cfg['instrument_source'])/'SEED_SFT.private.pt',map_location='cpu',weights_only=True))
        e.engine.model.eval().requires_grad_(False);e.instrument=e.engine.snapshot();assert unchanged(e)
        write(e.out/'INSTRUMENT_BINDING.json',{'recognizer':'chronologically_latest_preRL_R121_SEED_SFT','recognizer_seed':74006,
              'proposer':'generic_base_no_task_LoRA_B','generic_proposer_B_zero':True,'frozen':True,'new_optimizer_updates':0,
              'recognizer_TRAIN_in_sample':True,'selected_by_VAL':False,'VAL_TEST_access':0})
        native(e);science(e)
    finally:
        if e is not None:write(e.out/'COSTS.json',{'counts':e.counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,
               'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
               'new_feature_disk_bytes':0,'new_checkpoint_bytes':0,'historical_instrument_training_not_reexecuted':True})
