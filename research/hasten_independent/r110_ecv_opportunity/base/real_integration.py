"""One CPU-only, first-arrival TRAIN integration. No scientific pilot/VAL/TEST."""
import copy
import csv
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from .memory import from_views
from .learner import Learner,optimizer,learn
from .contracts import VisibleState,DataAccessLedger
from .evaluate_controller import prepare
from .branch_env import PulseEnvironment

BOXES=[(0,0,1,1),(.125,.125,.875,.875),(0,0,.6,.6),(.4,0,1,.6),(0,.4,.6,1),(.4,.4,1,1)]


def load_generic(config):
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS
    assets=Path(config['biomed_assets']);spec=json.loads((assets/'open_clip_config.json').read_text())
    mc=copy.deepcopy(spec['model_cfg'])
    mc['text_cfg'].update(hf_model_name=str(assets/'text'),hf_tokenizer_name=str(assets),hf_model_pretrained=False)
    _MODEL_CONFIGS['generic_branch']=mc
    base,_,transform=open_clip.create_model_and_transforms('generic_branch',pretrained=None,pretrained_hf=False,precision='fp32',device='cpu',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()})
    state=torch.load(config['biomed_checkpoint'],map_location='cpu',weights_only=False);state=state.get('state_dict',state)
    key='text.transformer.embeddings.position_ids'
    if key in state and key not in base.state_dict():
        assert torch.equal(state[key],base.text.transformer.embeddings.position_ids);del state[key]
    base.load_state_dict(state,strict=True)
    visual=base.visual.eval().requires_grad_(False)
    assert not any('Random' in type(t).__name__ for t in transform.transforms)
    return visual,transform


@torch.no_grad()
def encode(config,rows,ids,visual,transform,counter):
    allowed=set(ids);z=[]
    assert len(allowed)==54, 'FROZEN_ENGINEERING_EXPOSURE'
    for lo in range(0,len(ids),6):
        group=ids[lo:lo+6];images=[]
        for i in group:
            assert i in allowed
            with Image.open(Path(config['image_root'])/rows[i]['relative_path']) as image:images.append(image.convert('RGB'))
        features=[]
        for l,t,r,b in BOXES:
            pixels=[]
            for image in images:
                w,h=image.size
                pixels.append(transform(image.crop((math.floor(w*l),math.floor(h*t),math.ceil(w*r),math.ceil(h*b)))))
            features.append(F.normalize(visual(torch.stack(pixels)).float(),dim=-1))
            counter['TRAIN_encoder_rows']=counter.get('TRAIN_encoder_rows',0)+len(group)
        z.append(torch.stack(features,1))
    return torch.cat(z)


def run(cfg):
    out=Path(cfg['output']);start=time.time();cpu=time.process_time();torch.set_num_threads(2)
    os.sched_setaffinity(0,set(cfg['cpu_affinity']))
    original=json.loads(Path(cfg['host_config']).read_text())
    assert list(map(len,original['stages']))==[5,5,5,4,4]
    rows=list(csv.DictReader((Path(original['manifest'])/'train.csv').open()))
    labels=original['order'][:3];selected=[];supports=[];queries=[]
    for label in labels:
        bygroup={}
        for i,row in enumerate(rows):
            if int(row['original_label'])==label:bygroup.setdefault(row['identity_component'],[]).append(i)
        units=sorted(bygroup)[:18];assert len(units)==18
        ids=[bygroup[u][0] for u in units]
        selected.extend(ids);supports.extend(ids[:15]);queries.extend(ids[15:])
    exposure={'split':'TRAIN','real_stage':0,'parents':54,'support_parents':45,'query_parents':9,'private_ids':selected,'support_ids':supports,'query_ids':queries,'patient_independence_verified':False}
    (out/'EXPOSURE.private.json').write_text(json.dumps(exposure,indent=2))
    counter={};visual,tf=load_generic(original);z=encode(original,rows,selected,visual,tf,counter);del visual
    labelmap={label:i for i,label in enumerate(labels)}
    bank=from_views(z,torch.tensor(selected),torch.tensor([labelmap[int(rows[i]['original_label'])] for i in selected]))
    support=bank.parents_only(torch.tensor(supports));query=bank.parents_only(torch.tensor(queries))
    groups={i:rows[i]['identity_component'] for i in selected}
    roles={groups[i]:'S' if i in supports else 'AUDIT' for i in selected}
    ledger=DataAccessLedger({i:0 for i in selected},groups,roles,0)
    model=Learner(74002);model.arrive(support,3);opt=optimizer(model)
    learn(model,opt,support,200,74003,counter,None,64,'engineering_real')
    model.trained_groups={groups[i] for i in supports}
    visible=VisibleState(support,None,(15,15,15),(0,0,0),0,256,groups,False,ledger)
    candidates,classes,actions=prepare(visible,model,opt,74002,counter)
    assert len(candidates)>1, 'NO_NONTRIVIAL_ENGINEERING_ACTION'
    # Fixed pre-result pair: ZERO and first distinct candidate, no query selection.
    pair=candidates[:2]
    env=PulseEnvironment(model,opt,visible,[],[query],set(),ledger,counter,post_steps=100,pre_steps=200,batch=64)
    outcomes=env.compare(pair,74002)
    # Legacy CLASS_STAGE fields and exactly its integer-count macro arithmetic.
    table=[]
    for arm,item in zip(['ZERO','FIRST_DISTINCT'],outcomes):
        endpoint=item['endpoints'][0]
        for c,(correct,total) in enumerate(zip(endpoint['correct'],endpoint['denominator'])):
            table.append(dict(stage=0,arm=arm,**{'class':c},original_label=labels[c],legacy_index=original['legacy_indices'][c],tiny=False,count=total,correct=correct,recall=100*correct/total))
    with (out/'CLASS_STAGE.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    reloaded=list(csv.DictReader((out/'CLASS_STAGE.csv').open()))
    for item in reloaded:assert abs(float(item['recall'])-100*int(item['correct'])/int(item['count']))<1e-12
    assert all(not x for x in classes[:,10].bool().tolist()), 'UNVERIFIED_PATIENT_STATS_PRESENTED_AS_AVAILABLE'
    result=dict(passed=True,engineering_only=True,science_updates=0,VAL_rows=0,TEST_rows=0,GPU_used=False,
                actual_stage_count=5,formal_memory_tokens=256,memory_bytes=[len(c.memory)*2064+184 for c in pair],
                parents_encoded=54,support_parents=45,query_parents=9,candidates_generated=len(candidates),fixed_candidates_executed=2,
                counts=counter,patient_independence_verified=False,split_half_available_classes=0,
                nontrivial_action_changed_parameters=outcomes[1]['parameters_differ_from_zero'],
                legacy_class_csv_readback=True,legacy_full_VAL_evaluator_executed=False,
                CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start)
    assert counter['engineering_real_classifier_optimizer']==200
    assert counter['paired_classifier_updates']==200
    assert counter['TRAIN_encoder_rows']==324
    assert result['nontrivial_action_changed_parameters']
    (out/'REAL_INTEGRATION.json').write_text(json.dumps(result,indent=2))
    return result

