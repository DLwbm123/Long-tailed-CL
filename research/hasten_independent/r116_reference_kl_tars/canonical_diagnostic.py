"""TRAIN-only canonical-response discoverability of the fixed R114 SFT."""
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .core import write


def run(cfg):
    start,cpu=time.time(),time.process_time()
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{120})
    torch.cuda.set_per_process_memory_fraction((12*2**30-64*2**20)/torch.cuda.get_device_properties(0).total_memory)
    root=Path(cfg['base_source']);name='q114_diagnostic_dependency'
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    Engine=__import__(name+'.core',fromlist=['Engine']).Engine
    e=Engine(cfg);e.restore(torch.load(root/'SFT.private.pt',weights_only=True,map_location='cpu'))
    # Template construction only: the synthetic image is never passed to vision.
    x=e.processor(text=[e.text],images=[Image.new('RGB',(224,224))],return_tensors='pt')
    e.prompt_ids=x.input_ids.cuda();e.grid=x.image_grid_thw.cuda()
    rows=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()))
    labels=torch.tensor([int(r['original_label']) for r in rows]);counts=torch.bincount(labels,minlength=23)
    classes=torch.where(counts<25)[0].tolist();selected=[i for i,c in enumerate(labels.tolist()) if c in classes]
    assert len(selected)==56
    cache=np.memmap(root/'TRAIN_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(6821,64,2048))
    probabilities={}
    with torch.no_grad():
        for lo in range(0,len(selected),2):
            ids=selected[lo:lo+2]
            f=torch.from_numpy(np.asarray(cache[ids]).copy()).view(torch.bfloat16).cuda()
            tokens,mask=e.label_tokens([rows[i]['official_class_name'].replace('-',' ') for i in ids])
            joint=(e.token_logps(f,tokens,mask).double()*mask).sum(-1).exp().cpu().tolist()
            probabilities.update(zip(ids,joint))
    tape=torch.load(root/'QUERY_TAPE.private.pt',weights_only=True,map_location='cpu').flatten().tolist()
    occurrence={i:tape.count(i) for i in selected}
    out=[]
    for c in classes:
        ids=[i for i in selected if int(labels[i])==c];p=torch.tensor([probabilities[i] for i in ids],dtype=torch.float64)
        discovery=-torch.expm1(8*torch.log1p(-p.clamp_max(1-1e-15)))
        natural=sum(occurrence[i]*float(d) for i,d in zip(ids,discovery))
        balanced=4096/23*float(discovery.mean())
        out.append({'class':c,'name':rows[ids[0]]['official_class_name'],'TRAIN_count':len(ids),
                    'canonical_probability_min':float(p.min()),'canonical_probability_median':float(p.quantile(.5)),
                    'canonical_probability_mean':float(p.mean()),'canonical_probability_max':float(p.max()),
                    'natural_tape_queries':sum(occurrence[i] for i in ids),
                    'expected_groups_with_canonical_success_on_natural_tape_fixed_SFT_proxy':natural,
                    'expected_groups_with_canonical_success_uniform_classes_4096_fixed_SFT_proxy':balanced})
    write(Path(cfg['output'])/'CANONICAL_DISCOVERABILITY.json',{'scope':'post_hoc_R114_TRAIN_only_before_R116_VAL',
          'fixed_SFT_not_current_RL_policy':True,'rows':out,'teacher_forced_TRAIN_queries':56,
          'real_image_encoder_rows':0,'synthetic_processor_images_no_vision_forward':1,'optimizer_updates':0,'VAL_TEST':0,
          'R116_frozen_training_modified':False,'probability_boundary':'Teacher-forced exact canonical-token-sequence proxy; accepted separator variants are omitted and cached BF16 sampling differs numerically. Not an exact realized reward or mixed-group probability.',
          'counterfactual_boundary':'Uniform-class query exposure is an expectation with fixed SFT, not an executed learning method or evidence it improves accuracy.',
          'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
