"""Retrospective frozen model readouts on a fixed old-VAL subset."""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .core import ARMS,READOUTS,STAGES,question,old_access,metrics,write


def dependency(path):
    spec=importlib.util.spec_from_file_location('instrument',Path(path)/'__init__.py',submodule_search_locations=[str(path)])
    package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
    return __import__('instrument.worker',fromlist=['Experiment'])


class Run:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['output']);self.source=Path(cfg['saved_source']);self.base=dependency(cfg['base_source']);self.e=self.base.Experiment(cfg)
        self.val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()));self.labels=[int(r['original_label']) for r in self.val]
        self.ids=[i for i,y in enumerate(self.labels) if y in STAGES[0]];assert len(self.ids)==624
        self.counts={'generation_responses':0,'generation_attempts':0,'cached_feature_forward_rows':0,'native_extra_cached_forward_rows':0,'prefill_input_tokens':0,'generated_active_tokens':0,'decoded_token_cells':0,'decoder_forward_token_rows':0,'TRAIN_cached_feature_rows':0,'TRAIN_encoder_rows':0,'VAL_encoder_rows':0,'optimizer_updates':0,'SFT_updates':0}
        cache=Path(cfg['feature_source'])/'VAL_FEATURES.private.bin';assert cache.stat().st_size==1698*64*2048*2
        self.cache=np.memmap(cache,mode='r',dtype=np.uint16,shape=(1698,64,2048));self.journal=[];self.bindings=[];self.readouts=[];self.seeded={}
        self.common=torch.load(self.source/'COMMON_PREFIX_SFT.private.pt',map_location='cpu',weights_only=True)
        self.e.engine.restore(self.common);self.e.engine.freeze_reference()
        self.e.engine.model.requires_grad_(False)
        assert not any(p.requires_grad for p in self.e.engine.model.parameters())
        self.original=torch.load(self.source/'VAL_0_COMMON.private.pt',map_location='cpu',weights_only=True);assert self.original['ids']==self.ids
        self.prompts=json.loads((self.source/'PROMPT_BINDINGS.json').read_text());self.bind(0)
        self.model_name='COMMON';self.menu=0

    def bind(self,stage):
        engine=self.e.engine;q=question(stage,self.e.names);assert q==self.prompts[str(stage)]['question']
        text=engine.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':q}]}],tokenize=False,add_generation_prompt=True)
        assert text.count('<|image_pad|>')==1
        engine.text=text;engine.prompt_ids=engine.tokenizer(text.replace('<|image_pad|>','<|image_pad|>'*64),return_tensors='pt').input_ids.cuda()
        engine.grid=torch.tensor([[1,16,16]],device='cuda');assert engine.prompt_ids.shape[1]==self.prompts[str(stage)]['input_tokens']
        self.menu=stage;self.bindings.append({'stage_menu':stage,'question':q,'input_tokens':engine.prompt_ids.shape[1],'visual_tokens':64})

    def features(self,ids,kind,uses=1):
        ids=old_access(ids,self.labels);self.journal.append({'ids':ids,'model':self.model_name,'stage_menu':self.menu,'kind':kind,'uses':uses,'epoch':time.time()})
        self.counts['cached_feature_forward_rows']+=len(ids)*uses
        if kind=='native_forward_parity':self.counts['native_extra_cached_forward_rows']+=len(ids)*uses
        return torch.from_numpy(np.asarray(self.cache[ids]).copy()).view(torch.bfloat16).cuda()

    def generate(self,ids,kind):
        t=time.time();self.counts['generation_attempts']+=len(ids)
        _,mask,_,text,done=self.e.engine.generate(self.features(ids,kind),sample=False)
        self.counts['generation_responses']+=len(ids);self.counts['prefill_input_tokens']+=len(ids)*self.e.engine.prompt_ids.shape[1]
        self.counts['generated_active_tokens']+=int(mask.sum());self.counts['decoded_token_cells']+=mask.numel();self.counts['decoder_forward_token_rows']+=len(ids)*(mask.shape[1]-1)
        lookup={self.base.normalize(n):c for c,n in self.e.names.items()};pred=[lookup.get(t,-1) if bool(d) else -1 for t,d in zip(text,done)]
        return pred,time.time()-t

    def status(self,phase,**kw):
        assert time.time()<self.cfg['deadline']-120
        write(self.root/'STATUS.json',{'phase':phase,'epoch':time.time(),'counts':self.counts,**kw})

    def native(self):
        self.status('NATIVE_FROZEN_READOUT')
        e=self.e.engine;f=self.features(self.ids[:2],'native_forward_parity',2);ids=e.prompt_ids.expand(2,-1);attention=torch.ones_like(ids);pos,_=e.positions(ids,attention)
        with torch.no_grad():
            full=e.model(inputs_embeds=e.embeddings(ids,f),position_ids=pos,attention_mask=attention,use_cache=False,output_hidden_states=True)
            hidden=e.model.model(inputs_embeds=e.embeddings(ids,f),position_ids=pos,attention_mask=attention,use_cache=False,return_dict=True).last_hidden_state
            h=float((hidden[:,-1:]-full.hidden_states[-1][:,-1:]).abs().max())
            head=e.model.lm_head(hidden[:,-1:])[:,0].float();matched=float((head-e.model.lm_head(full.hidden_states[-1][:,-1:])[:,0].float()).abs().max());raw=float((head-full.logits[:,-1].float()).abs().max())
        assert h<=.05 and matched<=.05
        first,t0=self.generate(self.ids[:16],'native_reused');assert first==self.original['predictions'][:16],'INITIAL_FIRST16_PREDICTION_PARITY'
        self.seeded['COMMON_INITIAL_MENU']=(first,t0)
        self.bind(4);last,t4=self.generate(self.ids[:16],'native_reused');self.seeded['COMMON_FINAL_MENU']=(last,t4)
        forecast=1.5*7*39*max(t0,t4)+600
        proof={'passed':forecast<self.cfg['deadline']-time.time(),'same_hidden_parity':h,'same_shape_head_parity':matched,'raw_full_head_difference':raw,'first16_saved_prediction_parity':True,'first16_initial_menu_seconds':t0,'first16_final_menu_seconds':t4,'forecast_seconds':forecast,'remaining_seconds':self.cfg['deadline']-time.time(),'native_generation_rows_reused':32,'native_extra_cached_forward_rows':4,'new_visual_encoder_rows':0,'optimizer_updates':0,'retrospective_all_classes_already_arrived':True}
        write(self.root/'NATIVE_ADMISSION.json',proof);assert proof['passed'],'ORIGINAL_CLOCK_TIME_NOT_ADMITTED'

    def science(self):
        for name,arm,menu in READOUTS:
            self.model_name=arm
            state=self.common if arm=='COMMON' else torch.load(self.source/f'STATE_4_{arm}.private.pt',map_location='cpu',weights_only=True)['weights']
            self.e.engine.restore(state);self.e.engine.freeze_reference();self.bind(menu)
            assert all(torch.equal(a,u) and torch.equal(b,v) for (a,b),(u,v) in zip(self.e.engine.snapshot(),state))
            self.status('FIXED_READOUT',readout=name)
            prefix,prefix_wall=self.seeded.get(name,([],0.));pred=list(prefix);wall=prefix_wall;started=time.time()
            for lo in range(len(pred),624,16):
                self.status('FIXED_READOUT',readout=name,rows_done=lo)
                row,seconds=self.generate(self.ids[lo:lo+16],'readout');pred+=row;wall+=seconds
            assert len(pred)==624
            if name=='COMMON_INITIAL_MENU':assert pred==self.original['predictions'],'FULL_INITIAL624_PREDICTION_PARITY'
            assert all(torch.equal(a,u) and torch.equal(b,v) for (a,b),(u,v) in zip(self.e.engine.snapshot(),state))
            saved={'ids':self.ids,'predictions':pred};torch.save(saved,self.root/f'PRED_{name}.private.pt')
            result={'readout':name,'model':arm,'stage_menu':menu,'weights_equal_loaded_before_and_after':True,'no_trainable_parameters':True,'generation_wall_seconds':wall,'readout_started_epoch':started,**metrics(pred,[self.labels[i] for i in self.ids],self.e.class_counts.tolist())}
            self.readouts.append(result);write(self.root/'READOUT_RESULTS.json',self.readouts)
        assert self.counts['generation_responses']==4368 and self.counts['cached_feature_forward_rows']==4372
        assert self.counts['generation_attempts']==4368
        write(self.root/'MEASUREMENTS_SEALED.json',{'epoch':time.time(),'readouts':7,'generation_responses':4368,'cached_feature_forward_rows':4372,'all_parameters_immutable':True,'no_optimizer_created':True})
        self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION')


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    start=time.time();cpu=time.process_time();r=None
    try:r=Run(cfg);r.native();r.science()
    finally:
        if r is not None:
            write(r.root/'ACCESS.private.json',r.journal);write(r.root/'PROMPT_BINDINGS.json',r.bindings)
            write(r.root/'COSTS.json',{'counts':r.counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'new_weight_or_feature_copies':0,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        else:write(Path(cfg['output'])/'LOAD_FAILURE_COSTS.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'original_clock_not_reset':True})
