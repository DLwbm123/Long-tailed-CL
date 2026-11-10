"""Semantic conditional retention with shared generation and bounded current access."""
import os,json,time,copy,importlib.util,sys
from pathlib import Path
import torch
from .core import ARMS,STAGES,write,summarize,semantic_kd,semantic_fixtures


def dependency(path):
    spec=importlib.util.spec_from_file_location('q128_dependency',Path(path)/'__init__.py',submodule_search_locations=[str(path)])
    package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
    return __import__(spec.name+'.worker',fromlist=['Run','cpu_tree'])

P=dependency(json.loads(Path(os.environ['Q_CONFIG']).read_text())['parent_source']);P.ARMS=ARMS


class Run(P.Run):
    def __init__(self,cfg):
        super().__init__({**cfg,'contrast_kappa':.2,'contrast_delta':.2,'CE_aux_coefficient':1.,'positive_KD_coefficient':1.})
        self.aux=None;self.aux_last={'loss':0.,'seconds':0.};self.tf={};self.tf_kind='base';self.semantic_records=[]
        original=self.e.engine.token_logps
        def counted(features,tokens,mask):
            row=self.tf.setdefault(self.tf_kind,{'rows':0,'input_token_cells':0,'active_response_tokens':0})
            row['rows']+=len(tokens);row['input_token_cells']+=len(tokens)*(self.e.engine.prompt_ids.shape[1]+tokens.shape[1]);row['active_response_tokens']+=int(mask.sum())
            return original(features,tokens,mask)
        self.e.engine.token_logps=counted
        self.counts.update({'semantic_anchor_TF_rows':0,'semantic_actor_eval_TF_rows':0,'semantic_actor_backward_TF_rows':0})

    def score_semantic(self,features,classes,kind,profile_length=None):
        before=self.tf_kind;self.tf_kind=kind;rows=[]
        try:
            for i in range(len(features)):
                values=[]
                for lo in range(0,len(classes),2):
                    chunk=classes[lo:lo+2];tok,mask=self.e.engine.label_tokens([self.e.names[c] for c in chunk])
                    if profile_length is not None:
                        tok=tok[:,:1].expand(len(chunk),profile_length).contiguous();tok[:,-1]=self.e.engine.eos;mask=torch.ones_like(tok,dtype=torch.bool)
                    lp=self.e.engine.token_logps(features[i:i+1].expand(len(chunk),-1,-1),tok,mask)
                    values.append((lp*mask).sum(-1));self.counts[kind+'_TF_rows']+=len(chunk)
                rows.append(torch.cat(values))
            return torch.stack(rows)
        finally:self.tf_kind=before

    def backward_semantic(self,features,classes,coeff,profile_length=None):
        before=self.tf_kind;self.tf_kind='semantic_actor_backward'
        try:
            for i in range(len(features)):
                for lo in range(0,len(classes),2):
                    chunk=classes[lo:lo+2];tok,mask=self.e.engine.label_tokens([self.e.names[c] for c in chunk])
                    if profile_length is not None:
                        tok=tok[:,:1].expand(len(chunk),profile_length).contiguous();tok[:,-1]=self.e.engine.eos;mask=torch.ones_like(tok,dtype=torch.bool)
                    lp=self.e.engine.token_logps(features[i:i+1].expand(len(chunk),-1,-1),tok,mask)
                    (coeff[i,lo:lo+len(chunk)]*(lp*mask).sum(-1)).sum().backward();self.counts['semantic_actor_backward_TF_rows']+=len(chunk)
        finally:self.tf_kind=before

    def step(self,opt):
        self.aux_last={'loss':0.,'seconds':0.}
        if self.aux is not None:
            f,classes,target,parts=self.aux;start=time.time()
            with torch.no_grad():scores=self.score_semantic(f,classes,'semantic_actor_eval');loss,coeff=semantic_kd(scores,target,parts)
            self.backward_semantic(f,classes,coeff);self.aux_last={'loss':float(loss),'seconds':time.time()-start}
        return super().step(opt)

    def targets(self,current):
        classes=sum(STAGES[:self.stage],[]);values=[];started=time.time();actor=self.e.engine.snapshot()
        for lo in range(0,len(current),16):
            if lo%64==0:self.status('BUILD_CURRENT_INPUT_OLD_SEMANTIC_TARGET',rows_done=lo,rows_total=len(current))
            f=self.get('TRAIN',current[lo:lo+16],'current_semantic_teacher')
            with torch.no_grad(),self.e.engine.reference():values.append(self.score_semantic(f,classes,'semantic_anchor').cpu())
        assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),actor))
        scores=torch.cat(values);assert scores.shape==(len(current),len(classes))
        self.checkpoint({'current_ids':current,'old_classes':classes,'scores':scores},f'SEMANTIC_TARGETS_{self.stage}_{self.arm}.private.pt')
        self.semantic_records.append({'stage':self.stage,'arm':self.arm,'current_rows':len(current),'old_class_count':len(classes),'TF_rows':len(current)*len(classes),'seconds':time.time()-started,'anchor_same_current_prompt':True,'actor_restored':True,'old_TRAIN_access':False})
        return classes,scores,{v:i for i,v in enumerate(current)}

    def partitions(self,stage):
        return [list(range(sum(map(len,STAGES[:stage]))))] if self.arm=='SEM_GKD_TARS_RAPO' else [list(range(sum(map(len,STAGES[:s])),sum(map(len,STAGES[:s+1])))) for s in range(stage)]

    def native(self):
        self.status('NATIVE_SEMANTIC_CONDITIONAL_RETENTION');write(self.out/'SEMANTIC_FIXTURES.json',semantic_fixtures())
        ids=[i for i,c in enumerate(self.y) if c in STAGES[0]][:16];first=ids[:2];expected=self.e.engine.prompt_ids.clone()
        self.access.check('TRAIN',first,0,'NATIVE','native_images');features,raw=self.e.engine.encode(self.e.images(self.e.train,first));self.counts['TRAIN_encoder_rows']+=2
        assert torch.equal(expected,self.e.engine.prompt_ids)
        with torch.no_grad():
            full=self.e.engine.model(**raw,use_cache=False,output_hidden_states=True);self.counts['TRAIN_encoder_rows']+=2
            prompt=self.e.engine.prompt_ids.expand(2,-1);pos,_=self.e.engine.positions(prompt,torch.ones_like(prompt))
            hidden=self.e.engine.model.model(inputs_embeds=self.e.engine.embeddings(prompt,features),position_ids=pos,attention_mask=torch.ones_like(prompt),use_cache=False,return_dict=True).last_hidden_state
            head=self.e.engine.model.lm_head(hidden[:,-1:])[:,0].float();hdiff=float((hidden[:,-1:]-full.hidden_states[-1][:,-1:]).abs().max());pdiff=float((head-self.e.engine.model.lm_head(full.hidden_states[-1][:,-1:])[:,0].float()).abs().max());rawdiff=float((head-full.logits[:,-1].float()).abs().max())
        assert hdiff<=.05 and pdiff<=.05
        max_answer=max(len(self.e.engine.tokenizer.encode(name,add_special_tokens=False))+1 for name in self.e.names.values());assert max_answer<=24
        self.bind_prompt(0,profiling=True);unit=torch.ones(23,device='cuda');records=[];non_sem=[]
        for arm in ARMS:
            self.arm=arm;self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();self.aux=None
            f=self.get('TRAIN',first,'native_training');labels=self.labels(first)
            if arm!='BASE_TARS_RAPO':
                with torch.no_grad(),self.e.engine.reference():target=self.score_semantic(f,STAGES[0],'semantic_anchor')
                self.aux=(f,STAGES[0],target,[list(range(5))])
            start=time.time();self.counts['engineering_optimizer_attempts']+=1
            if arm.endswith('_CE'):
                self.counts['engineering_teacherforced_queries']+=2;rec=self.ce(f,labels,self.opt(),unit)
            else:rec,_=self.rl(f,labels,self.opt(),unit,True,1.,124000,'engineering_responses')
            elapsed=time.time()-start;non_sem.append(elapsed-self.aux_last['seconds']);records.append({'arm':arm,'seconds':elapsed,'semantic_seconds':self.aux_last['seconds'],'semantic_loss':self.aux_last['loss'],**rec})
        self.aux=None;self.arm='NATIVE_SFT_PATH';self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();self.counts['engineering_optimizer_attempts']+=1;self.counts['engineering_teacherforced_queries']+=16
        ce=self.ce(self.get('TRAIN',ids,'native_training'),self.labels(ids),self.opt(),unit)
        self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();f=self.get('TRAIN',first,'native_semantic_profile');classes=[i%5 for i in range(19)]
        torch.cuda.synchronize();start=time.time()
        with torch.no_grad(),self.e.engine.reference():target=self.score_semantic(f,classes,'semantic_anchor',max_answer)
        torch.cuda.synchronize();anchor=time.time()-start;start=time.time()
        with torch.no_grad():scores=self.score_semantic(f,classes,'semantic_actor_eval',max_answer);_,coeff=semantic_kd(scores,target.roll(1,-1),[list(range(19))])
        torch.cuda.synchronize();actor_eval=time.time()-start;start=time.time();self.opt().zero_grad(set_to_none=True);self.backward_semantic(f,classes,coeff,max_answer);torch.cuda.synchronize();actor_backward=time.time()-start
        profile_grad=float(torch.nn.utils.clip_grad_norm_(self.e.engine.parameters,1.));assert profile_grad>0;self.opt().zero_grad(set_to_none=True)
        start=time.time();self.generate(self.get('TRAIN',ids,'native_profile'),'engineering_responses');greedy=time.time()-start
        assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),self.prefix));assert all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in self.e.engine.adapters)
        forecast=1.5*(512*4*sum(non_sem)+131694*anchor/38+150528*(actor_eval+actor_backward)/38+(17548+22812)/16*greedy)+1800
        proof={'passed':forecast<self.cfg['deadline']-time.time(),'matched_hidden_difference':hdiff,'matched_head_difference':pdiff,'raw_full_head_difference':rawdiff,'native_paths':records+[{'arm':'NATIVE_SFT_PATH',**ce}],'all_native_updates_discarded':True,'semantic_profile_gradient_norm':profile_grad,'semantic_profile_shifted_target_engineering_only':True,'profile_19_answers_only_firsttask_vocabulary':True,'max_answer_profile_tokens_CPU_only_future_length':max_answer,'student_profile':self.profile_prompt,'anchor38_seconds':anchor,'actor_eval38_seconds':actor_eval,'actor_backward38_seconds':actor_backward,'non_semantic_2query_seconds':non_sem,'greedy16_seconds':greedy,'original_forecast_seconds':forecast,'remaining_seconds':self.cfg['deadline']-time.time(),'native_generation_responses':64,'native_TRAIN_encodings':4,'native_optimizer_attempts':5,'native_semantic_rows_per_kind':68,'native_response_sampling_TF_differences_preserved':[x.get('sampled_vs_TF_logp_max_difference') for x in records]}
        write(self.out/'NATIVE_ADMISSION.json',proof);assert proof['passed'],'FULL_MATRIX_NOT_TIME_ADMITTED'
        self.bind_prompt(0);self.arm='COMMON'

    def science(self):
        source=Path(self.cfg['prefix_source']);initial=next(x for x in json.loads((source/'EVALUATIONS.json').read_text()) if x['stage']==0)
        self.evaluations=[{**copy.deepcopy(initial),'arm':a,'reused_historical_prefix_readout':True} for a in ARMS]
        write(self.out/'STAGE_0_ALL_MODELS_SEALED.json',{'epoch':time.time(),'arms':ARMS,'historical_prefix_readout_reused':True})
        states={a:{'weights':P.cpu_tree(self.prefix),'optimizer':None,'sigma':1.,'updates':0} for a in ARMS};self.save_logs()
        for stage in range(1,5):
            self.stage=stage;self.bind_prompt(stage);current=[i for i,c in enumerate(self.y) if c in STAGES[stage]]
            g=torch.Generator().manual_seed(12474005+stage);tape=torch.tensor(current)[torch.randint(len(current),(128,8),generator=g)];torch.save(tape,self.out/f'QUERY_TAPE_{stage}.private.pt');order=ARMS[(stage-1)%4:]+ARMS[:(stage-1)%4]
            for arm in order:
                self.arm=arm;prior=states[arm];self.e.engine.restore(prior['weights']);self.e.engine.freeze_reference();self.aux=None;opt=self.opt()
                if prior['optimizer'] is not None:opt.load_state_dict(prior['optimizer'])
                assert not opt.state or {int(x['step']) for x in opt.state.values()}=={prior['updates']}
                self.bindings.append({'stage':stage,'arm':arm,'optimizer_steps_before':prior['updates'],'sigma_before':prior['sigma'],'previous_weights_restored':all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),prior['weights'])),'anchor_from_own_previous_weights':all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in self.e.engine.adapters),'old_TRAIN_access_allowed':False})
                weights=self.calibration(current);sigma=prior['sigma']
                if arm!='BASE_TARS_RAPO':classes,targets,lookup=self.targets(current)
                for k in range(128):
                    self.current_update=k;self.status('CONTINUAL_SEMANTIC_CONDITIONAL_RETENTION',update=k,total=128);batch=tape[k].tolist();f=self.get('TRAIN',batch);labels=self.labels(batch)
                    self.aux=None if arm=='BASE_TARS_RAPO' else (f,classes,targets[[lookup[i] for i in batch]].to('cuda'),self.partitions(stage))
                    self.counts['post_optimizer_attempts']+=1
                    if arm.endswith('_CE'):
                        self.counts['post_CE_queries']+=8;rec=self.ce(f,labels,opt,weights);rec.update(sigma_before=sigma,sigma_after=sigma,class_groups={})
                    else:rec,sigma=self.rl(f,labels,opt,weights,True,sigma,12480000+1000*stage+k)
                    rec.update(semantic_KD_loss=self.aux_last['loss'],semantic_KD_seconds=self.aux_last['seconds']);self.counts['post_completed_updates']+=1;self.curves.append({'phase':'POST','stage':stage,'arm':arm,'update':k,**rec})
                    if k%16==0:self.save_logs()
                self.aux=None;state={'weights':self.e.engine.snapshot(),'optimizer':P.cpu_tree(opt.state_dict()),'sigma':sigma,'updates':prior['updates']+128};states[arm]=state;self.checkpoint(state,f'STATE_{stage}_{arm}.private.pt');del opt
            if stage==1:
                same=all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(states['SEM_GKD_TARS_RAPO']['weights'],states['SEM_TKD_TARS_RAPO']['weights']))
                write(self.out/'FIRST_POST_IDENTITY.json',{'one_old_task_GKD_TKD_weights_exact':same});assert same,'FIRST_POST_GKD_TKD_IDENTITY'
            write(self.out/f'STAGE_{stage}_ALL_MODELS_SEALED.json',{'epoch':time.time(),'arms':ARMS,'historical_TRAIN_replay':0})
            for arm in ARMS:self.arm=arm;self.e.engine.restore(states[arm]['weights']);self.evaluations.append(self.evaluate(arm,True))
            self.save_logs()
        final=summarize(self.evaluations,self.e.class_counts.tolist());write(self.out/'FINAL_RESULTS.json',final);self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=final['all_gates_pass'])

    def save_logs(self):
        super().save_logs()
        if hasattr(self,'semantic_records'):write(self.out/'SEMANTIC_TARGET_BINDINGS.json',self.semantic_records);write(self.out/'TF_COSTS.json',self.tf)


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    begin=time.time();cpu=time.process_time();r=None
    try:r=Run(cfg);r.native();r.science()
    finally:
        if r is not None:r.save_logs();write(r.out/'COSTS.json',{'counts':r.counts,'TF':r.tf,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_feature_model_copy_bytes':0,'zero_old_TRAIN_replay_by_accessor':True})
        else:write(Path(cfg['output'])/'LOAD_FAILURE_COSTS.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,'generation_responses':0,'optimizer_attempts':0,'original_clock_not_reset':True})
