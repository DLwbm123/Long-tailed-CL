"""Current-task-only generative continual training with independent arm state."""
import copy
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .core import ARMS,STAGES,Access,fixtures,rewards_and_advantages,score,summarize,weights_for_current,write,stage_prompt,PROMPT_TEMPLATE,interface_fixtures,contrast_fixtures,teacher_negatives,contrast_advantages,token_surrogate,TEACHER_TEMPLATE


def load_base(path):
    spec=importlib.util.spec_from_file_location('q114_dependency',Path(path)/'__init__.py',submodule_search_locations=[str(path)])
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return __import__(spec.name+'.worker',fromlist=['Experiment','normalize','surrogate'])


def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)


class Run:
    def __init__(self,cfg):
        assert cfg['contrast_kappa']==.2 and cfg['contrast_delta']==.2 and cfg['CE_aux_coefficient']==cfg['positive_KD_coefficient']==1.
        self.cfg=cfg;self.out=Path(cfg['output']);self.base=load_base(cfg['base_source']);self.e=self.base.Experiment(cfg)
        self.val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
        self.y=[int(r['original_label']) for r in self.e.train];self.vy=[int(r['original_label']) for r in self.val]
        assert len(self.y)==6821 and len(self.vy)==1698 and len({r['identity_component'] for r in self.e.train})==6821
        self.access=Access(self.y,self.vy);self.curves=[];self.evaluations=[];self.seals={};self.group_rows=[];self.bindings=[]
        self.counts={'engineering_optimizer_attempts':0,'engineering_teacherforced_queries':0,'SFT_updates':0,'SFT_queries':0,
            'post_optimizer_attempts':0,'post_CE_queries':0,'post_RL_responses':0,'calibration_responses':0,
            'VAL_responses':0,'engineering_responses':0,'generation_responses':0,'generation_attempts':0,
            'SFT_optimizer_attempts':0,'post_completed_updates':0,'TRAIN_encoder_rows':0,'VAL_encoder_rows':0}
        self.counts.update({'CE_aux_queries':0,'positive_teacher_rows':0,'negative_teacher_rows':0,'teacher_TF_input_token_cells':0,'teacher_active_response_tokens':0,'generation_input_tokens':0,'generated_active_tokens':0,'decoded_token_cells':0,'decoder_forward_token_rows':0,'native_profile_teacher_rows':0})
        self.rollout_records=[];self.teacher_binding_stats={};self.teacher_sync_step=0
        self.maps={}
        for split,n in [('TRAIN',6821),('VAL',1698)]:
            path=Path(cfg['feature_source'])/f'{split}_FEATURES.private.bin'
            assert path.stat().st_size==n*64*2048*2
            self.maps[split]=np.memmap(path,mode='r',dtype=np.uint16,shape=(n,64,2048))
        self.stage=0;self.arm='NATIVE';self.latest_sigma=1.
        assert all(int(torch.count_nonzero(a.b))==0 for a in self.e.engine.adapters)
        source=Path(cfg['prefix_source'])
        prefix_proof=json.loads((source/'FINAL_VERIFICATION.json').read_text());assert prefix_proof['passed']
        provenance=json.loads((source/'GENERIC_INITIALIZATION.json').read_text());assert provenance['task_LoRA_B_zero'] and not provenance['old_task_SFT_restored']
        self.prefix=torch.load(source/'COMMON_PREFIX_SFT.private.pt',map_location='cpu',weights_only=True)
        self.generic=cpu_tree(self.prefix);self.teacher_current=cpu_tree(self.prefix)
        self.e.engine.restore(self.prefix);self.e.engine.freeze_reference()
        assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),self.prefix))
        self.prompt_records={};self.bind_prompt(0)
        write(self.out/'PREFIX_INITIALIZATION.json',{'new_engine_generic_B0_before_restore':True,'R125_verified_chronological_first_task_only_prefix':True,'prefix_weights_exact_after_restore':True,'no_full_TRAIN_SFT_or_VAL_selection':True,'new_SFT_updates':0,'historical_SFT_updates':306,'historical_SFT_queries':4868,'historical_initial_VAL_responses':624,'source_runtime':'aedf8075657e8feafe013439345acb9f893b3bee','future_image_feature_feedback_to_model':False,'historical_prefix_CPU_not_separately_instrumented_not_zero':True})

    def bind_prompt(self,stage,profiling=False):
        assert self.cfg['prompt_template']==PROMPT_TEMPLATE
        engine=self.e.engine;seen=[c for task in STAGES[:stage+1] for c in task]
        question=stage_prompt(stage,self.e.names)
        def rendered(q):
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':q}]}]
            return engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        def tokens(text):
            assert text.count('<|image_pad|>')==1
            return engine.tokenizer(text.replace('<|image_pad|>','<|image_pad|>'*64),return_tensors='pt').input_ids
        maximum=max(tokens(rendered(stage_prompt(s,self.e.names))).shape[1] for s in range(5))
        # Only known first-task words and neutral padding enter the model during admission.
        if profiling:
            assert stage==0
            while tokens(rendered(question)).shape[1]<maximum:question+=' Category'
        self.student_question=question;self.profiling=profiling
        self.teacher_max_tokens=max(tokens(rendered(TEACHER_TEMPLATE.format(answer=self.e.names[c],student_question=stage_prompt(s,self.e.names)))).shape[1] for s in range(5) for c in [x for task in STAGES[:s+1] for x in task])
        engine.text=rendered(question);engine.prompt_ids=tokens(engine.text).cuda()
        engine.grid=torch.tensor([[1,16,16]],device='cuda')
        assert int((engine.prompt_ids==engine.model.config.image_token_id).sum())==64
        self.prompt_stage=stage
        record={'stage':stage,'classes':seen,'question':question,'input_tokens':engine.prompt_ids.shape[1],
                'visual_tokens':64,'training_generation_reference_and_audit_share_binding':True,
                'future_image_or_feedback_used':False,'engineering_profile':profiling}
        if profiling:
            record['longest_schema_prompt_tokens']=maximum
            assert record['input_tokens']>=maximum
            self.profile_prompt=record
        else:
            self.prompt_records[str(stage)]=record
            write(self.out/'PROMPT_BINDINGS.json',self.prompt_records)

    def status(self,phase,**extra):
        assert time.time()<self.cfg['deadline']-120,'ORIGINAL_CLOCK'
        write(self.out/'STATUS.json',{'phase':phase,'epoch':time.time(),'stage':self.stage,'arm':self.arm,'counts':self.counts,**extra})

    def get(self,split,ids,purpose='current_training',sealed=False):
        assert self.prompt_stage==self.stage
        ids=self.access.check(split,ids,self.stage,self.arm,purpose,sealed)
        return torch.from_numpy(np.asarray(self.maps[split][ids]).copy()).view(torch.bfloat16).cuda()

    def step(self,opt):
        params=self.e.engine.parameters;before=[p.detach().clone() for p in params]
        norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
        opt.step();delta=float(torch.stack([(p.detach()-q).square().sum() for p,q in zip(params,before)]).sum().sqrt())
        return float(norm),delta

    def ce(self,features,labels,opt,weights):
        opt.zero_grad(set_to_none=True);total=0.
        for lo in range(0,len(labels),2):
            y=labels[lo:lo+2];tok,mask=self.e.engine.label_tokens([self.e.names[int(c)] for c in y])
            lp=self.e.engine.token_logps(features[lo:lo+2],tok,mask)
            loss=-((lp*mask).sum(-1)/mask.sum(-1)*weights[y.to('cuda')]).sum()/len(labels)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        norm,delta=self.step(opt);return {'loss':total,'gradient_norm':norm,'parameter_L2':delta,'nonzero_update':delta>0}

    def generate(self,features,kind,generator=None,sample=False):
        n=len(features);self.counts['generation_attempts']+=n
        write(self.out/'GENERATION_RECEIPT.json',{'inflight':True,'kind':kind,'rows':n,'counts':self.counts,'epoch':time.time()})
        result=self.e.engine.generate(features,generator=generator,sample=sample)
        self.counts['generation_input_tokens']+=n*self.e.engine.prompt_ids.shape[1]
        self.counts['generated_active_tokens']+=int(result[1].sum());self.counts['decoded_token_cells']+=result[1].numel();self.counts['decoder_forward_token_rows']+=n*(result[1].shape[1]-1)
        self.counts['generation_responses']+=n;self.counts[kind]+=n
        write(self.out/'GENERATION_RECEIPT.json',{'inflight':False,'kind':kind,'rows':n,'counts':self.counts,'epoch':time.time()})
        return result

    def teacher_logps(self,features,tok,mask,answers,branch,profile=False):
        begin=time.time();engine=self.e.engine;actor=engine.snapshot();prompt=engine.prompt_ids;text=engine.text
        versions=[(a.ref_a._version,a.ref_b._version) for a in engine.adapters]
        engine.restore(self.teacher_current);result=[]
        try:
            with torch.no_grad():
                for i,answer in enumerate(answers):
                    q=TEACHER_TEMPLATE.format(answer=answer,student_question=self.student_question)
                    def ids_for(value):
                        rendered=engine.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':value}]}],tokenize=False,add_generation_prompt=True)
                        assert rendered.count('<|image_pad|>')==1
                        return engine.tokenizer(rendered.replace('<|image_pad|>','<|image_pad|>'*64),return_tensors='pt').input_ids
                    ids=ids_for(q)
                    if profile:
                        while ids.shape[1]<self.teacher_max_tokens:q+=' Category';ids=ids_for(q)
                    engine.prompt_ids=ids.cuda()
                    result.append(engine.token_logps(features[i:i+1],tok[i:i+1],mask[i:i+1]))
                    self.counts[branch+'_teacher_rows']+=1
                    self.counts['teacher_TF_input_token_cells']+=ids.shape[1]+tok.shape[1]
                    self.counts['teacher_active_response_tokens']+=int(mask[i].sum())
                    if profile:self.counts['native_profile_teacher_rows']+=1
                    key=f'{self.stage}:{self.arm}:{branch}';row=self.teacher_binding_stats.setdefault(key,{'stage':self.stage,'arm':self.arm,'branch':branch,'rows':0,'input_tokens_min':ids.shape[1],'input_tokens_max':ids.shape[1]})
                    row['rows']+=1;row['input_tokens_min']=min(row['input_tokens_min'],ids.shape[1]);row['input_tokens_max']=max(row['input_tokens_max'],ids.shape[1])
        finally:
            engine.restore(actor);engine.prompt_ids=prompt;engine.text=text
        assert versions==[(a.ref_a._version,a.ref_b._version) for a in engine.adapters]
        assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(engine.snapshot(),actor))
        return torch.cat(result).detach(),time.time()-begin

    def ce_accumulate(self,features,labels,weights):
        total=0.
        for lo in range(0,len(labels),2):
            y=labels[lo:lo+2];tok,mask=self.e.engine.label_tokens([self.e.names[int(c)] for c in y])
            lp=self.e.engine.token_logps(features[lo:lo+2],tok,mask)
            loss=-((lp*mask).sum(-1)/mask.sum(-1)*weights[y.to('cuda')]).sum()/len(labels)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        return total

    def rl(self,features,labels,opt,weights,rapo,sigma,seed,kind='post_RL_responses'):
        n=len(labels);g=torch.Generator(device='cuda').manual_seed(seed);f=features.repeat_interleave(8,0)
        tok,mask,sampled_lp,text,done=self.generate(f,kind,g,True)
        targets=[self.base.normalize(self.e.names[int(c)]) for c in labels for _ in range(8)]
        task=torch.tensor([int(t==y and bool(d)) for t,y,d in zip(text,targets,done)],device='cuda',dtype=torch.float32).reshape(n,8)
        seen=set(c for cs in STAGES[:self.stage+1] for c in cs)
        parsed,negatives=teacher_negatives(text,done.tolist(),labels.tolist(),self.e.names,seen)
        eligible=torch.tensor([c>=0 for c in negatives],device='cuda');old=[];ref=[]
        with torch.no_grad():
            for lo in range(0,n*8,2):old.append(self.e.engine.token_logps(f[lo:lo+2],tok[lo:lo+2],mask[lo:lo+2]))
            with self.e.engine.reference():
                for lo in range(0,n*8,2):ref.append(self.e.engine.token_logps(f[lo:lo+2],tok[lo:lo+2],mask[lo:lo+2]))
        old=torch.cat(old).detach();ref=torch.cat(ref).detach()
        adv,ret,total,drift,new_sigma=rewards_and_advantages(task,old,ref,mask,True,sigma)
        tokenadv=adv.flatten()[:,None].expand_as(old).clone();teacher_seconds=0.;positive_rows=negative_rows=0;evidence=torch.zeros_like(old)
        if self.arm=='POS_KD_TARS_RAPO':
            answers=[self.e.names[int(c)] for c in labels for _ in range(8)]
            positive,seconds=self.teacher_logps(f,tok,mask,answers,'positive');teacher_seconds+=seconds;positive_rows=n*8
            evidence=(positive-old).detach();tokenadv=tokenadv+evidence
        elif self.arm=='CONTRAST_TARS_RAPO' and bool(eligible.any()):
            chosen=eligible.nonzero().flatten();gold=[self.e.names[int(labels[int(i)//8])] for i in chosen];wrong=[self.e.names[negatives[int(i)]] for i in chosen]
            positive,seconds=self.teacher_logps(f[chosen],tok[chosen],mask[chosen],gold,'positive');teacher_seconds+=seconds
            negative,seconds=self.teacher_logps(f[chosen],tok[chosen],mask[chosen],wrong,'negative');teacher_seconds+=seconds
            evidence[chosen]=positive-negative;positive_rows=negative_rows=len(chosen)
            tokenadv=contrast_advantages(adv,evidence,eligible)
        opt.zero_grad(set_to_none=True);loss_sum=0.;pgs=[];kls=[];ratio_error=0.
        w=weights[labels.to('cuda')].repeat_interleave(8)
        for lo in range(0,n*8,2):
            lp=self.e.engine.token_logps(f[lo:lo+2],tok[lo:lo+2],mask[lo:lo+2])
            ratio_error=max(ratio_error,float(((lp.detach()-old[lo:lo+2]).abs()*mask[lo:lo+2]).max()))
            loss,pg,kl=token_surrogate(lp,old[lo:lo+2],ref[lo:lo+2],mask[lo:lo+2],tokenadv[lo:lo+2].detach(),w[lo:lo+2])
            (loss*(len(lp)/(n*8))).backward();loss_sum+=float(loss.detach())*len(lp)/(n*8);pgs.extend(pg.tolist());kls.extend(kl.tolist())
        assert ratio_error==0.,'UNUPDATED_BEHAVIOR_RATIO_NOT_ONE'
        ce_loss=0.
        if self.arm=='CE_AUX_TARS_RAPO':
            ce_loss=self.ce_accumulate(features,labels,weights)
            self.counts['CE_aux_queries']+=n
            if kind=='engineering_responses':self.counts['engineering_teacherforced_queries']+=n
        norm,delta=self.step(opt);sums=task.sum(-1).long();classes={};elig_query=eligible.reshape(n,8).any(-1)
        for c in sorted(set(int(y) for y in labels)):
            chosen=labels.to('cuda')==c;s=sums[chosen]
            classes[str(c)]={'queries':int(chosen.sum()),'all_wrong':int((s==0).sum()),'all_correct':int((s==8).sum()),'mixed':int(((s>0)&(s<8)).sum()),'correct_responses':int(s.sum()),'eligible_teacher_responses':int(eligible.reshape(n,8)[chosen].sum()),'all_wrong_queries_with_valid_teacher_pair':int(((sums==0)&elig_query&chosen).sum())}
        if kind=='post_RL_responses':self.rollout_records.append({'stage':self.stage,'arm':self.arm,'update':self.current_update,'labels':labels.tolist(),'text':text,'done':done.tolist(),'parsed':parsed,'negatives':negatives})
        return {'loss':loss_sum+ce_loss,'CE_aux_loss':ce_loss,'gradient_norm':norm,'parameter_L2':delta,'nonzero_update':delta>0,'sigma_before':float(sigma),'sigma_after':new_sigma,'task_reward_mean':float(task.mean()),'retention_reward_mean':float(ret.mean()),'drift_mean':float(drift.mean()),'drift_max':float(drift.max()),'total_reward_mean':float(total.mean()),'total_reward_batch_std':float(total.std(unbiased=False)),'advantage_max_abs':float(adv.abs().max()),'token_advantage_max_abs':float(tokenadv.abs().max()),'teacher_evidence_masked_abs_max':float((evidence.abs()*mask).max()),'positive_teacher_rows':positive_rows,'negative_teacher_rows':negative_rows,'eligible_teacher_responses':int(eligible.sum()),'teacher_scoring_seconds':teacher_seconds,'teacher_last_sync_step':self.teacher_sync_step,'teacher_and_actor_anchor_restored':True,'KL_mean':sum(kls)/len(kls),'PG_mean':sum(pgs)/len(pgs),'unupdated_behavior_logp_difference':ratio_error,'sampled_vs_TF_logp_max_difference':float(((sampled_lp-old).abs()*mask).max()),'class_groups':classes},new_sigma

    def labels(self,ids):return torch.tensor([self.y[int(i)] for i in ids],dtype=torch.long)

    def opt(self):return torch.optim.AdamW(self.e.engine.parameters,lr=2e-5,weight_decay=0)

    def native(self):
        self.status('NATIVE_CURRENT_LABEL_CONTRAST');write(self.out/'SOFTWARE_CHECK.json',{**fixtures(),**interface_fixtures(),**contrast_fixtures()})
        ids=[i for i,c in enumerate(self.y) if c in STAGES[0]][:16];first=ids[:2]
        self.access.check('TRAIN',first,0,'NATIVE','native_images');expected=self.e.engine.prompt_ids.clone()
        features,raw=self.e.engine.encode(self.e.images(self.e.train,first));self.counts['TRAIN_encoder_rows']+=2
        assert torch.equal(expected,self.e.engine.prompt_ids)
        with torch.no_grad():
            full=self.e.engine.model(**raw,use_cache=False,output_hidden_states=True);self.counts['TRAIN_encoder_rows']+=2
            prompt=self.e.engine.prompt_ids.expand(2,-1);pos,_=self.e.engine.positions(prompt,torch.ones_like(prompt))
            hidden=self.e.engine.model.model(inputs_embeds=self.e.engine.embeddings(prompt,features),position_ids=pos,attention_mask=torch.ones_like(prompt),use_cache=False,return_dict=True).last_hidden_state
            hdiff=float((hidden[:,-1:]-full.hidden_states[-1][:,-1:]).abs().max());head=self.e.engine.model.lm_head(hidden[:,-1:])[:,0].float()
            pdiff=float((head-self.e.engine.model.lm_head(full.hidden_states[-1][:,-1:])[:,0].float()).abs().max());rawdiff=float((head-full.logits[:,-1].float()).abs().max())
        assert hdiff<=.05 and pdiff<=.05
        self.bind_prompt(0,profiling=True);unit=torch.ones(23,device='cuda');records=[];non_teacher=[]
        for arm in ARMS:
            self.arm=arm;self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();self.teacher_current=cpu_tree(self.prefix);self.teacher_sync_step=0
            self.counts['engineering_optimizer_attempts']+=1;t=time.time()
            rec,_=self.rl(self.get('TRAIN',first,'native_training'),self.labels(first),self.opt(),unit,True,1.,124000,'engineering_responses')
            seconds=time.time()-t;non_teacher.append(seconds-rec['teacher_scoring_seconds']);records.append({'arm':arm,'total_seconds':seconds,'non_teacher_seconds':non_teacher[-1],**rec})
        assert records[2]['positive_teacher_rows']>0 and records[3]['positive_teacher_rows']>0,'NATIVE_NEW_TEACHER_PATH_NOT_EXERCISED'
        self.arm='NATIVE_SFT_PATH';self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();self.counts['engineering_optimizer_attempts']+=1;self.counts['engineering_teacherforced_queries']+=16
        t=time.time();ce=self.ce(self.get('TRAIN',ids,'native_training'),self.labels(ids),self.opt(),unit);ce_seconds=time.time()-t
        self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();self.teacher_current=cpu_tree(self.prefix)
        f=self.get('TRAIN',first,'native_teacher_profile').repeat_interleave(8,0);seedtok,_=self.e.engine.label_tokens([self.e.names[int(self.y[first[0]])]])
        tok=seedtok[:,:1].expand(16,24).contiguous();mask=torch.ones_like(tok,dtype=torch.bool)
        _,positive_seconds=self.teacher_logps(f,tok,mask,[self.e.names[int(self.y[first[i//8]])] for i in range(16)],'positive',profile=True)
        _,negative_seconds=self.teacher_logps(f,tok,mask,[self.e.names[(int(self.y[first[i//8]])+1)%5] for i in range(16)],'negative',profile=True)
        t=time.time();self.generate(self.get('TRAIN',ids,'native_profile'),'engineering_responses');greedy_seconds=time.time()-t
        assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),self.prefix))
        assert all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in self.e.engine.adapters)
        forecast=1.5*(512*4*sum(non_teacher)+512*4*(2*positive_seconds+negative_seconds)+(17548+22812)/16*greedy_seconds)+1800
        proof={'passed':forecast<self.cfg['deadline']-time.time(),'single_hidden_parity':hdiff,'same_shape_last_head_parity':pdiff,'raw_full_head_vs_last_head_difference':rawdiff,'native_processor_token_binding_exact':True,'native_training_paths':records+[{'arm':'NATIVE_SFT_PATH',**ce}],'prefix_and_reference_restored':True,'original_forecast_seconds':forecast,'remaining_seconds':self.cfg['deadline']-time.time(),'non_teacher_path_seconds':non_teacher,'worst_positive16x24_TF_seconds':positive_seconds,'worst_negative16x24_TF_seconds':negative_seconds,'greedy16_seconds':greedy_seconds,'SFT_path_seconds_not_science_forecast':ce_seconds,'native_generation_responses':80,'native_TRAIN_encodings':4,'native_optimizer_attempts':5,'native_extra_teacher_profile_rows':32,'no_future_words_to_profile_model':True,'conservative_longest_prompt_profile':self.profile_prompt,'conservative_teacher_prompt_tokens':self.teacher_max_tokens}
        write(self.out/'NATIVE_ADMISSION.json',proof);assert proof['passed'],'FULL_FOUR_ARM_MATRIX_NOT_TIME_ADMITTED'
        self.bind_prompt(0);self.arm='COMMON';self.latest_sigma=1.

    def save_logs(self):
        write(self.out/'TRAINING_CURVES.json',self.curves);write(self.out/'ACCESS.private.json',self.access.journal)
        write(self.out/'EVALUATIONS.json',self.evaluations);write(self.out/'STATE_BINDINGS.json',self.bindings);write(self.out/'ROLLOUT_RECORDS.private.json',self.rollout_records);write(self.out/'TEACHER_BINDING_SUMMARY.json',list(self.teacher_binding_stats.values()))

    def evaluate(self,arm,sealed):
        started=time.time();self.status('ISOLATED_SEEN_CLASS_AUDIT')
        seen=[c for task in STAGES[:self.stage+1] for c in task];ids=[i for i,c in enumerate(self.vy) if c in seen]
        good=[0]*23;den=[0]*23;predictions=[];lookup={self.base.normalize(n):c for c,n in self.e.names.items()}
        for lo in range(0,len(ids),16):
            chunk=ids[lo:lo+16];_,_,_,text,done=self.generate(self.get('VAL',chunk,'isolated_audit',sealed),'VAL_responses')
            pred=[lookup.get(t,-1) if bool(d) else -1 for t,d in zip(text,done)];predictions.extend(pred)
            for i,p in zip(chunk,pred):den[self.vy[i]]+=1;good[self.vy[i]]+=int(p==self.vy[i])
        result={'arm':arm,'stage':self.stage,'shared_first_task_readout':self.stage==0,
                'evaluation_started_epoch':started,'evaluation_finished_epoch':time.time(),
                'invalid_predictions':sum(p<0 for p in predictions),'predictions_outside_seen_schema':sum(p>=0 and p not in seen for p in predictions),
                **score(good,den,seen,self.e.class_counts.tolist())}
        torch.save({'ids':ids,'predictions':predictions},self.out/f'VAL_{self.stage}_{arm}.private.pt')
        return result

    def calibration(self,current):
        good={c:0 for c in STAGES[self.stage]};count={c:sum(y==c for y in self.y) for c in STAGES[self.stage]}
        lookup={self.base.normalize(n):c for c,n in self.e.names.items()}
        for lo in range(0,len(current),16):
            if lo%64==0:self.status('CALIBRATING_CURRENT_TASK',rows_done=lo,rows_total=len(current))
            ids=current[lo:lo+16];_,_,_,text,done=self.generate(self.get('TRAIN',ids,'current_calibration'),'calibration_responses')
            for row,t,d in zip(ids,text,done):good[self.y[row]]+=int(lookup.get(t,-1)==self.y[row] and bool(d))
        classes=STAGES[self.stage];w,z,median=weights_for_current([count[c] for c in classes],[good[c] for c in classes])
        weights=torch.ones(23,device='cuda')
        if self.arm in ARMS:
            for c,value in zip(classes,w):weights[c]=float(value)
        write(self.out/f'CALIBRATION_{self.stage}_{self.arm}.json',{'classes':classes,'counts':[count[c] for c in classes],
            'correct':[good[c] for c in classes],'mean_normalized_TARS_weights':w.tolist(),'Z':z,'raw_median':median,
            'used_weights':[float(weights[c]) for c in classes],'future_or_old_TRAIN_queries':0,'fixed_for_task':True})
        return weights

    def science(self):
        source=Path(self.cfg['prefix_source']);prior_evals=json.loads((source/'EVALUATIONS.json').read_text());initial=next(r for r in prior_evals if r['stage']==0)
        for arm in ARMS:self.evaluations.append({**copy.deepcopy(initial),'arm':arm,'reused_historical_prefix_readout':True})
        write(self.out/'STAGE_0_ALL_MODELS_SEALED.json',{'epoch':time.time(),'historical_prefix_readout_reused':True,'new_stage0_evaluation':False,'arms':ARMS})
        states={a:{'weights':cpu_tree(self.prefix),'optimizer':None,'sigma':1.,'updates':0,'teacher':cpu_tree(self.prefix) if a in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO'] else None,'teacher_sync_step':0} for a in ARMS}
        self.save_logs()
        for stage in range(1,5):
            self.stage=stage;self.bind_prompt(stage);current=[i for i,c in enumerate(self.y) if c in STAGES[stage]]
            g=torch.Generator().manual_seed(12474005+stage);tape=torch.tensor(current)[torch.randint(len(current),(128,8),generator=g)]
            torch.save(tape,self.out/f'QUERY_TAPE_{stage}.private.pt');order=ARMS[(stage-1)%4:]+ARMS[:(stage-1)%4]
            for arm in order:
                self.arm=arm;prior=states[arm];self.e.engine.restore(prior['weights']);self.e.engine.freeze_reference();self.teacher_current=prior['teacher'];self.teacher_sync_step=prior['teacher_sync_step'];opt=self.opt()
                if prior['optimizer'] is not None:opt.load_state_dict(prior['optimizer'])
                steps=[int(x['step']) for x in opt.state.values() if 'step' in x];assert not steps or set(steps)=={prior['updates']}
                self.bindings.append({'stage':stage,'arm':arm,'optimizer_steps_before':prior['updates'],'sigma_before':prior['sigma'],'previous_weights_restored':all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),prior['weights'])),'anchor_from_own_previous_weights':all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in self.e.engine.adapters),'teacher_sync_step_before':prior['teacher_sync_step'],'old_TRAIN_access_allowed':False})
                assert self.bindings[-1]['previous_weights_restored'] and self.bindings[-1]['anchor_from_own_previous_weights']
                weights=self.calibration(current);sigma=prior['sigma']
                for k in range(128):
                    self.current_update=k;global_step=prior['updates']+k;self.status('CONTINUAL_CURRENT_LABEL_CONTRAST',update=k,total=128)
                    if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO'] and global_step%10==0:self.teacher_current=self.e.engine.snapshot();self.teacher_sync_step=global_step
                    batch=tape[k].tolist();self.counts['post_optimizer_attempts']+=1
                    rec,sigma=self.rl(self.get('TRAIN',batch),self.labels(batch),opt,weights,True,sigma,12480000+1000*stage+k)
                    self.counts['post_completed_updates']+=1;self.curves.append({'phase':'POST','stage':stage,'arm':arm,'update':k,**rec})
                    if k%16==0:self.save_logs()
                state={'weights':self.e.engine.snapshot(),'optimizer':cpu_tree(opt.state_dict()),'sigma':sigma,'updates':prior['updates']+128,'teacher':cpu_tree(self.teacher_current),'teacher_sync_step':self.teacher_sync_step}
                states[arm]=state;self.checkpoint(state,f'STATE_{stage}_{arm}.private.pt');del opt
            write(self.out/f'STAGE_{stage}_ALL_MODELS_SEALED.json',{'epoch':time.time(),'arms':ARMS,'historical_TRAIN_replay':0})
            for arm in ARMS:self.arm=arm;self.e.engine.restore(states[arm]['weights']);self.evaluations.append(self.evaluate(arm,True))
            self.save_logs()
        final=summarize(self.evaluations,self.e.class_counts.tolist());write(self.out/'FINAL_RESULTS.json',final)
        self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=final['all_gates_pass'])

    def checkpoint(self,state,name):
        tmp=self.out/(name+'.partial');torch.save(state,tmp);tmp.replace(self.out/name)


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    begin=time.time();cpu=time.process_time();r=None
    try:
        r=Run(cfg);r.native();r.science()
    finally:
        if r is not None:
            r.save_logs();write(r.out/'COSTS.json',{'counts':r.counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'new_feature_model_copy_bytes':0,'zero_old_TRAIN_replay_by_accessor':True})
        else:
            write(Path(cfg['output'])/'LOAD_FAILURE_COSTS.json',{'CPU_seconds':time.process_time()-cpu,
                'wall_seconds':time.time()-begin,'generation_responses':0,'optimizer_attempts':0,'TRAIN_VAL_encoder_rows':0,
                'original_clock_not_reset':True})
