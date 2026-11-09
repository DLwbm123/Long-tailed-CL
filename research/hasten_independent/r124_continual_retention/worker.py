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

from .core import ARMS,STAGES,Access,fixtures,rewards_and_advantages,score,summarize,weights_for_current,write


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
        self.cfg=cfg;self.out=Path(cfg['output']);self.base=load_base(cfg['base_source']);self.e=self.base.Experiment(cfg)
        self.val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
        self.y=[int(r['original_label']) for r in self.e.train];self.vy=[int(r['original_label']) for r in self.val]
        assert len(self.y)==6821 and len(self.vy)==1698 and len({r['identity_component'] for r in self.e.train})==6821
        self.access=Access(self.y,self.vy);self.curves=[];self.evaluations=[];self.seals={};self.group_rows=[];self.bindings=[]
        self.counts={'engineering_optimizer_attempts':0,'engineering_teacherforced_queries':0,'SFT_updates':0,'SFT_queries':0,
            'post_optimizer_attempts':0,'post_CE_queries':0,'post_RL_responses':0,'calibration_responses':0,
            'VAL_responses':0,'engineering_responses':0,'generation_responses':0,'generation_attempts':0,
            'SFT_optimizer_attempts':0,'post_completed_updates':0,'TRAIN_encoder_rows':0,'VAL_encoder_rows':0}
        self.maps={}
        for split,n in [('TRAIN',6821),('VAL',1698)]:
            path=Path(cfg['feature_source'])/f'{split}_FEATURES.private.bin'
            assert path.stat().st_size==n*64*2048*2
            self.maps[split]=np.memmap(path,mode='r',dtype=np.uint16,shape=(n,64,2048))
        self.stage=0;self.arm='NATIVE';self.latest_sigma=1.
        assert all(int(torch.count_nonzero(a.b))==0 for a in self.e.engine.adapters)
        self.generic=self.e.engine.snapshot();self.e.engine.freeze_reference()
        write(self.out/'GENERIC_INITIALIZATION.json',{'task_LoRA_B_zero':True,'old_task_SFT_restored':False,
            'seed':74005,'read_only_generic_vision_cache':True,'future_image_feature_feedback_to_model':False,
            'schema_and_schedule_known_to_experiment_scheduler':True})

    def status(self,phase,**extra):
        assert time.time()<self.cfg['deadline']-120,'ORIGINAL_CLOCK'
        write(self.out/'STATUS.json',{'phase':phase,'epoch':time.time(),'stage':self.stage,'arm':self.arm,'counts':self.counts,**extra})

    def get(self,split,ids,purpose='current_training',sealed=False):
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
        self.counts['generation_responses']+=n;self.counts[kind]+=n
        write(self.out/'GENERATION_RECEIPT.json',{'inflight':False,'kind':kind,'rows':n,'counts':self.counts,'epoch':time.time()})
        return result

    def rl(self,features,labels,opt,weights,rapo,sigma,seed,kind='post_RL_responses'):
        n=len(labels);g=torch.Generator(device='cuda').manual_seed(seed);f=features.repeat_interleave(8,0)
        tok,mask,sampled_lp,text,done=self.generate(f,kind,g,True)
        targets=[self.base.normalize(self.e.names[int(c)]) for c in labels for _ in range(8)]
        task=torch.tensor([int(t==y and bool(d)) for t,y,d in zip(text,targets,done)],device='cuda',dtype=torch.float32).reshape(n,8)
        old=[];ref=[]
        with torch.no_grad():
            for lo in range(0,n*8,2):old.append(self.e.engine.token_logps(f[lo:lo+2],tok[lo:lo+2],mask[lo:lo+2]))
            with self.e.engine.reference():
                for lo in range(0,n*8,2):ref.append(self.e.engine.token_logps(f[lo:lo+2],tok[lo:lo+2],mask[lo:lo+2]))
        old=torch.cat(old).detach();ref=torch.cat(ref).detach()
        adv,ret,total,drift,new_sigma=rewards_and_advantages(task,old,ref,mask,rapo,sigma)
        opt.zero_grad(set_to_none=True);loss_sum=0.;pgs=[];kls=[];ratio_error=0.
        w=weights[labels.to('cuda')].repeat_interleave(8)
        for lo in range(0,n*8,2):
            lp=self.e.engine.token_logps(f[lo:lo+2],tok[lo:lo+2],mask[lo:lo+2])
            ratio_error=max(ratio_error,float(((lp.detach()-old[lo:lo+2]).abs()*mask[lo:lo+2]).max()))
            loss,pg,kl=self.base.surrogate(lp,old[lo:lo+2],ref[lo:lo+2],mask[lo:lo+2],adv.flatten()[lo:lo+2],w[lo:lo+2])
            (loss*(len(lp)/(n*8))).backward();loss_sum+=float(loss.detach())*len(lp)/(n*8);pgs.extend(pg.tolist());kls.extend(kl.tolist())
        assert ratio_error==0.,'UNUPDATED_BEHAVIOR_RATIO_NOT_ONE'
        norm,delta=self.step(opt)
        sums=task.sum(-1).long();classes={}
        for c in sorted(set(int(y) for y in labels)):
            chosen=labels.to('cuda')==c;s=sums[chosen]
            classes[str(c)]={'queries':int(chosen.sum()),'all_wrong':int((s==0).sum()),'all_correct':int((s==8).sum()),
                             'mixed':int(((s>0)&(s<8)).sum()),'correct_responses':int(s.sum())}
        return {'loss':loss_sum,'gradient_norm':norm,'parameter_L2':delta,'nonzero_update':delta>0,'sigma_before':float(sigma),
            'sigma_after':new_sigma,'task_reward_mean':float(task.mean()),'retention_reward_mean':float(ret.mean()),
            'drift_mean':float(drift.mean()),'drift_max':float(drift.max()),'total_reward_mean':float(total.mean()),
            'total_reward_batch_std':float(total.std(unbiased=False)),
            'advantage_max_abs':float(adv.abs().max()),'KL_mean':sum(kls)/len(kls),'PG_mean':sum(pgs)/len(pgs),
            'unupdated_behavior_logp_difference':ratio_error,'sampled_vs_TF_logp_max_difference':float(((sampled_lp-old).abs()*mask).max()),
            'class_groups':classes},new_sigma

    def labels(self,ids):return torch.tensor([self.y[int(i)] for i in ids],dtype=torch.long)

    def opt(self):return torch.optim.AdamW(self.e.engine.parameters,lr=2e-5,weight_decay=0)

    def native(self):
        self.status('NATIVE_CONTINUAL_RETENTION');software=fixtures();write(self.out/'SOFTWARE_CHECK.json',software)
        ids=[i for i,c in enumerate(self.y) if c in STAGES[0]][:16];first=ids[:2]
        self.access.check('TRAIN',first,0,'NATIVE','native_images')
        features,raw=self.e.engine.encode(self.e.images(self.e.train,first));self.counts['TRAIN_encoder_rows']+=2
        with torch.no_grad():
            full=self.e.engine.model(**raw,use_cache=False,output_hidden_states=True);self.counts['TRAIN_encoder_rows']+=2
            prompt=self.e.engine.prompt_ids.expand(2,-1);pos,_=self.e.engine.positions(prompt,torch.ones_like(prompt))
            hidden=self.e.engine.model.model(inputs_embeds=self.e.engine.embeddings(prompt,features),position_ids=pos,
                attention_mask=torch.ones_like(prompt),use_cache=False,return_dict=True).last_hidden_state
            hdiff=float((hidden[:,-1:]-full.hidden_states[-1][:,-1:]).abs().max())
            head=self.e.engine.model.lm_head(hidden[:,-1:])[:,0].float()
            pdiff=float((head-self.e.engine.model.lm_head(full.hidden_states[-1][:,-1:])[:,0].float()).abs().max())
            rawdiff=float((head-full.logits[:,-1].float()).abs().max())
        assert hdiff<=.05 and pdiff<=.05
        unit=torch.ones(23,device='cuda');timings=[];records=[]
        for arm in ARMS[:-1]:
            self.e.engine.restore(self.generic);self.e.engine.freeze_reference();self.arm=arm
            self.counts['engineering_optimizer_attempts']+=1
            begin=time.time();rec,_=self.rl(self.get('TRAIN',first,'native_training'),self.labels(first),self.opt(),unit,'RAPO' in arm,1.,124000,'engineering_responses')
            timings.append(time.time()-begin);records.append({'arm':arm,**rec})
        self.e.engine.restore(self.generic);self.e.engine.freeze_reference();self.arm='TARS_CE'
        self.counts['engineering_optimizer_attempts']+=1;self.counts['engineering_teacherforced_queries']+=16
        begin=time.time();ce=self.ce(self.get('TRAIN',ids,'native_training'),self.labels(ids),self.opt(),unit);ce_seconds=time.time()-begin
        self.e.engine.restore(self.generic);self.e.engine.freeze_reference();begin=time.time()
        self.generate(self.get('TRAIN',ids,'native_profile'),'engineering_responses');greedy_seconds=time.time()-begin
        assert all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),self.generic))
        assert all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in self.e.engine.adapters)
        val_rows=sum(sum(c in set(x for task in STAGES[:s+1] for x in task) for c in self.vy)*(1 if s==0 else 5) for s in range(5))
        calib=5*sum(c not in STAGES[0] for c in self.y)
        forecast=1.5*(2048*max(timings)*4+512*ce_seconds/2+306*ce_seconds+(calib+val_rows)/16*greedy_seconds)+1800
        proof={'passed':forecast<self.cfg['deadline']-time.time(),'single_hidden_parity':hdiff,'same_shape_last_head_parity':pdiff,
            'raw_full_head_vs_last_head_difference':rawdiff,'native_training_paths':records+[{'arm':'TARS_CE',**ce}],
            'generic_and_reference_restored':True,'original_forecast_seconds':forecast,'remaining_seconds':self.cfg['deadline']-time.time(),
            'native_generation_responses':80,'native_TRAIN_encodings':4,'native_optimizer_attempts':5,
            'calibration_responses_expected':calib,'VAL_responses_expected':val_rows,'science_RL_responses_expected':131072,
            'total_generation_responses_expected':80+calib+val_rows+131072,'cache_provenance':'read_only_R114_generic_frozen_vision_no_fresh_batch_bitexact_claim'}
        write(self.out/'NATIVE_ADMISSION.json',proof);assert proof['passed'],'FULL_CONTINUAL_MATRIX_NOT_TIME_ADMITTED'
        self.arm='COMMON';self.latest_sigma=1.

    def save_logs(self):
        write(self.out/'TRAINING_CURVES.json',self.curves);write(self.out/'ACCESS.private.json',self.access.journal)
        write(self.out/'EVALUATIONS.json',self.evaluations);write(self.out/'STATE_BINDINGS.json',self.bindings)

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
        if self.arm in ['TARS','TARS_RAPO','TARS_CE']:
            for c,value in zip(classes,w):weights[c]=float(value)
        write(self.out/f'CALIBRATION_{self.stage}_{self.arm}.json',{'classes':classes,'counts':[count[c] for c in classes],
            'correct':[good[c] for c in classes],'mean_normalized_TARS_weights':w.tolist(),'Z':z,'raw_median':median,
            'used_weights':[float(weights[c]) for c in classes],'future_or_old_TRAIN_queries':0,'fixed_for_task':True})
        return weights

    def science(self):
        first=[i for i,c in enumerate(self.y) if c in STAGES[0]];assert len(first)==2434
        self.stage=0;self.arm='COMMON';opt=torch.optim.AdamW(self.e.engine.parameters,lr=2e-4,weight_decay=0)
        g=torch.Generator().manual_seed(74005);unit=torch.ones(23,device='cuda')
        for epoch in range(2):
            ids=torch.tensor(first)[torch.randperm(len(first),generator=g)]
            for lo in range(0,len(ids),16):
                k=self.counts['SFT_updates'];self.status('COMMON_PREFIX_SFT',update=k,total=306)
                opt.param_groups[0]['lr']=2e-4*.5*(1+math.cos(math.pi*k/306));batch=ids[lo:lo+16].tolist()
                self.counts['SFT_optimizer_attempts']+=1
                rec=self.ce(self.get('TRAIN',batch),self.labels(batch),opt,unit)
                self.counts['SFT_updates']+=1;self.counts['SFT_queries']+=len(batch)
                self.curves.append({'phase':'SFT','stage':0,'arm':'COMMON','update':k,**rec})
                if k%32==0:self.save_logs()
        assert self.counts['SFT_updates']==306 and self.counts['SFT_queries']==4868
        common=self.e.engine.snapshot();self.checkpoint(common,'COMMON_PREFIX_SFT.private.pt');del opt
        write(self.out/'STAGE_0_ALL_MODELS_SEALED.json',{'epoch':time.time(),'common_first_task_only':True,'arms':ARMS})
        result=self.evaluate('COMMON',True)
        for arm in ARMS:self.evaluations.append({**copy.deepcopy(result),'arm':arm})
        states={a:{'weights':cpu_tree(common),'optimizer':None,'sigma':1.,'updates':0} for a in ARMS}
        self.save_logs()
        for stage in range(1,5):
            self.stage=stage;current=[i for i,c in enumerate(self.y) if c in STAGES[stage]]
            generator=torch.Generator().manual_seed(12474005+stage)
            tape=torch.tensor(current)[torch.randint(len(current),(128,8),generator=generator)]
            torch.save(tape,self.out/f'QUERY_TAPE_{stage}.private.pt')
            order=ARMS[(stage-1)%5:]+ARMS[:(stage-1)%5]
            for arm in order:
                self.arm=arm;prior=states[arm];self.e.engine.restore(prior['weights']);self.e.engine.freeze_reference()
                opt=self.opt()
                if prior['optimizer'] is not None:opt.load_state_dict(prior['optimizer'])
                optsteps=[int(x['step']) for x in opt.state.values() if 'step' in x]
                assert not optsteps or set(optsteps)=={prior['updates']}
                self.bindings.append({'stage':stage,'arm':arm,'optimizer_steps_before':prior['updates'],
                    'optimizer_state_loaded':prior['optimizer'] is not None,'sigma_before':prior['sigma'],'previous_stage':stage-1,
                    'anchor_from_own_previous_weights':all(torch.equal(a.a,a.ref_a) and torch.equal(a.b,a.ref_b) for a in self.e.engine.adapters),
                    'previous_weights_restored':all(torch.equal(a,x) and torch.equal(b,y) for (a,b),(x,y) in zip(self.e.engine.snapshot(),prior['weights'])),
                    'old_TRAIN_access_allowed':False})
                assert self.bindings[-1]['anchor_from_own_previous_weights'] and self.bindings[-1]['previous_weights_restored']
                weights=self.calibration(current);sigma=prior['sigma']
                for k in range(128):
                    self.status('CONTINUAL_CURRENT_TASK',update=k,total=128)
                    ids=tape[k].tolist();features=self.get('TRAIN',ids);labels=self.labels(ids)
                    self.counts['post_optimizer_attempts']+=1
                    if arm=='TARS_CE':rec=self.ce(features,labels,opt,weights);self.counts['post_CE_queries']+=len(ids)
                    else:rec,sigma=self.rl(features,labels,opt,weights,'RAPO' in arm,sigma,12480000+1000*stage+k)
                    self.counts['post_completed_updates']+=1
                    self.curves.append({'phase':'POST','stage':stage,'arm':arm,'update':k,**rec})
                    if k%16==0:self.save_logs()
                state={'weights':self.e.engine.snapshot(),'optimizer':cpu_tree(opt.state_dict()),'sigma':sigma,'updates':prior['updates']+128}
                states[arm]=state;self.checkpoint(state,f'STATE_{stage}_{arm}.private.pt');del opt,features
            write(self.out/f'STAGE_{stage}_ALL_MODELS_SEALED.json',{'epoch':time.time(),'arms':ARMS,'current_task_only':True})
            for arm in ARMS:
                self.arm=arm;self.e.engine.restore(states[arm]['weights']);self.evaluations.append(self.evaluate(arm,True))
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
