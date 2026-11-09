"""TARS-proportional query allocation and complete surrogate/KL correction."""
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .core import write

ARMS=['ALLOC_TARS_FULL_IS','ALLOC_TARS_PG_ONLY_IS','ALLOC_GRPO_FULL_IS','ALLOC_CE_FULL_IS']


def objective_check():
    p=torch.tensor([.8,.2],dtype=torch.float64);w=torch.tensor([1.,4.],dtype=torch.float64)
    q=p*w/(p*w).sum();rho=p/q
    x=torch.tensor([.1,-.3],dtype=torch.float64,requires_grad=True)
    pg=x.square()+x;kl=(x-.5).square()
    natural=(p*(-w*pg+.04*kl)).sum()
    corrected=(q*rho*(-w*pg+.04*kl)).sum()
    wrong=(q*(-rho*w*pg+.04*kl)).sum()
    assert torch.allclose(natural,corrected,atol=1e-12)
    a=torch.autograd.grad(natural,x,retain_graph=True)[0]
    b=torch.autograd.grad(corrected,x,retain_graph=True)[0]
    assert torch.allclose(a,b,atol=1e-12)
    assert abs(float((wrong-natural).detach()))>1e-4
    assert torch.allclose(rho*w,torch.full_like(w,float((p*w).sum())))
    return {'passed':True,'full_objective_and_gradient_match':True,'PG_only_correction_detected_as_different':True}


def make_experiment(cfg):
    root=Path(cfg['base_source']);name='q114_dependency'
    assert json.loads((root/'CONFIG.private.json').read_text())['source_commit']=='5c0a106fe77f98b5ae15f04e953ad93cad8831e5'
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    module=__import__(name+'.worker',fromlist=['Experiment']);Base=module.Experiment

    class Experiment(Base):
        def __init__(self,c):
            super().__init__(c)
            self.sft=torch.load(root/'SFT.private.pt',map_location='cpu',weights_only=True)
            self.engine.restore(self.sft);self.engine.freeze_reference()
            info=json.loads((root/'TARS_WEIGHTS.json').read_text())
            self.w=torch.tensor(info['weights'],dtype=torch.float64)
            assert self.class_counts.tolist()==info['counts']
            self.p=self.class_counts.double()/self.class_counts.sum()
            self.Z=float((self.p*self.w).sum());self.q=self.p*self.w/self.Z;self.rho=self.p/self.q
            assert torch.allclose(self.rho*self.w,torch.full_like(self.w,self.Z),atol=1e-12)

        def coefficients(self,arm):
            if arm=='ALLOC_GRPO_FULL_IS':return self.rho.float().cuda(),self.rho.float().cuda()
            if arm=='ALLOC_TARS_PG_ONLY_IS':return (self.rho*self.w).float().cuda(),torch.ones(23,device='cuda')
            return (self.rho*self.w).float().cuda(),self.rho.float().cuda()

        def rl_corrected(self,features,labels,opt,pg_coeff,kl_coeff,seed):
            G,n=8,len(labels);f=features.repeat_interleave(G,0)
            generator=torch.Generator(device='cuda').manual_seed(seed)
            tokens,mask,sampled,text,done=self.engine.generate(f,generator,True)
            target=[module.normalize(self.names[int(c)]) for c in labels for _ in range(G)]
            reward=torch.tensor([int(t==y and bool(d)) for t,y,d in zip(text,target,done)],device='cuda',dtype=torch.float32).reshape(n,G)
            adv=module.advantages(reward).flatten().detach()
            with torch.no_grad(),self.engine.reference():
                ref=torch.cat([self.engine.token_logps(f[i:i+2],tokens[i:i+2],mask[i:i+2]) for i in range(0,n*G,2)])
            with torch.no_grad():
                old=torch.cat([self.engine.token_logps(f[i:i+2],tokens[i:i+2],mask[i:i+2]) for i in range(0,n*G,2)])
            pgw=pg_coeff[labels.cuda()].repeat_interleave(G);klw=kl_coeff[labels.cuda()].repeat_interleave(G)
            opt.zero_grad(set_to_none=True);loss_total,kl_total,weighted_kl,delta_max=0.,0.,0.,0.
            for lo in range(0,n*G,2):
                lp=self.engine.token_logps(f[lo:lo+2],tokens[lo:lo+2],mask[lo:lo+2]);m=mask[lo:lo+2]
                delta_max=max(delta_max,float(((lp.detach()-old[lo:lo+2]).abs()*m).max()))
                ratio=(lp-old[lo:lo+2]).exp();a=adv[lo:lo+2,None]
                pg=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)
                d=ref[lo:lo+2]-lp;assert float(d.detach().abs().max())<80
                kl=torch.expm1(d)-d;length=m.sum(-1)
                pg=(pg*m).sum(-1)/length;kl=(kl*m).sum(-1)/length
                loss=(-pgw[lo:lo+2]*pg+.04*klw[lo:lo+2]*kl).sum()/(n*G)
                assert torch.isfinite(loss);loss.backward()
                loss_total+=float(loss.detach());kl_total+=float(kl.detach().sum())/(n*G)
                weighted_kl+=float((klw[lo:lo+2]*kl.detach()).sum())/(n*G)
            assert delta_max==0,'UNCHANGED_BEHAVIOR_RATIO_NOT_ONE'
            norm=self.step(opt);self.counts['generation_responses']+=n*G
            return {'loss':loss_total,'KL':kl_total,'importance_weighted_KL':weighted_kl,'gradient_norm':norm,
                    'reward_mean':float(reward.mean()),'groups':n,'constant_groups':int((reward.std(-1,unbiased=False)==0).sum()),
                    'correct_by_class':torch.bincount(labels,weights=reward.sum(-1).cpu(),minlength=23).tolist(),
                    'responses_by_class':(torch.bincount(labels,minlength=23)*G).tolist(),'tokens':int(mask.sum()),
                    'truncated_responses':int((~done).sum()),'unchanged_policy_logp_max_difference':delta_max,
                    'cached_sampling_vs_behavior_forward_max_difference':float(((sampled-old).abs()*mask).max()),
                    'mean_surrogate_coefficient':float(pgw.mean()),'mean_KL_coefficient':float(klw.mean())}

        def native(self):
            self.status('NATIVE_ENGINEERING');check=objective_check()
            features,_=self.engine.encode(self.images(self.train,[0,1]));self.counts['TRAIN_encoder_rows']+=2
            timings={};details=[]
            for arm in ARMS:
                self.engine.restore(self.sft);opt=self.optimizer(2e-5);t=time.time();pg,kl=self.coefficients(arm)
                if arm=='ALLOC_CE_FULL_IS':
                    loss,norm=self.ce(features,self.labels[:2],opt,pg);rec={'loss':loss,'gradient_norm':norm}
                else:rec=self.rl_corrected(features,self.labels[:2],opt,pg,kl,117000)
                torch.cuda.synchronize();timings[arm]=time.time()-t;details.append({'arm':arm,'result':rec})
                self.counts['engineering_optimizer_updates']+=1;del opt
            self.engine.restore(self.sft)
            estimate=1.5*4*256*sum(timings.values())+1800
            passed=estimate<self.cfg['deadline']-time.time()
            write(self.out/'NATIVE_ADMISSION.json',{'passed':passed,'engineering_only':True,'TRAIN_images':2,'VAL_TEST':0,
                                                  'objective_fixture':check,'two_query_seconds':timings,
                                                  'complete_projection_seconds_with_margin_reserve':estimate,
                                                  'remaining_seconds':self.cfg['deadline']-time.time(),'paths':details})
            assert passed,'FULL_MATRIX_NOT_TIME_ADMITTED'

        def science(self):
            train=np.memmap(root/'TRAIN_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(6821,64,2048))
            generator=torch.Generator().manual_seed(11774005)
            classes=torch.multinomial(self.q,2048,replacement=True,generator=generator).reshape(256,8)
            pools={c:torch.where(self.labels==c)[0] for c in range(23)}
            tape=torch.stack([torch.stack([pools[int(c)][torch.randint(len(pools[int(c)]),(),generator=generator)] for c in row]) for row in classes])
            torch.save(tape,self.out/'QUERY_TAPE.private.pt')
            write(self.out/'ALLOCATION.json',{'natural_class_probability':self.p.tolist(),'query_class_probability':self.q.tolist(),
                                             'TARS_weights':self.w.tolist(),'importance_p_over_q':self.rho.tolist(),'Z':self.Z,
                                             'corrected_surrogate_coefficient':(self.rho*self.w).tolist(),
                                             'realized_queries_by_class':torch.bincount(classes.flatten(),minlength=23).tolist(),
                                             'queries_per_arm':2048,'same_tape_all_new_arms':True})
            self.records=[]
            for arm in ARMS:
                self.engine.restore(self.sft);opt=self.optimizer(2e-5);pg,kl=self.coefficients(arm)
                for k,ids in enumerate(tape):
                    self.status('POSTTRAINING',arm=arm,update=k,total=256);start=time.time()
                    f,y=self.features(train,ids),self.labels[ids]
                    if arm=='ALLOC_CE_FULL_IS':
                        loss,norm=self.ce(f,y,opt,pg);rec={'loss':loss,'gradient_norm':norm,'groups':0}
                    else:rec=self.rl_corrected(f,y,opt,pg,kl,11480000+k)
                    self.counts['posttraining_optimizer_updates']+=1
                    rec.update(arm=arm,update=k,phase='POSTTRAINING',seconds=time.time()-start);self.records.append(rec)
                    if k%8==0:write(self.out/'TRAINING_CURVES.json',self.records)
                torch.save(self.engine.snapshot(),self.out/f'{arm}.private.pt')
                write(self.out/f'{arm}_SEALED.json',{'arm':arm,'updates':256,'epoch':time.time(),'VAL_TEST':0})
                del opt;gc.collect();torch.cuda.empty_cache()
            write(self.out/'TRAINING_CURVES.json',self.records)
            write(self.out/'ALL_BRANCHES_SEALED.json',{'arms':ARMS,'epoch':time.time(),'new_VAL_TEST_so_far':0})
            rows=list(csv.DictReader((Path(self.cfg['manifest'])/'val.csv').open()));assert len(rows)==1698
            assert not {r['identity_component'] for r in rows}&{r['identity_component'] for r in self.train}
            val=np.memmap(root/'VAL_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(1698,64,2048));results=[]
            for arm in ARMS:
                self.engine.restore(torch.load(self.out/f'{arm}.private.pt',weights_only=True,map_location='cpu'))
                results.append(self.evaluate(val,rows,arm))
            prior=json.loads((Path(self.cfg['natural_control_source'])/'FINAL_RESULTS.json').read_text())
            natural=next(r for r in prior['results'] if r['name']=='PLAIN_TARS');sft=prior['historical_SFT']
            main,_,ordinary,ce=results
            gates={'macro_over_SFT_ge_1pp':main['macro']-sft['macro']>=.01,'tail_over_SFT_ge_2pp':main['tail']-sft['tail']>=.02,
                   'head_over_SFT_ge_minus1pp':main['head']-sft['head']>=-.01,
                   'macro_over_ALLOC_GRPO_positive':main['macro']>ordinary['macro'],'tail_over_ALLOC_GRPO_positive':main['tail']>ordinary['tail'],
                   'head_over_natural_TARS_nonnegative':main['head']>=natural['head'],'tail_over_natural_TARS_positive':main['tail']>natural['tail'],
                   'macro_over_ALLOC_CE_nonnegative':main['macro']>=ce['macro'],'tail_over_ALLOC_CE_positive':main['tail']>ce['tail']}
            write(self.out/'FINAL_RESULTS.json',{'results':results,'historical_SFT':sft,'reused_natural_TARS256':natural,
                                                'gates':gates,'all_gates_pass':all(gates.values()),'TEST':0,'seed':74005,
                                                'scope':'static_development_tailness_query_allocation_diagnostic',
                                                'independent_confirmation':False,'patient_independence_verified':False})
            self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=all(gates.values()))

    return Experiment(cfg)


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    e=None
    try:e=make_experiment(cfg);e.native();e.science()
    finally:
        if e is not None:write(Path(cfg['output'])/'COSTS.json',{'counts':e.counts,'CPU_seconds':time.process_time()-e.cpu,
              'wall_seconds':time.time()-e.begin,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'shared_R114_training_and_R116_natural_anchor_not_newly_executed':True})
