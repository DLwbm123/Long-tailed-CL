"""Fixed-bank absolute conditional-KL constraints for tailness-aware RL."""
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


def load_dependency(name, root):
    root = Path(root)
    spec = importlib.util.spec_from_file_location(name, root/'__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return __import__(name+'.worker', fromlist=['Experiment'])


def make_experiment(cfg):
    assert json.loads((Path(cfg['base_source'])/'CONFIG.private.json').read_text())['source_commit']=='5c0a106fe77f98b5ae15f04e953ad93cad8831e5'
    Base = load_dependency('q114_dependency',cfg['base_source']).Experiment
    geometry = load_dependency('q115_dependency',cfg['geometry_source'])

    class Experiment(Base):
        def __init__(self, config):
            super().__init__(config)
            self.original_sft = torch.load(Path(config['base_source'])/'SFT.private.pt',weights_only=True,map_location='cpu')
            self.engine.restore(self.original_sft)
            self.engine.freeze_reference()
            self.reference_vector = self.flatten().clone()
            self.arm = None
            self.bank = []
            self.extra = {'reference_generated_responses':0,'reference_risk_query_rows':0,
                          'risk_gradient_query_rows':0,'native_optimizer_updates':0,'science_optimizer_updates':0}

        def flatten(self):
            return torch.cat([p.detach().reshape(-1) for p in self.engine.parameters])

        @torch.no_grad()
        def assign(self, value):
            lo=0
            for p in self.engine.parameters:
                p.copy_(value[lo:lo+p.numel()].view_as(p));lo+=p.numel()
            assert lo==len(value)

        def full_logps(self, features, tokens, mask):
            e=self.engine
            ids=torch.cat([e.prompt_ids.expand(len(tokens),-1),tokens],1)
            attention=torch.cat([torch.ones_like(e.prompt_ids).expand(len(tokens),-1),mask.long()],1)
            positions,_=e.positions(ids,attention)
            hidden=e.model.model(inputs_embeds=e.embeddings(ids,features),position_ids=positions,
                                 attention_mask=attention,use_cache=False,return_dict=True).last_hidden_state
            return e.model.lm_head(hidden[:,e.prompt_ids.shape[1]-1:-1]).double().log_softmax(-1)

        def chunk_kl(self, b):
            lp=self.full_logps(b['features'],b['tokens'],b['mask'])
            value=(b['prob']*(b['logp']-lp)).sum(-1)
            assert float(value.detach().min()) > -1e-9
            value=value.clamp_min(0)
            return (value*b['mask']).sum(-1)/b['mask'].sum(-1)

        @torch.no_grad()
        def build_bank(self, features, kind):
            self.bank=[]
            assert len(features)==48
            for lo in range(0,48,2):
                f=features[lo:lo+2]
                with self.engine.reference():
                    tok,mask,_,_,_=self.engine.generate(f)
                    lp=self.full_logps(f,tok,mask).detach()
                self.bank.append({'features':f,'tokens':tok,'mask':mask,'logp':lp,'prob':lp.exp(),'class_index':lo//4})
            self.extra['reference_generated_responses']+=48
            self.counts['generation_responses']+=48
            values=self.risk_values()
            assert max(values)<1e-9,('REFERENCE_NOT_ZERO',max(values))
            write(self.out/f'{kind}_REFERENCE_BANK.json',{'anchors':48,'classes':12,'anchors_per_class':4,
                                                        'reference_conditional_KL_at_SFT':values,
                                                        'probability_storage_bytes':sum((b['prob'].numel()+b['logp'].numel())*8 for b in self.bank),
                                                        'synthetic_repeated_engineering_features':kind=='NATIVE'})

        @torch.no_grad()
        def risk_values(self):
            values=torch.zeros(12,device='cuda',dtype=torch.float64)
            for b in self.bank:
                values[b['class_index']]+=self.chunk_kl(b).sum()/4
            self.extra['reference_risk_query_rows']+=48
            assert torch.isfinite(values).all()
            return values.tolist()

        def prepare_risk(self):
            self.risk_before=self.risk_values()
            assert max(self.risk_before)<=.001,('CURRENT_STATE_OUTSIDE_FIXED_BUDGET',max(self.risk_before))
            for p in self.engine.parameters:p.grad=None
            if torch.equal(self.flatten(),self.reference_vector):
                self.risk_gradient=torch.zeros_like(self.reference_vector)
            else:
                worst=max(range(12),key=lambda c:self.risk_before[c])
                for b in self.bank:
                    if b['class_index']==worst:
                        (self.chunk_kl(b).sum()/4).backward()
                        self.extra['risk_gradient_query_rows']+=2
                self.risk_gradient=torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                               for p in self.engine.parameters]).clone()
            assert torch.isfinite(self.risk_gradient).all()
            for p in self.engine.parameters:p.grad=None

        @torch.no_grad()
        def step(self,opt):
            norm=torch.nn.utils.clip_grad_norm_(self.engine.parameters,1.)
            assert torch.isfinite(norm)
            before=self.flatten();opt.step();proposal=self.flatten()-before
            if self.arm=='PLAIN_TARS':
                self.last_step={'constraint_enforced':False,'applied_L2':float(proposal.norm()),
                                'nonzero_parameter_update':bool(proposal.norm()>0)}
                return float(norm)
            candidate,active=(geometry.cone_project(proposal,self.risk_gradient) if self.arm=='KL_PROJECTED_TARS' else (proposal,False))
            alphas=[1.,.5,.25,.125,.0625,0.]
            all_values=[]
            for alpha in alphas:
                self.assign(before+alpha*candidate)
                all_values.append(self.risk_values())
            assert max(all_values[-1])<=.001,'ZERO_STEP_NOT_FEASIBLE'
            choice=next(i for i,v in enumerate(all_values) if max(v)<=.001)
            self.assign(before+alphas[choice]*candidate)
            actual=self.flatten()-before
            self.last_step={'constraint_enforced':True,'max_reference_KL_before':max(self.risk_before),
                            'max_reference_KL_after':max(all_values[choice]),'class_reference_KL_after':all_values[choice],
                            'max_KL_trials':[max(v) for v in all_values],'accepted_alpha':alphas[choice],
                            'proposal_L2':float(proposal.norm()),'candidate_L2':float(candidate.norm()),
                            'applied_L2':float(actual.norm()),'risk_gradient_L2':float(self.risk_gradient.norm()),
                            'cone_active':active,'nonzero_parameter_update':bool(actual.norm()>0),
                            'fixed_reference_budget_verified':max(all_values[choice])<=.001}
            return float(norm)

        def native(self):
            self.status('NATIVE_ENGINEERING')
            geometry.geometry_check()
            f,_=self.engine.encode(self.images(self.train,[0,1]));self.counts['TRAIN_encoder_rows']+=2
            self.build_bank(f.repeat(24,1,1),'NATIVE')
            timing,details={},[]
            for arm in ['KL_PROJECTED_TARS','KL_SHRINK_TARS','PLAIN_TARS']:
                self.arm=arm;self.engine.restore(self.original_sft);opt=self.optimizer(2e-5);start=time.time()
                if arm!='PLAIN_TARS':self.prepare_risk()
                rec=self.rl(f,self.labels[:2],opt,torch.ones(23,device='cuda'),116000)
                torch.cuda.synchronize();timing[arm]=time.time()-start
                details.append({'arm':arm,'RL':rec,'constraint':self.last_step})
                self.counts['engineering_optimizer_updates']+=1;self.extra['native_optimizer_updates']+=1
                del opt
            self.engine.restore(self.original_sft)
            # The 48-anchor cost is already full size in native admission; only
            # the two-query RL component scales to the eight-query science batch.
            plain=timing['PLAIN_TARS']
            estimate=256*(12*plain+sum(max(0.,timing[a]-plain) for a in ['KL_PROJECTED_TARS','KL_SHRINK_TARS']))+1800
            passed=estimate<self.cfg['deadline']-time.time()
            write(self.out/'NATIVE_ADMISSION.json',{'passed':passed,'engineering_only':True,'TRAIN_images':2,'VAL_TEST':0,
                                                  'repeated_feature_reference_bank_not_scientific':True,
                                                  'two_query_seconds':timing,'complete_projection_with_reserve_seconds':estimate,
                                                  'remaining_seconds':self.cfg['deadline']-time.time(),'paths':details})
            self.bank=[];gc.collect();torch.cuda.empty_cache()
            assert passed,'FULL_MATRIX_NOT_TIME_ADMITTED'

        def science(self):
            root=Path(self.cfg['base_source'])
            train=np.memmap(root/'TRAIN_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(6821,64,2048))
            tape=torch.load(root/'QUERY_TAPE.private.pt',map_location='cpu',weights_only=True)[:256]
            w=torch.tensor(json.loads((root/'TARS_WEIGHTS.json').read_text())['weights'],device='cuda',dtype=torch.float32)
            head=torch.where(self.class_counts>=250)[0].tolist();assert len(head)==12
            generator=torch.Generator().manual_seed(11674005);ids=[]
            for c in head:
                pool=torch.where(self.labels==c)[0]
                ids.extend(pool[torch.randperm(len(pool),generator=generator)[:4]].tolist())
            torch.save(torch.tensor(ids),self.out/'ANCHOR_IDS.private.pt')
            self.engine.restore(self.original_sft)
            self.status('FIXED_REFERENCE_PRECOMPUTE')
            self.build_bank(self.features(train,ids),'SCIENCE')
            self.records=[]
            for arm in ['KL_PROJECTED_TARS','KL_SHRINK_TARS','PLAIN_TARS']:
                self.arm=arm;self.engine.restore(self.original_sft);opt=self.optimizer(2e-5)
                for k,query in enumerate(tape):
                    self.status('POSTTRAINING',arm=arm,update=k,total=256);start=time.time()
                    if arm!='PLAIN_TARS':self.prepare_risk()
                    rec=self.rl(self.features(train,query),self.labels[query],opt,w,11480000+k)
                    self.counts['posttraining_optimizer_updates']+=1;self.extra['science_optimizer_updates']+=1
                    rec.update(arm=arm,update=k,phase='POSTTRAINING',seconds=time.time()-start,**self.last_step)
                    self.records.append(rec)
                    if k%8==0:write(self.out/'TRAINING_CURVES.json',self.records)
                final_risk=self.risk_values()
                torch.save(self.engine.snapshot(),self.out/f'{arm}.private.pt')
                write(self.out/f'{arm}_SEALED.json',{'arm':arm,'updates':256,'epoch':time.time(),
                                                    'final_reference_KL_by_class':final_risk,'VAL_TEST':0})
                del opt;gc.collect();torch.cuda.empty_cache()
            write(self.out/'TRAINING_CURVES.json',self.records)
            write(self.out/'ALL_BRANCHES_SEALED.json',{'arms':['KL_PROJECTED_TARS','KL_SHRINK_TARS','PLAIN_TARS'],'epoch':time.time(),'new_VAL_TEST_so_far':0})
            rows=list(csv.DictReader((Path(self.cfg['manifest'])/'val.csv').open()));assert len(rows)==1698
            assert not {r['identity_component'] for r in rows}&{r['identity_component'] for r in self.train}
            val=np.memmap(root/'VAL_FEATURES.private.bin',mode='r',dtype=np.uint16,shape=(1698,64,2048))
            results=[]
            for arm in ['KL_PROJECTED_TARS','KL_SHRINK_TARS','PLAIN_TARS']:
                self.engine.restore(torch.load(self.out/f'{arm}.private.pt',map_location='cpu',weights_only=True))
                results.append(self.evaluate(val,rows,arm))
            old=json.loads((root/'FINAL_RESULTS.json').read_text())['results'];sft=next(r for r in old if r['name']=='SFT')
            main,shrink,plain=results
            gates={'macro_over_SFT_ge_1pp':main['macro']-sft['macro']>=.01,
                   'tail_over_SFT_ge_2pp':main['tail']-sft['tail']>=.02,
                   'head_over_SFT_ge_minus1pp':main['head']-sft['head']>=-.01,
                   'macro_over_KL_SHRINK_nonnegative':main['macro']>=shrink['macro'],
                   'tail_over_KL_SHRINK_positive':main['tail']>shrink['tail'],
                   'head_over_PLAIN_positive':main['head']>plain['head'],
                   'tail_over_PLAIN_nonnegative':main['tail']>=plain['tail']}
            write(self.out/'FINAL_RESULTS.json',{'results':results,'historical_SFT':sft,'gates':gates,
                                                'all_gates_pass':all(gates.values()),'seed':74005,'TEST':0,
                                                'scope':'static_development_fixed_reference_KL_diagnostic',
                                                'independent_confirmation':False,'patient_independence_verified':False})
            self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION',all_gates_pass=all(gates.values()))

    return Experiment(cfg)


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    e=None
    try:
        e=make_experiment(cfg);e.native();e.science()
    finally:
        if e is not None:write(Path(cfg['output'])/'COSTS.json',{'counts':e.counts,'extra':e.extra,
                                                              'CPU_seconds':time.process_time()-e.cpu,'wall_seconds':time.time()-e.begin,
                                                              'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                                              'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                                                              'shared_R114_training_not_newly_executed':True})
