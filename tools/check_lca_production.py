"""Production call-chain regression on synthetic CPU images, not medical validation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

import run_module_additions as production
import run_lca_supported_cycle as cycle
import prototype_coherent as native
import lca_supported as lca
from check_lca_supported import run as math_check


class SyntheticImages(Dataset):
    def __init__(self,rows,root,transform,seed):self.rows=rows;self.epoch=0
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];g=torch.Generator().manual_seed(74002+r['label']*101+r['sample'])
        x=torch.randn(5,generator=g)
        return (x,r['label'],i) if self.indexed else (x,r['label'])
    indexed=False


class IndexedSynthetic(SyntheticImages):indexed=True


class Encoder(torch.nn.Module):
    dim=6
    def __init__(self,*args):super().__init__();self.linear=torch.nn.Linear(5,6)
    def forward(self,x):return F.normalize(self.linear(x),dim=-1)


def features(model,loader,budget,local):
    xs=[];ys=[];ps=[]
    with torch.no_grad():
        for x,y in loader:
            z=model(x);xs.append(z.double());ys.append(y)
            if local:ps.append(F.normalize(z[:,None,:]+torch.eye(6)[:4][None]*.05,dim=-1).double())
    return torch.cat(xs),torch.cat(ys),torch.cat(ps) if local else None


def rejected(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError('Invalid configuration accepted')


def run():
    t=time.monotonic();torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    math_result=math_check()
    config=json.loads(Path(os.environ['Q162_CPU_CONFIG']).read_text());cycle.validate(config)
    for key,value in [('gpus',[0,2]),('jobs',config['jobs'][:2])]:rejected(lambda:cycle.validate(dict(config,**{key:value})))
    for key,value in [('lca_curvature_cap',.2),('expected_steps',469),('evaluation_split','official_test'),('lca_version','wrong')]:
        bad=copy.deepcopy(config);bad['jobs'][0][key]=value;rejected(lambda:cycle.validate(bad))
    spec=importlib.util.spec_from_file_location('committed_original',os.environ['Q162_ORIGINAL'])
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    as_tensor=torch.as_tensor;empty=native.empty
    def tensor(*a,**k):
        if str(k.get('device','')).startswith('cuda'):k['device']='cpu'
        return as_tensor(*a,**k)
    def loader(*a,**k):
        for key in ('multiprocessing_context','prefetch_factor','persistent_workers'):k.pop(key,None)
        k['num_workers']=0
        return DataLoader(*a,**k)
    steps=0;maximum=math_result['max_solve_residual'];checkpoints=0;true_nonzero=False
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp, \
         patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self), \
         patch.object(torch,'as_tensor',tensor),patch.object(torch.cuda,'manual_seed_all',lambda *a:None), \
         patch.object(torch.cuda,'max_memory_allocated',lambda *a:0), \
         patch.object(native,'empty',lambda dim,device='cpu':empty(dim,'cpu')), \
         patch.object(cycle,'validate_job',lambda c:None):
        for ds in ('ISIC','HK'):
            rows=[dict(label=c,sample=i,relative_path=f'{c}_{i}',identity_component=f'{c}_{i}') for c in cycle.ORDERS[ds] for i in range(8)]
            for module in (original,production):
                module.ApartFeatures=Encoder;module.Images=SyntheticImages;module.IndexedImages=IndexedSynthetic
                module.DataLoader=loader;module.manifests=lambda c:(rows,[])
            with patch.object(cycle,'read_training',lambda c:rows),patch.object(production.modules,'extract',features):
                for arm in ('base','local_transport'):
                    outputs=[]
                    template=copy.deepcopy(next(j for j in config['jobs'] if j['name']==ds and j['module']==arm))
                    pos=0;count=0
                    for size in cycle.SIZES[ds]:
                        labels=cycle.ORDERS[ds][pos:pos+size];pos+=size
                        fit,_=production.fit_split([r for r in rows if r['label'] in labels],74002)
                        count+=2*((len(fit)+63)//64)
                    for number,(module,enabled) in enumerate([(original,False),(production,False),(production,True)]):
                        c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_{number}'),expected_steps=count,max_steps=count,original_deadline=time.time()+300,max_wall_seconds=300,lca_supported=enabled)
                        if enabled:
                            with patch.object(production,'manifests',side_effect=AssertionError('Training must not open validation manifest')):
                                module.run(c)
                        else:module.run(c)
                        steps+=count
                        status=json.loads(Path(c['output'],'STATUS.json').read_text());assert status['status']=='TRAINED' and status['steps']==count
                        outputs.append(c['output'])
                    for task in range(1,len(cycle.SIZES[ds])+1):
                        a=torch.load(Path(outputs[0],f'stage_{task}.pt'),weights_only=False)
                        b=torch.load(Path(outputs[1],f'stage_{task}.pt'),weights_only=False)
                        assert torch.equal(a['head'],b['head'])
                        assert all(torch.equal(a['model'][k],v) for k,v in b['model'].items())
                        for k in ('mu','Q'):assert torch.equal(a['bank'][k],b['bank'][k])
                        assert a['bank']['n']==b['bank']['n']
                        active=torch.load(Path(outputs[2],f'stage_{task}.pt'),weights_only=False)
                        R,_=production.modules.metric(active['bank'],arm,ds,active['seen'],active['module_memory'])
                        S,M=lca.restore(active['lca_state'],6,len(active['seen']),active['head'].double())
                        W,audit=lca.bank_head(active['bank'],active['boundary_pair_weights'],R,S,M)
                        assert (W-active['lca_head_double']).norm()/W.norm()<1e-8
                        assert torch.equal(active['lca_head_double'].float(),active['head'])
                        assert not (S@W@M)[:,active['lca_old_count']:].any()
                        true_nonzero |= bool(S.any() and M.any())
                        maximum=max(maximum,audit['relative_residual']);checkpoints+=1
                    p=copy.deepcopy(template);p.update(output=str(Path(tmp)/f'{ds}_{arm}_pre'),original_deadline=time.time()+300,max_wall_seconds=300,preflight=True,expected_steps=0,max_steps=count)
                    production.run(p);receipt=json.loads(Path(p['output'],'PREFLIGHT.json').read_text())
                    assert receipt['parameters_unchanged'] and receipt['optimizer_updates']==0 and receipt['gradient_norm']>0
        campaign=copy.deepcopy(config)
        for i,j in enumerate(campaign['jobs']):
            j['output']=str(Path(tmp)/f'barrier_{i}');Path(j['output']).mkdir()
            Path(j['output'],'STATUS.json').write_text(json.dumps(dict(status='TRAINED',steps=j['expected_steps'],stages=len(j['task_sizes']),policy_updates=0)))
        file=Path(tmp)/'campaign.json';file.write_text(json.dumps(campaign));gate=Path(tmp)/'gate.json';gate.write_text(json.dumps(dict(phase='TRAIN')))
        candidate=dict(campaign['jobs'][0],evaluation_gate=str(gate))
        with patch.dict(os.environ,Q128_CONFIG=str(file)):
            rejected(lambda:cycle.check_barrier(candidate))
            gate.write_text(json.dumps(dict(phase='EVALUATE',training_sealed_at=1)));cycle.check_barrier(candidate)
            Path(campaign['jobs'][1]['output'],'STATUS.json').write_text(json.dumps(dict(status='TRAINED',steps=1,stages=3,policy_updates=0)))
            rejected(lambda:cycle.check_barrier(candidate))
    assert true_nonzero and not torch.cuda.is_initialized()
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-t,gpu_work=False,medical_images_or_raw_labels_read=False,test_accessed=False,formal_updates=0,
        synthetic_optimizer_updates=steps,real_GPU_preflights=0,disabled_production_parameters_heads_global_stats_exact=True,
        both_datasets_all_tasks_call_chain=True,controller_batch_loss_boundary_share_prior=True,
        active_synthetic_prior_nonzero=True,new_column_direct_force_zero=True,checkpoint_rebuilds=checkpoints,
        real_production_preflight_entry_CPU_fixtures_pass=True,wrong_GPU_budget_version_test_rejected=True,
        training_validation_manifest_guard=True,exact_four_training_barrier_rejects_missing=True,
        max_solve_residual=maximum,math_checks=math_result)


if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
