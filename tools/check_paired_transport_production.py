"""Synthetic full production chain, disabled controls, high-dimensional operators and sealed reports."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import run_module_additions as production
import run_paired_transport_cycle as cycle
import paired_moment_transport as pmt
import paired_transport_design as affine
import prototype_coherent as native
import core1_competition as competition
from check_lca_production import SyntheticImages,IndexedSynthetic,Encoder,features,rejected


@torch.no_grad()
def paired_features(model,loader,budget,local):
    xs=[];ys=[];ps=[];ids=[]
    for x,y,row_ids in loader:
        z=model(x);xs.append(z.double());ys.append(y);ids.append(row_ids)
        if local:ps.append(torch.nn.functional.normalize(z[:,None,:]+torch.eye(6)[:4][None]*.05,dim=-1).double())
    return torch.cat(xs),torch.cat(ys),torch.cat(ps) if local else None,torch.cat(ids)


def envelope():
    torch.manual_seed(74002);d=12;k=4
    yy=torch.arange(2).repeat_interleave(6);xx=torch.randn(12,d,dtype=torch.float64)*.1
    weights=xx.new_full((12,),1/6);snapshot=affine.framed(native.append(native.empty(d),xx,yy,yy,weights),'start')
    cy=yy+2;current=torch.randn(12,d,dtype=torch.float64)*.1;after=current+.02*current[:,0:1]*torch.ones(1,d,dtype=torch.float64)
    ids=torch.arange(12);old,b,_,a=pmt.prepare(snapshot,None,current,after,None,None,cy,ids,ids,cy,weights,'start','end','global')
    R=production.modules.metric(b,'base','HK',list(range(k)),None)[0];pairs=competition.competition(b,native.head(b)[0])
    previous=competition.bank_head(b,pairs,regularizer=R)[0];inv,cross=native.proximal_base(old,R,k)
    target=torch.nn.functional.one_hot(cy,k).double();alpha=weights/k
    def objective(z):
        with torch.no_grad():
            nh=native.proximal_head(z.detach(),target,alpha,inv,cross,previous)
            q,mu,mass=competition.moments(old,z.detach(),cy,alpha,k)
            head,solve=competition.solve(q,mu,mass,pairs,previous=previous,proximal=.01,inverse=inv,x=z.detach(),alpha=alpha,native=nh,regularizer=R)
        value=.5*(alpha*(z@head-target).square().sum(1)).sum()+.5*competition.pair_loss(old,z,cy,alpha,head,pairs)
        value+=.5*native.old_square_losses(old,head).sum()/k+.0005*(head*(R@head)).sum()+.005*(head-previous).square().sum()
        return value,solve
    z=after.clone().requires_grad_();v,a=objective(z);g=torch.autograd.grad(v,z)[0];direction=torch.randn_like(z);direction/=direction.norm();epsilon=1e-5
    finite_difference=float((objective(z.detach()+epsilon*direction)[0]-objective(z.detach()-epsilon*direction)[0])/(2*epsilon))
    error=abs(finite_difference-float((g*direction).sum()))
    assert error<1e-7
    # Separate regional translations reduce to the original local-memory operation.
    region=torch.randn(12,4,d,dtype=torch.float64);delta=torch.randn(4,d,dtype=torch.float64)*.01;local=torch.randn(2,4,d,dtype=torch.float64)
    _,translated,regional,a=pmt.prepare(snapshot,local,current,current+.01,region,region+delta,cy,ids,ids,cy,weights,'start','end','global_regional')
    reference=native.translate(snapshot,current.new_full((d,),.01))
    assert torch.allclose(translated['Q'][:2],reference['Q'],atol=1e-12)
    assert torch.allclose(regional,production.modules.local_memory(local,delta,region+delta,cy),atol=1e-12)
    rejected(lambda:pmt.prepare(snapshot,None,current,after,None,None,cy,ids,ids.flip(0),cy,weights,'start','end','global'))
    rejected(lambda:affine.push_bank(translated,affine.fit(current,after,cy,ids,ids,'start','end','global')))
    return dict(envelope_directional_error=error,global_regional_translation_native=True,row_misalignment_and_frame_repeat_refused=True)


def run():
    start=time.monotonic();assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2)
    config=json.loads(Path(os.environ['Q215_CPU_CONFIG']).read_text());cycle.validate(config)
    for key,val in [('gpus',[0,2]),('jobs',config['jobs'][:4])]:rejected(lambda:cycle.validate(dict(config,**{key:val})))
    for key,val in [('pmt_relative_ridge',.2),('expected_steps',469),('evaluation_split','official_test'),('pmt_version','wrong'),('pmt_mode','other'),('dynamic_anchor',True)]:
        bad=copy.deepcopy(config);bad['jobs'][0][key]=val;rejected(lambda:cycle.validate(bad))
    math=envelope()
    high=[pmt.shadow('cpu','HK',mode,full_rank=i==2) for i,mode in enumerate(pmt.MODES)]
    spec=importlib.util.spec_from_file_location('committed_original',os.environ['Q215_ORIGINAL']);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    as_tensor=torch.as_tensor;empty=native.empty
    def tensor(*a,**k):
        if str(k.get('device','')).startswith('cuda'):k['device']='cpu'
        return as_tensor(*a,**k)
    def loader(*a,**k):
        for key in ('multiprocessing_context','prefetch_factor','persistent_workers'):k.pop(key,None)
        k['num_workers']=0;return DataLoader(*a,**k)
    steps=stages=preflights=0;maximum=0.;nonzero=False;unknown=False;_unused=None
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp,patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self), \
        patch.object(torch,'as_tensor',tensor),patch.object(torch.cuda,'manual_seed_all',lambda *a:None), \
        patch.object(torch.cuda,'max_memory_allocated',lambda *a:0),patch.object(native,'empty',lambda dim,device='cpu':empty(dim,'cpu')), \
        patch.object(cycle,'validate_job',lambda c:None),patch.object(pmt,'extract',paired_features):
        for ds in ('ISIC','HK'):
            rows=[dict(label=c,sample=i,relative_path=f'{c}_{i}',identity_component=f'{c}_{i}') for c in cycle.ORDERS[ds] for i in range(8)]
            for module in (original,production):
                module.ApartFeatures=Encoder;module.Images=SyntheticImages;module.IndexedImages=IndexedSynthetic;module.DataLoader=loader;module.manifests=lambda c:(rows,[])
            with patch.object(cycle,'read_training',lambda c:rows),patch.object(production.modules,'extract',features):
                for arm in ('base','local_transport'):
                    template=copy.deepcopy(next(j for j in config['jobs'] if j['name']==ds and j['module']==arm))
                    pos=count=0
                    for size in cycle.SIZES[ds]:
                        labels=cycle.ORDERS[ds][pos:pos+size];pos+=size
                        fit,_=production.fit_split([r for r in rows if r['label'] in labels],74002);count+=2*((len(fit)+63)//64)
                    outputs=[];modes=[None,None]+[mode for background,mode in pmt.MODES if background==arm]
                    for number,mode in enumerate(modes):
                        module=original if number==0 else production
                        c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_{number}'),expected_steps=count,max_steps=count,
                            original_deadline=time.time()+600,max_wall_seconds=600,paired_moment_transport=mode is not None,pmt_mode=mode or 'global')
                        if mode:
                            with patch.object(production,'manifests',side_effect=AssertionError('Early val manifest')):module.run(c)
                        else:module.run(c)
                        outputs.append(c['output']);steps+=count
                        assert json.loads(Path(c['output'],'STATUS.json').read_text())['status']=='TRAINED'
                    for task in range(1,len(cycle.SIZES[ds])+1):
                        a=torch.load(Path(outputs[0],f'stage_{task}.pt'),weights_only=False);b=torch.load(Path(outputs[1],f'stage_{task}.pt'),weights_only=False)
                        assert torch.equal(a['head'],b['head']) and all(torch.equal(a['model'][k],v) for k,v in b['model'].items())
                        assert all(torch.equal(a['bank'][k],b['bank'][k]) for k in ('mu','Q')) and a['bank']['n']==b['bank']['n']
                        for ca,cb in zip(a['bank']['components'],b['bank']['components']):
                            assert ca.keys()==cb.keys() and all(torch.equal(ca[key],cb[key]) if torch.is_tensor(ca[key]) else ca[key]==cb[key] for key in ca)
                        assert (a['module_memory'] is None and b['module_memory'] is None) or torch.equal(a['module_memory'],b['module_memory'])
                        for output in outputs[2:]:
                            st=torch.load(Path(output,f'stage_{task}.pt'),weights_only=False)
                            R,_=production.modules.metric(st['bank'],arm,ds,st['seen'],st['module_memory'])
                            h,audit=competition.bank_head(st['bank'],st['boundary_pair_weights'],regularizer=R)
                            assert (h-st['pmt_head_double']).norm()/h.norm()<1e-8 and torch.equal(st['pmt_head_double'].float(),st['head'])
                            assert st['bank']['_frame']==f'task_{task}_sealed' and st['pmt_frame']==st['bank']['_frame']
                            assert st['pmt_metadata']==cycle.checkpoint_metadata(dict(template,pmt_mode=st['pmt_metadata']['pmt_mode'],max_steps=count),task,st['steps'])
                            unknown|=any(c['radius'] is None for c in st['bank']['components'])
                            nonzero|=any(not torch.equal(st['model'][key],b['model'][key]) for key in st['model'])
                            maximum=max(maximum,audit['relative_residual']);stages+=1
                    for mode in modes[2:]:
                        c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_{mode}_pre'),preflight=True,workers=0,expected_steps=0,
                            original_deadline=time.time()+300,max_wall_seconds=300,pmt_mode=mode)
                        with patch.object(production,'manifests',side_effect=AssertionError('Early val manifest')):production.run(c)
                        f=json.loads(Path(c['output'],'PREFLIGHT.json').read_text());assert f['parameters_unchanged'] and f['optimizer_updates']==0 and f['gradient_norm']>0 and f['synthetic_highest_bank']['dimension']==6
                        preflights+=1
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
        gate=Path(tmp)/'gate';gate.write_text(json.dumps(dict(phase='TRAIN')));source=Path(tmp)/'config';source.write_text(json.dumps(config));os.environ['Q128_CONFIG']=str(source)
        rejected(lambda:cycle.check_barrier(dict(config['jobs'][0],evaluation_gate=str(gate))))
        gate.write_text(json.dumps(dict(phase='EVALUATE',training_sealed_at=1)))
        try:cycle.check_barrier(dict(config['jobs'][0],evaluation_gate=str(gate)))
        except FileNotFoundError:pass
        else:raise AssertionError('Missing stages accepted')
    from check_paired_transport_report import check
    report=check(config)
    assert unknown and nonzero and maximum<=1e-8 and preflights==6
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,test_accessed=False,val_reads=0,
        optimizer_updates_formal=0,synthetic_optimizer_updates=steps,synthetic_zero_update_preflights=preflights,
        baseline_disabled_parameters_heads_banks_components_regions_exact=True,checkpoint_roundtrip_stages=stages,
        unknown_radius_complete_training_chain=True,nonzero_training_effect=True,max_solve_residual=maximum,
        full_highdimensional_checks=high,envelope_and_invariants=math,report_checks=report,
        scope_budget_and_early_val_refused=True,cpu_pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False),flush=True)
