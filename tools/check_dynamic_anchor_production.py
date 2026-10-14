"""Full synthetic call-chain CPU regression, including both unchanged controls."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import run_module_additions as production
import run_dynamic_anchor_cycle as cycle
import dynamic_geometry_anchor as anchor
import prototype_coherent as native
from check_lca_production import SyntheticImages,IndexedSynthetic,Encoder,features,rejected
from check_dynamic_geometry_anchor import run as math_check


def run():
    start=time.monotonic();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    math_result=math_check();config=json.loads(Path(os.environ['Q185_CPU_CONFIG']).read_text());cycle.validate(config)
    for key,val in [('gpus',[0,2]),('jobs',config['jobs'][:4])]:rejected(lambda:cycle.validate(dict(config,**{key:val})))
    for key,val in [('anchor_relative_curvature',.2),('expected_steps',469),('evaluation_split','official_test'),('anchor_version','wrong'),('anchor_mode','other')]:
        bad=copy.deepcopy(config);bad['jobs'][0][key]=val;rejected(lambda:cycle.validate(bad))
    spec=importlib.util.spec_from_file_location('committed_original',os.environ['Q185_ORIGINAL'])
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    as_tensor=torch.as_tensor;empty=native.empty
    def tensor(*a,**k):
        if str(k.get('device','')).startswith('cuda'):k['device']='cpu'
        return as_tensor(*a,**k)
    def loader(*a,**k):
        for key in ('multiprocessing_context','prefetch_factor','persistent_workers'):k.pop(key,None)
        k['num_workers']=0;return DataLoader(*a,**k)
    steps=checkpoints=preflights=0;maximum=0.;active=False;parameter_different=False
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp,patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self), \
        patch.object(torch,'as_tensor',tensor),patch.object(torch.cuda,'manual_seed_all',lambda *a:None), \
        patch.object(torch.cuda,'max_memory_allocated',lambda *a:0),patch.object(native,'empty',lambda dim,device='cpu':empty(dim,'cpu')), \
        patch.object(cycle,'validate_job',lambda c:None):
        for ds in ('ISIC','HK'):
            rows=[dict(label=c,sample=i,relative_path=f'{c}_{i}',identity_component=f'{c}_{i}') for c in cycle.ORDERS[ds] for i in range(8)]
            for module in (original,production):
                module.ApartFeatures=Encoder;module.Images=SyntheticImages;module.IndexedImages=IndexedSynthetic;module.DataLoader=loader
                module.manifests=lambda c:(rows,[])
            with patch.object(cycle,'read_training',lambda c:rows),patch.object(production.modules,'extract',features):
                for arm in ('base','local_transport'):
                    template=copy.deepcopy(next(j for j in config['jobs'] if j['name']==ds and j['module']==arm))
                    pos=count=0
                    for size in cycle.SIZES[ds]:
                        labels=cycle.ORDERS[ds][pos:pos+size];pos+=size
                        fit,_=production.fit_split([r for r in rows if r['label'] in labels],74002);count+=2*((len(fit)+63)//64)
                    outputs=[]
                    for number,(module,mode) in enumerate([(original,None),(production,None),(production,'dynamic'),(production,'static')]):
                        c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_{number}'),expected_steps=count,max_steps=count,
                            original_deadline=time.time()+300,max_wall_seconds=300,dynamic_anchor=mode is not None,anchor_mode=mode or 'dynamic')
                        if mode:
                            with patch.object(production,'manifests',side_effect=AssertionError('Early val manifest')):module.run(c)
                        else:module.run(c)
                        outputs.append(c['output']);steps+=count
                        assert json.loads(Path(c['output'],'STATUS.json').read_text())['status']=='TRAINED'
                    for task in range(1,len(cycle.SIZES[ds])+1):
                        a=torch.load(Path(outputs[0],f'stage_{task}.pt'),weights_only=False);b=torch.load(Path(outputs[1],f'stage_{task}.pt'),weights_only=False)
                        assert torch.equal(a['head'],b['head']) and all(torch.equal(a['model'][k],v) for k,v in b['model'].items())
                        assert all(torch.equal(a['bank'][k],b['bank'][k]) for k in ('mu','Q')) and a['bank']['n']==b['bank']['n']
                        for output in outputs[2:]:
                            st=torch.load(Path(output,f'stage_{task}.pt'),weights_only=False)
                            R,_=production.modules.metric(st['bank'],arm,ds,st['seen'],st['module_memory'])
                            state=anchor.restore(st['anchor_state'],6,len(st['seen']),R)
                            head,audit=anchor.bank_head(st['bank'],st['boundary_pair_weights'],R,state)
                            assert (head-st['anchor_head_double']).norm()/head.norm()<1e-8 and torch.equal(st['anchor_head_double'].float(),st['head'])
                            active |= bool(state['weights'].any());parameter_different |= any(not torch.equal(st['model'][k],a['model'][k]) for k in a['model'])
                            maximum=max(maximum,audit['relative_residual']);checkpoints+=1
                    for mode in ('dynamic','static'):
                        c=copy.deepcopy(template);c.update(output=str(Path(tmp)/f'{ds}_{arm}_{mode}_pre'),preflight=True,workers=0,expected_steps=0,
                            original_deadline=time.time()+300,max_wall_seconds=300,anchor_mode=mode)
                        with patch.object(production,'manifests',side_effect=AssertionError('Early val manifest')):production.run(c)
                        receipt=json.loads(Path(c['output'],'PREFLIGHT.json').read_text())
                        assert receipt['parameters_unchanged'] and receipt['optimizer_updates']==0 and receipt['gradient_norm']>0
                        preflights+=1
    # Fail before val access; exact gate and all eight statuses are mandatory.
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
        gate=Path(tmp)/'gate';gate.write_text(json.dumps(dict(phase='TRAIN')))
        source=Path(tmp)/'config';source.write_text(json.dumps(config));os.environ['Q128_CONFIG']=str(source)
        rejected(lambda:cycle.check_barrier(dict(config['jobs'][0],evaluation_gate=str(gate))))
        gate.write_text(json.dumps(dict(phase='EVALUATE',training_sealed_at=1)))
        try:cycle.check_barrier(dict(config['jobs'][0],evaluation_gate=str(gate)))
        except FileNotFoundError:pass
        else:raise AssertionError('Missing training statuses accepted')
    from summarize_dynamic_anchor import report,NEW,MODULES
    from run_dynamic_anchor_worker import paired
    import numpy as np
    yy=np.array([0,0,1,1]);off=np.array([0,1,1,0]);on=np.array([1,0,1,0]);ss=np.ones((4,2))
    pp=paired(yy,[0,1],on,off,ss,ss,1)
    assert pp['classes'][0]['rescued']==1 and pp['classes'][0]['damaged']==1
    assert all(r['correct_on']-r['correct_control']==r['rescued']-r['damaged'] for r in pp['classes'])
    snapshot=dict(program=dict(status='COMPLETE',active={},failures=[],elapsed_seconds=0.,costs=[dict(returncode=0,gpu=i%2,residence_seconds=0.,operation='synthetic',job=str(i)) for i in range(24)]),
        source_commit='synthetic',lock=json.loads(Path(os.environ['Q185_CPU_PROTOCOL']).read_text()),runs={},preflights={},
        round_name='synthetic',cpu_checks={},engineering_events=[],closed_round_cost_ledger={})
    from lt_benchmark import benchmark_metrics,task_blocks
    for ds in cycle.ORDERS:
        for arm in MODULES:
            stages=[]
            for task in range(1,len(cycle.SIZES[ds])+1):
                seen=cycle.ORDERS[ds][:sum(cycle.SIZES[ds][:task])];counts={str(c):snapshot['lock']['data'][ds]['val_counts'][str(c)] for c in seen}
                classes=[dict(label=c,n=counts[str(c)],role='old' if i<len(seen)-cycle.SIZES[ds][task-1] else 'new',correct_on=0,correct_control=0,rescued=0,damaged=0,true_margin_mean_on=0.,true_margin_mean_control=0.) for i,c in enumerate(seen)]
                stages.append(dict(task=task,seen=seen,validation_n=sum(counts.values()),accuracy=0.,balanced_accuracy=0.,per_class_n=counts,per_class_recall={str(c):0. for c in seen},tail_recall=0.,solve_residual=0.,checkpoint_rebuild_relative_error=0.,paired=dict(classes=classes,confusion={})))
            metrics=dict(stages=stages,**benchmark_metrics(stages,task_blocks(cycle.ORDERS[ds],cycle.SIZES[ds]),{int(c):n for c,n in snapshot['lock']['data'][ds]['train_counts'].items()}),
                **{key:0. for key in ('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall')},
                test_accessed=False,source_commit='synthetic',anchor_version=anchor.VERSION,module=arm,checkpoint_head_rebuild_verified=True,paired_control_inference_recall_verified=True)
            diagnostics=[dict(epoch=ep,max_batch_solve_residual=0.,dynamic_anchor=dict(fit_only=True,detached=True),boundary_dynamic_anchor=dict(meta_excluded_from_target=True)) for t in cycle.SIZES[ds] for ep in (1,2)]
            snapshot['runs'][ds+'_'+arm]=dict(status=dict(status='COMPLETE',steps=sum(cycle.STEPS[ds]),stages=len(cycle.SIZES[ds]),policy_updates=0,max_solve_residual=0.),metrics=metrics,scalar_diagnostics=diagnostics)
            if arm in NEW:snapshot['preflights'][ds+'_'+arm]=dict(receipt_status='PASS',receipt_complete=True,optimizer_updates=0,policy_updates=0,parameters_unchanged=True,test_accessed=False,gradient_norm=1.,max_solve_residual=0.)
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
        report(snapshot,Path(tmp)/'good')
        bad=copy.deepcopy(snapshot);bad['runs']['ISIC_dynamic_anchor']['metrics']['stages'][0]['per_class_n']['4']+=1
        rejected(lambda:report(bad,Path(tmp)/'bad'))
    assert active and parameter_different and maximum<=1e-8
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,test_accessed=False,val_reads=0,
        optimizer_updates_formal=0,synthetic_optimizer_updates=steps,synthetic_zero_update_preflights=preflights,
        baseline_disabled_parameters_heads_and_banks_exact=True,checkpoint_roundtrip_stages=checkpoints,
        frozen_fit_only_targets=True,nonzero_training_effect=True,max_solve_residual=maximum,scope_budget_and_early_val_refused=True,report_all_signs_and_bad_denominator_checked=True,paired_rescue_damage_checked=True,
        math_checks=math_result,cpu_pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
