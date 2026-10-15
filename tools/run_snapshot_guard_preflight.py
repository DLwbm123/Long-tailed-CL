"""Preflight-only adapter around the unchanged production run; never dispatch train/eval."""
import copy
import json
import os
from pathlib import Path
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

import torch
import run_module_additions as training
import run_snapshot_guard_scope as scope
import snapshot_guard_production as guard
import check_snapshot_guard_production as fixture


def validate(c):
    scope.validate_job(c)
    if (c.get('operation')!='preflight_only' or c.get('preflight') is not True
        or c.get('GPU_execution_authorized') is not True or c.get('formal_training_authorized') is not False
        or c.get('expected_steps')!=0 or c.get('workers')!=0):
        raise ValueError('Only genuine zero-update preflight is authorized')


def forbidden_step(*args,**kwargs):
    raise RuntimeError('Optimizer step forbidden by preflight-only authorization')


def move(v,device):
    if torch.is_tensor(v):return v.to(device)
    if isinstance(v,dict):return {k:move(x,device) for k,x in v.items()}
    if isinstance(v,list):return [move(x,device) for x in v]
    if isinstance(v,tuple):return tuple(move(x,device) for x in v)
    return v


def shadow(case,campaign,device):
    original=guard.Session.project;transfers=[]
    def project(session,head,bank,pairs,R,off,local,x,y,*args):
        inputs=move((head,bank,pairs,R,off,local,x,y),device)
        if device!='cpu':torch.cuda.synchronize()
        w,cert=original(session,*inputs,*args)
        if device!='cpu':torch.cuda.synchronize()
        transfers.append(session.events[-1]['logical_host_input_tensor_bytes'])
        return w.cpu(),cert
    started=time.monotonic();threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        with patch.object(guard.Session,'project',project):r=fixture.large(case,campaign)
    finally:torch.set_num_threads(threads)
    assert r['status']=='PASS' and r['frozen_math_certificate_status']=='CERTIFIED'
    return {k:r[k] for k in ('status','all_constraints_n','positive_duals','minimum_raw_slack','stationarity',
        'complementarity','primal_dual_gap','observed_support_rank','H_inverse_calls','retained_gain',
        'discriminative_delta','identity','input_statistics_unchanged','checkpoint_bytes','checkpoint_rebuild')}|dict(
        synthetic_only=True,dimension=case['dimension'],old_classes=case['old_classes'],seen_classes=case['classes'],
        fit_n=case['fit_n'],device_inputs=device,logical_host_input_bytes=transfers[-1],optimizer_updates=0,
        formal_statistics_written=False,operation_wall_seconds=time.monotonic()-started)


def run(c,campaign,lock):
    validate(c);seen=[];created=[];original_encoder=training.ApartFeatures;original_shadow=training.pmt.shadow
    def encoder(*a,**kw):
        m=original_encoder(*a,**kw);created.append((m,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}));return m
    def highest(*a,**kw):
        r=original_shadow(*a,**kw)
        case=next(v for v in lock['cases'] if v.get('dataset')==c['name'] and v.get('LT')==(c['module']=='local_transport') and v.get('event')=='boundary_final')
        seen.append(shadow(case,campaign,'cuda:0'));return r
    blocked_roots=[c[k] for k in ('control_checkpoint_root','pmt_control_checkpoint_root')]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0]);base=Path(name).name.lower()
            if base in ('val.csv','test.csv','reserved.csv') or any(name.startswith(p+'/') for p in blocked_roots):
                raise RuntimeError('Validation/test/historical checkpoint access forbidden')
    sys.addaudithook(audit)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        for context in (patch.object(training,'ApartFeatures',encoder),patch.object(training.pmt,'shadow',highest),
            patch.object(torch.optim.AdamW,'step',forbidden_step),
            patch.object(training,'manifests',side_effect=RuntimeError('No validation manifests in preflight'))):stack.enter_context(context)
        training.run(c)
    torch.cuda.synchronize();assert len(created)==len(seen)==1
    m,before=created[0];parameters=dict(m.named_parameters());buffers=dict(m.named_buffers())
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items())
    out=Path(c['output']);r=json.loads((out/'PREFLIGHT.json').read_text());status=json.loads((out/'STATUS.json').read_text())
    assert r['status']=='PASS' and r['optimizer_updates']==status['steps']==r['policy_updates']==0
    assert r['parameters_unchanged'] and r['test_accessed'] is False and r['snapshot_guard']['epoch_event']['status']=='CERTIFIED'
    assert all(__import__('math').isfinite(v) and v>0 for v in r['adapter_pool_gradients'].values())
    assert not list(out.glob('stage_*.pt')) and not (out/'metrics.json').exists()
    # Only scalar groups leave the private raw preflight receipt.
    keep=('status','encoder_dim','adapter_updates','optimizer_updates','policy_updates','task','epoch','gradient_norm','loss','batch_n','parameters_unchanged','test_accessed','max_solve_residual','adapter_pool_gradients')
    result={k:r[k] for k in keep}|dict(source_commit=c['source_commit'],production_math_source='7a077b69f16d843573b2bc7af5a46d9701c29bb0',
        parameters_exactly_unchanged=True,buffers_exactly_unchanged=True,parameter_tensors=len(parameters),buffer_tensors=len(buffers),
        real_first_batch=True,real_first_batch_highest_old_bank=False,real_guard_certificate=r['snapshot_guard']['epoch_event']['status'],
        PMT_highest_bank={k:r['synthetic_highest_bank'][k] for k in ('synthetic_only','optimizer_updates','formal_statistics_written','dimension','full_rank_memory_probe','old_classes','seen_classes','virtual_moment_error','relative_residual')},
        guard_highest_bank=seen[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        operation_wall_seconds=time.monotonic()-started,formal_updates=0,validation_accessed=False,formal_training_started=False)
    (out/'PREFLIGHT_SCALARS.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    return result


def cpu_check(c,campaign):
    validate(c)
    for changes in ({'preflight':False},{'operation':'train'},{'operation':'evaluate'},{'expected_steps':1},{'GPU_execution_authorized':False},{'formal_training_authorized':True}):
        try:validate(c|changes)
        except ValueError:pass
        else:raise AssertionError('Unauthorized operation accepted')
    model=torch.nn.Linear(2,2);opt=torch.optim.AdamW(model.parameters());before=copy.deepcopy(model.state_dict())
    try:forbidden_step(opt)
    except RuntimeError:pass
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    case=dict(dataset='ISIC',dimension=6,classes=8,old_classes=6,fit_n=16,old_train_counts=[3]*6,current_train_counts=[10,10],event='boundary_final',task=3,LT=False)
    r=shadow(case,campaign,'cpu');assert r['all_constraints_n']==112 and r['logical_host_input_bytes']==0
    return dict(status='PASS_CPU_HARNESS_ONLY',operation_denials=True,step_denial=True,small_shadow=r,GPU_processes=0,medical_images_read=0,test_accessed=False)


if __name__=='__main__':
    campaign=json.loads(Path(os.environ['Q247_CONFIG']).read_text());lock=json.loads(Path(os.environ['Q247_LOCK']).read_text())
    if os.environ.get('Q247_CPU_CHECK')=='1':print(json.dumps(cpu_check(campaign['preflights'][0],campaign)))
    else:run(json.loads(os.environ['Q247_JOB']),campaign,lock)
