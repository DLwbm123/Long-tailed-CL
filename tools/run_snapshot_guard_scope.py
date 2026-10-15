"""Frozen four-new-trajectory scope; existing Core/LT/PMT controls are reused."""
import json
import os
from pathlib import Path
from run_paired_transport_cycle import validate_job as pmt_validate,checkpoint_metadata as pmt_metadata,ORDERS,SIZES,STEPS
from snapshot_guard_production import VERSION

MODES=(('base','global'),('local_transport','global_regional'))


def label(module,mode):return 'snapshot_guard' if module=='base' else 'local_transport_snapshot_guard'


def validate_job(c):
    pmt_validate(c)
    if (c.get('snapshot_guard_mode')!='on' or c.get('snapshot_guard_authorized') is not True
        or c.get('snapshot_guard_version')!=VERSION or c.get('guard_scoring_dtype')!='float64'
        or (c['module'],c['pmt_mode']) not in MODES or c.get('guard_solver')!='full_dual_fixed_v1'
        or c.get('pmt_control_source_commit')!='21660bf6161c5941aadcb98d569fdaeb8a111333'):
        raise ValueError('Frozen snapshot guard scope differs')
    if any(k.startswith('guard_') and k not in ('guard_scoring_dtype','guard_solver') for k in c):
        raise ValueError('No guard tuning/cap/budget knobs permitted')


def metadata(c,task,steps):
    return pmt_metadata(c,task,steps)|{k:c[k] for k in ('snapshot_guard_version','snapshot_guard_mode','guard_scoring_dtype','guard_solver')}


def validate(c):
    if (c.get('gpus')!=[0,1] or c.get('version')!=VERSION or len(c.get('jobs',[]))!=4
        or len(c.get('preflights',[]))!=4):raise ValueError('Only four guarded trajectories on GPU01')
    if {(j['name'],j['module'],j['pmt_mode']) for j in c['jobs']}!={(ds,m,a) for ds in ORDERS for m,a in MODES}:
        raise ValueError('Frozen four-way guard matrix differs')
    for j,p in zip(c['jobs'],c['preflights']):
        validate_job(j);validate_job(p)
        if not p.get('preflight') or p.get('job_name')!='pre_'+j['job_name']:raise ValueError('Zero update pairing differs')
        keys=('name','module','order','seed','split_seed','source_commit','pmt_mode','snapshot_guard_version','guard_scoring_dtype','manifest','images','weight','legacy_repo')
        if any(j[k]!=p[k] for k in keys):raise ValueError('Guard preflight mismatch')
    if len({j['output'] for j in c['jobs']+c['preflights']})!=8:raise ValueError('Distinct fresh outputs required')


def check_barrier(c):
    campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text());validate(campaign)
    gate=json.loads(Path(c['evaluation_gate']).read_text())
    if gate.get('phase')!='EVALUATE' or not gate.get('training_sealed_at'):raise ValueError('Guard val sealed')
    for j in campaign['jobs']:
        s=json.loads((Path(j['output'])/'STATUS.json').read_text())
        if (s['status'] not in ('TRAINED','COMPLETE') or s['steps']!=j['expected_steps']
            or s['stages']!=len(j['task_sizes']) or s['policy_updates']):raise ValueError('Four complete training records required')
        expected=[metadata(j,t,sum(STEPS[j['name']][:t])) for t in range(1,len(j['task_sizes'])+1)]
        if json.loads((Path(j['output'])/'GUARD_TRAINING_SEAL.json').read_text())!=expected:raise ValueError('Guard all-stage seals differ')
        old=[pmt_metadata(j,t,sum(STEPS[j['name']][:t])) for t in range(1,len(j['task_sizes'])+1)]
        if json.loads((Path(j['output'])/'TRAINING_SEAL.json').read_text())!=old:raise ValueError('PMT all-stage seals differ')
