"""NumPy-only synthetic signed-count report and denominator rejection."""
import json,copy,tempfile,time
from pathlib import Path
import numpy as np
from concm_statistical_metrics import stage
from collect_concm_mechanism import report,validate,JOBS,HEADS

def check():
    start=time.monotonic();r=Path('docs/concm_mechanism1_20261013');lock=json.loads((r/'PROTOCOL_LOCK.json').read_text());prior=dict(runs={v:{} for v in ('MPC','DSM','MPC_DSM')});runs={};prepared={};pre={}
    for name in JOBS:
        d=lock['data'][name.split('_')[0]];k=len(d['order']);task=len(d['task_sizes']);labels=np.concatenate([np.full(d['val_counts'][str(c)],c) for c in d['order']]);y=np.array([d['order'].index(int(c)) for c in labels]);off=np.eye(k)[y];on=off.copy();on[-1]=0;on[-1,0]=2
        reps={}
        for v in HEADS:
            a=stage(off,off if v=='IDENTITY' else on,off if v=='IDENTITY' else on,labels,d,task)
            a.update(mechanical=dict(solver=dict(relative_residual=0),norms=dict(full_norm=2.,discriminative_norm=1.,common_norm=3**.5,common_energy_fraction=.75),reference_norms={},structure_coefficient=.1),normalized_fixed_competitor_margin={g:dict(off=z['off'],on=z['on'],delta=z['applied_delta']) for g,z in a['margins'].items()})
            reps[v]=a
            if v in ('MPC','ETF_FULL','MPC_DSM'):prior['runs'][{'ETF_FULL':'DSM'}.get(v,v)][name]=dict(stages=[a])
        lock['control_references'][name]['stages'][-1]=reps['MPC']['metrics']['OFF']
        runs[name]=dict(reports=reps,head_comparisons={on+'−'+off:reps['MPC'] for on,off in [('ETF_FULL','REG_ZERO'),('ETF_DISC','ETF_FULL'),('MPC_DSM','MPC'),('MPC_DSM','ETF_FULL')]},donor_summary=[],status=dict(status='COMPLETE'),source_commit='fixture',version='concm_final_mechanism_diagnostic_v1',optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False)
        pre[name]=dict(status='PASS',parameters_unchanged=True,CPU_GPU_scores_close=True,classification_metrics_read=False,source_commit='fixture',version='concm_final_mechanism_diagnostic_v1',optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False)
        prepared[name]=dict(status='PREPARED',source_commit='fixture',stages=[dict(variants={v:{} for v in HEADS})])
    costs=[dict(job=n,operation=op,returncode=0,process_started=True,gpu=i%2,residence_seconds=1.) for i,n in enumerate(JOBS) for op in ('preflight','evaluate')]
    s=dict(lock=lock,prior=prior,runs=runs,prepared=prepared,preflights=pre,source_commit='fixture',program=dict(status='COMPLETE',active={},costs=costs,cpu_costs=[dict(residence_seconds=1.) for n in JOBS],failures=[]),ledger={},cpu_checks=dict(process_residence_seconds=1.,pipeline_seconds=.1),engineering_events=[])
    assert validate(s)
    bad=copy.deepcopy(s);bad['runs']['HK_base']['reports']['MPC']['metrics']['ON']['per_class_n']['5']=2
    try:validate(bad)
    except ValueError:pass
    else:raise AssertionError('Wrong denominator accepted')
    with tempfile.TemporaryDirectory() as tmp:
        a=report(s,Path(tmp));assert a['status']=='PASS';results=json.loads((Path(tmp)/'results/RESULTS.json').read_text());assert len(results['rows'])==24;assert all(z['new_harmed']==1 for z in results['rows'] if z['head']!='IDENTITY')
    return dict(status='PASS',synthetic_only=True,medical_images_read=0,GPU_processes=0,complete_24_row_report=True,all_signed_new_counts=True,wrong_denominator_refused=True,pipeline_seconds=time.monotonic()-start)
if __name__=='__main__':print(json.dumps(check(),indent=2))
