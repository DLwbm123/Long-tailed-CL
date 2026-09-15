"""Bounded P0 -> P1 -> six P2 forks, validation reporting, then stop."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from run_medical_v2 import sha,write_json

BUDGET=8*3600

def verify(config):
    root=Path(config['code_root']);output=Path(config['output'])
    for name,expected in config['code_sha256'].items():assert sha(root/name)==expected,'BLOCKED_V3_CODE_DRIFT: '+name
    v2=json.loads(Path(config['v2_runtime']).read_text());v2root=Path(config['v2_runtime']).parent/'code'
    for name,expected in v2['code_sha256'].items():assert sha(v2root/name)==expected,'BLOCKED_V2_SOURCE_DRIFT: '+name
    summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text())
    for split,expected in summary['manifest_sha256'].items():assert sha(Path(config['protocol'])/(split+'.csv'))==expected,'BLOCKED_MANIFEST_DRIFT'
    assert [r['n_train_images'] for r in sorted(summary['class_counts'],key=lambda r:r['original_label'])]==[10529,3263,2835,1306,458,256,44,27]
    assert sha(config['weight'])=='c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800'
    assert config['class_orders']=={'1993':[4,0,3,7,5,6,2,1],'1994':[3,7,4,5,1,0,6,2],'1995':[1,3,0,6,4,5,2,7]}
    assert json.loads((Path(config['v2_output'])/'FINAL_STATUS.json').read_text())['status']=='COMPLETE_MIXED_SIGNAL'
    # Storage gate counts all 12 final states, six resumes and two atomic writes. No deletion/precision changes.
    old=list(Path(config['v2_output']).glob('199?_[BC]/session*.pt'));size=max(p.stat().st_size for p in old)
    # A's two float32 feature arrays occupy 58.4 MB. B was explicitly allowed to be
    # unavailable; do not reserve its undownloadable weight and cache as if present.
    auxiliary=450*1024**2 if (Path(config['v3_root'])/'weights/model.safetensors').exists() else 128*1024**2
    free=shutil.disk_usage(output).free;required=20*size+auxiliary+1024**3
    assert free>required, f'BLOCKED_STORAGE free={free}, required={required}'
    return {'free_bytes':free,'required_bytes':required,'checkpoint_bytes_upper':size,'auxiliary_bytes':auxiliary,'additional_reserve_bytes':1024**3,
            'retention':'12 final checkpoints + 6 resume files + 2 atomic write buffers; V2 untouched','test_predictions':0}

def recompute_v2(config):
    p=Path(config['v2_output'])/'results';rows=list(csv.DictReader((p/'per_class_metrics.csv').open()));summary=[]
    for seed in (1993,1994,1995):
        for variant in ('M0','M1','M2','M3'):
            rs=[r for r in rows if int(r['train_seed'])==seed and r['eval_variant']==variant and int(r['session'])==2]
            assert len(rs)==8
            value=sum(float(r['recall']) for r in rs)/8
            summary.append({'seed':seed,'variant':variant,'final_test_ba_readonly':value,
                            'old_recall':sum(float(r['recall']) for r in rs if int(r['head_index'])<6)/6,
                            'current_recall':sum(float(r['recall']) for r in rs if int(r['head_index'])>=6)/2})
    write_json(Path(config['output'])/'p0_existing_test_aggregate_recalculation.json',{'existing_aggregate_only':True,'new_test_predictions':0,'rows':summary})

def worker(config,phase,seed=None,branch=None):
    import torch
    torch.set_num_threads(4)
    if phase=='p0':
        from diagnose_medical_v3 import p0
        p0(config)
    elif phase=='p1':
        from frozen_medical_v3 import p1
        p1(config)
    elif phase in ('pilot','train'):
        from train_medical_v3 import trajectory
        trajectory(config,seed,branch,pilot=phase=='pilot')
    else:raise ValueError(phase)

def supervisor(config):
    out=Path(config['output']);start=time.time();resources=[];storage=verify(config);recompute_v2(config)
    assert json.loads((out/'engineering/ENGINEERING.json').read_text())['status']=='ENGINEERING_PASS'
    assert json.loads((out/'engineering/PREFLIGHT_BUDGET.json').read_text())['status']=='PASS'
    write_json(out/'STORAGE_GATE.json',storage)
    entry=config['neutral_worker'];active={}
    def launch(phase,seed=None,branch=None):
        name=phase if seed is None else f'{phase}_{seed}_{branch}';env=os.environ.copy()
        env['N3_JOB']=json.dumps({'phase':phase,'seed':seed,'branch':branch});env['N3_CONFIG']=config['runtime_path']
        log=open(out/(name+'.log'),'a');p=subprocess.Popen([sys.executable,entry],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[p.pid]=(p,log,name,time.time());return p
    def gpu_sample():
        r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
        with (out/'gpu_resource.csv').open('a') as f:f.write(r.stdout)
    def collect(p):
        process,log,name,t=active.pop(p.pid);log.close();r={'job':name,'seconds':time.time()-t,'returncode':p.returncode};resources.append(r)
        write_json(out/'resource_usage.json',{'jobs':resources,'physical_wall_seconds':time.time()-start,'conservative_process_seconds':sum(r['seconds'] for r in resources),'budget_seconds':BUDGET})
        if p.returncode:raise RuntimeError('BLOCKED_WORKER '+name+' exit='+str(p.returncode)+'; see '+name+'.log')
    def run(phase,seed=None,branch=None):
        write_json(out/'STATUS.json',{'status':'RUNNING','phase':phase,'seed':seed,'branch':branch,'test_predictions':0})
        p=launch(phase,seed,branch)
        while p.poll() is None:gpu_sample();time.sleep(30)
        collect(p)
    try:
        # The resource projection is rechecked with a complete formal epoch before launching all pairs.
        for phase in ('p0','p1'):
            if not (out/phase/'COMPLETE.json').exists():run(phase)
        if not (out/'1993_E/PILOT.json').exists():run('pilot',1993,'E')
        pilot=json.loads((out/'1993_E/PILOT.json').read_text());row=pilot['epoch'];components=row['components']
        counts=[10529,3263,2835,1306,458,256,44,27]
        n_increment=sum(sum(counts[c] for c in order[4:]) for order in config['class_orders'].values())
        rate=components['training_seconds']/components['train_images']
        train_estimate=2*10*n_increment*rate
        probe_overhead=max(0.,row['seconds']-components['training_seconds'])*120
        memory_estimate=n_increment*rate
        checkpoint_estimate=132*2.
        consumed=sum(r['seconds'] for r in resources)
        estimate=consumed+1.35*(train_estimate+probe_overhead+memory_estimate+checkpoint_estimate)
        budget={'status':'PASS' if estimate<=BUDGET else 'BLOCKED_BUDGET','budget_seconds':BUDGET,'consumed_process_seconds':consumed,
                'pilot_seconds_per_image':rate,'incremental_images_per_branch_all_seeds':n_increment,
                'training_estimate_seconds':train_estimate,'fixed_probe_estimate_seconds':probe_overhead,'memory_estimate_seconds':memory_estimate,
                'checkpoint_estimate_seconds':checkpoint_estimate,'reserve_factor':1.35,'total_conservative_estimate_seconds':estimate,
                'pilot_peak_bytes':pilot['peak_allocated_bytes'],'new_epochs':120,'pilot_is_first_formal_epoch':True}
        write_json(out/'BUDGET_GATE.json',budget)
        assert estimate<=BUDGET, 'BLOCKED_BUDGET: '+json.dumps(budget)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
        needed=2*(pilot['peak_allocated_bytes']+1024**3)+512*1024**2
        slots=2 if free>needed else 1
        write_json(out/'PARALLEL_GATE.json',{'free_gpu_bytes':free,'two_process_estimate_bytes':needed,'slots':slots,'maximum':2})
        queue=[(seed,branch) for seed in (1993,1994,1995) for branch in ('D','E')]
        write_json(out/'STATUS.json',{'status':'RUNNING','phase':'P2','slots':slots,'test_predictions':0})
        while queue or active:
            while queue and len(active)<slots:
                seed,branch=queue.pop(0);launch('train',seed,branch)
            gpu_sample();time.sleep(30)
            for p,_,_,_ in list(active.values()):
                if p.poll() is not None:collect(p)
        from report_medical_v3 import report
        result=report(config);write_json(out/'STATUS.json',result)
        resource={'jobs':resources,'physical_wall_seconds':time.time()-start,'conservative_process_seconds':sum(r['seconds'] for r in resources),'budget_seconds':BUDGET,'gpu_samples':'gpu_resource.csv','maximum_training_processes':slots}
        write_json(out/'resource_usage.json',resource)
        (out/'results/RESOURCE_REPORT.md').write_text('# Resource use\n\n'+json.dumps(resource,indent=2)+'\n\nProcess seconds sum overlapping workers conservatively; physical wall time is separately reported. GPU samples include memory, utilization and power. No other process was terminated.\n')
    except Exception as error:
        # Already-running authorized workers finish their current complete trajectories; no kill or silent restart.
        write_json(out/'STATUS.json',{'status':'BLOCKED','error':repr(error),'traceback':traceback.format_exc(),'active_workers':[v[2] for v in active.values()],'test_predictions':0})
        raise
