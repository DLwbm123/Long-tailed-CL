"""Complete the already-approved B baseline after P2; preserve the partial delivery."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from run_medical_v2 import sha,write_json
from frozen_medical_v3 import extract,baselines,DINO_SHA,DINO_REV
from evaluate_medical_v2 import write_csv


def complete(config,out):
    old=Path(config['output']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=Path(config['v3_root']);start=time.time()
    assert not (out/'DINO_SUPPLEMENT_LOCK.json').exists(),'Supplement already exists'
    weight=json.loads((p/'weights/VERIFIED_WEIGHT_LOCK.json').read_text())
    assert weight['revision']==DINO_REV and weight['files']['model.safetensors']['sha256']==DINO_SHA
    loading=json.loads((p/'weights/LOADING_AUDIT.json').read_text());assert loading['status']=='PASS'
    for name,h in config['code_sha256'].items():assert sha(Path(config['code_root'])/name)==h
    assert sha(old/'V3_SCOPE_AND_LOCK.json')==config['protocol_sha256']
    estimated=json.loads((old/'BUDGET_GATE.json').read_text())['total_conservative_estimate_seconds']
    assert estimated+900<=28800,'BLOCKED_SUPPLEMENT_BUDGET'
    lock={'status':'QUEUED','authorization':'User requested local/my-gpu download and transfer to complete the already approved DINOv2 baseline.',
          'parent_output':str(old),'parent_scope_sha256':config['protocol_sha256'],'P2_training_source_commit':config['code_commit'],
          'supplement_source_sha256':sha(__file__),'weight':weight,'loading':loading,
          'extra_phase':'P1 B only: one train/val feature extraction, fixed F-NCM and F-CBRidge; reuse A and P2 results.',
          'test_predictions':0,'reserved_extra_gpu_seconds':900,'conservative_total_with_supplement_seconds':estimated+900,
          'no_original_artifact_overwrite':True,'wait_for':'original supervisor to stop and fewer than two active GPU processes'}
    write_json(out/'DINO_SUPPLEMENT_LOCK.json',lock)
    write_json(out/'STATUS.json',{'status':'QUEUED_AFTER_P2','test_predictions':0})
    try:
        while True:
            original_status=json.loads((old/'STATUS.json').read_text())
            ended=(old/'FINAL_STATUS.json').exists() or original_status.get('status')=='BLOCKED' or (old/'LAUNCH_ERROR.json').exists()
            processes=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
            original_workers=any(line.strip().endswith('/tmp/n3w2.py') for line in processes)
            gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
            if ended and not original_workers and len(gpu_pids)<2:break
            time.sleep(30)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
        assert free>4*1024**3,'BLOCKED_SUPPLEMENT_GPU_MEMORY'
        assert shutil.disk_usage(out).free>200*1024**2+1024**3,'BLOCKED_SUPPLEMENT_DISK'
        import torch
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();work=time.time()
        write_json(out/'STATUS.json',{'status':'RUNNING_P1_B','started_at':work,'test_predictions':0})
        folder=out/'p1';folder.mkdir();b=folder/'B';b.mkdir()
        extract(config,'B',b);metrics,classes,equivalence=baselines(config,'B',b)
        # Reuse A artifacts without any repeated feature extraction or fitting.
        (folder/'A').symlink_to(old/'p1/A',target_is_directory=True)
        for name,rows in [('frozen_val_metrics.csv',metrics),('frozen_per_class_metrics.csv',classes)]:
            with (old/'p1'/name).open() as f:a=list(csv.DictReader(f))
            write_csv(folder/name,a+rows)
        previous=json.loads((old/'p1/COMPLETE.json').read_text())
        encoders=[r for r in previous['encoders'] if r['encoder']=='A']+[{'encoder':'B','status':'COMPLETE','equivalence':equivalence}]
        write_json(folder/'COMPLETE.json',{'encoders':encoders,'test_predictions':0,'original_partial_report_preserved':str(old/'p1/COMPLETE.json')})
        lines=['# Completed frozen representation comparison','','A is reused unchanged. B uses the originally specified revision and SHA256, now recovered through local download and transfer. No new model or hyperparameter was introduced. Final eight-class numbers represent one deterministic fit, not three independent repetitions.','','| Encoder | Classifier | Final validation BA |','|---|---|---:|']
        with (folder/'frozen_val_metrics.csv').open() as f:
            for r in csv.DictReader(f):
                if r['session']=='2' and r['order_seed']=='1993':lines.append(f"| {r['encoder']} | {r['classifier']} | {float(r['balanced_accuracy']):.4f} |")
        lines+=['','Both use final normalized CLS and per-sample L2, RGB 224-square bicubic/antialias, encoder-specific normalization. CBRidge uses lambda=0.001, no bias, float64 class-balanced sufficient statistics. Stage access, final order invariance and batch/stream equivalence passed. Detailed per-class and lesion-equal metrics are supplied.','','New test predictions: 0.']
        (folder/'P1_FROZEN_FEATURE_BASELINES.md').write_text('\n'.join(lines)+'\n')
        resource={'queue_seconds':work-start,'supplement_seconds':time.time()-work,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                  'original_resource_report':str(old/'resource_usage.json'),'test_predictions':0}
        assert resource['supplement_seconds']<=900,'BLOCKED_SUPPLEMENT_RUNTIME_BUDGET'
        write_json(out/'RESOURCE_SUPPLEMENT.json',resource)
        if (old/'FINAL_STATUS.json').exists():
            for name in ['p0']+[f'{s}_{br}' for s in (1993,1994,1995) for br in ('D','E')]:
                (out/name).symlink_to(old/name,target_is_directory=True)
            shutil.copyfile(old/'V3_SCOPE_AND_LOCK.json',out/'V3_SCOPE_AND_LOCK.json')
            from report_medical_v3 import report
            result=report(dict(config,output=str(out)))
            # The original generic report predates availability recovery. Preserve it and
            # explicitly identify the historical limitation as resolved in this delivery.
            decision=out/'results/V3_DECISION.md'
            text=decision.read_text().replace('DINOv2 unavailability limits the cross-encoder comparison where marked.',
                                            'The original DINOv2 download block is resolved in this supplement; both approved encoders are available.')
            decision.write_text(text)
            (out/'results/RESOURCE_REPORT.md').write_text('# Combined resource audit\n\nOriginal six-trajectory runtime is preserved separately. Supplemental B extraction and fixed classifiers:\n\n'+json.dumps(resource,indent=2)+'\n')
            result['dino_download_recovery']='COMPLETE';write_json(out/'FINAL_STATUS.json',result)
        else:result={'status':'P1_COMPLETE_P2_BLOCKED','original_status':original_status,'test_predictions':0}
        write_json(out/'STATUS.json',result)
    except Exception as e:
        write_json(out/'STATUS.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc(),'test_predictions':0})
        raise
