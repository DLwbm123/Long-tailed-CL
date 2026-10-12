"""Pure scalar report fixture: never queries current workers, images or checkpoints."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import time
from summarize_paper_medical import report,NEW,PROJECT,EXTERNAL


def check(round_dir):
    started=time.monotonic();r=Path(round_dir)
    lock=json.loads((r/'PROTOCOL_LOCK.json').read_text());spec=json.loads((r/'REPORT_COMPARISON_SPEC.json').read_text())
    published=json.loads(Path('docs/air_apart_isic_hk1_20261012/results/RESULTS.json').read_text())
    sources={'base':'module_additions1_20261009','local_transport':'module_local2_20261010','residual_drift':'module_drift10_20261011','local_transport_drift':'module_drift10_20261011','AIR':'air_apart_isic_hk1_20261012','APART_full':'air_apart_isic_hk1_20261012'}
    costs={m:json.loads(Path('docs',name,'results/RESULTS.json').read_text())['costs'] for m,name in sources.items()}
    for m in NEW:costs[m]=[]
    fixture=dict(program=dict(status='COMPLETE',active={},failures=[],costs=[dict(gpu=i%2,returncode=0,residence_seconds=1.) for i in range(12)],elapsed_seconds=1.),
        lock=lock,resource_comparison=spec,cost_sources=costs,runs=copy.deepcopy(published['runs']),preflights={},closed_round_cost_ledger={},round_name='synthetic_report_fixture',source_commit='synthetic_report_fixture',cpu_checks={})
    for ds,d in lock['data'].items():
        for method in NEW:
            v=copy.deepcopy(fixture['runs'][ds+'_base']);v['status'].update(status='COMPLETE',steps=d[method]['total_updates'],stages=len(d['task_sizes']),policy_updates=0)
            v['metrics'].update(module=method,nonofficial_paper_reimplementation=True,evaluation_split='development_validation')
            v['scalar_diagnostics']=[dict(loss_mean=1.) for _ in range(lock[method]['epochs']*len(d['task_sizes']))]
            fixture['runs'][ds+'_'+method]=v
            fixture['preflights'][ds+'_'+method]=dict(status='PASS',optimizer_updates=0,test_accessed=False,parameters_unchanged=True,gradient_norm=1.)
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'complete';a=report(fixture,out)
        contrasts=json.loads((out/'CONTRIBUTIONS.json').read_text());assert len(contrasts)==40
        project_contrasts=[c for c in contrasts if c['control'] in PROJECT];assert len(project_contrasts)==32
        assert {(c['dataset'],c['method'],c['control']) for c in project_contrasts}=={(ds,m,c) for ds in ['ISIC','HK'] for m in EXTERNAL for c in PROJECT}
        resources=json.loads((out/'RESOURCE_COMPARISON.json').read_text())['rows'];assert len(resources)==16
        for row in resources:
            ds=row['dataset'];method=row['method'];d=lock['data'][ds]
            if method=='AIR':assert row['optimizer_steps']==0 and row['analytic_batches'] in [586,216] and row['old_image_bank_unique']==0
            if method=='Hao_paper':assert row['old_image_bank_unique']==d[method]['final_unique_bank'] and row['pretraining'].startswith('ResNet18')
            if method in PROJECT:assert row['exact_optimizer_image_exposures'] is None and row['checkpoint_bytes'] is None
        comparison=json.loads((out/'COMPETITIVENESS.json').read_text());assert len(comparison['primary_contrasts'])==8
        assert comparison['primary_method']=='local_transport'
        for v in comparison['primary_contrasts']:
            a=published['runs'][v['dataset']+'_local_transport']['metrics'];b=fixture['runs'][v['dataset']+'_'+v['external']]['metrics']
            assert abs(v['delta_pp']['final_balanced_accuracy']-100*(a['final_balanced_accuracy']-b['final_balanced_accuracy']))<1e-10
        assert any(v['favourable_metrics'] for v in comparison['primary_contrasts']) and any(v['unfavourable_metrics'] for v in comparison['primary_contrasts'])
        with (out/'FINAL_CLASS_DIFFERENCES.csv').open() as f:assert len(list(csv.DictReader(f)))==496
        with (out/'STAGE_CLASS_RECALLS.csv').open() as f:assert len(list(csv.DictReader(f)))==1008
        bad=copy.deepcopy(fixture);bad['runs']['ISIC_DAT_paper']['metrics']['stages'][-1]['per_class_n']['0']+=1
        try:report(bad,Path(tmp)/'bad_n')
        except ValueError:pass
        else:raise AssertionError('Changed denominator accepted')
        bad=copy.deepcopy(fixture);bad['resource_comparison']['primary_project_method']='local_transport_drift'
        try:report(bad,Path(tmp)/'best_row')
        except ValueError:pass
        else:raise AssertionError('Posthoc primary switch accepted')
    for name in ['summarize_paper_medical.py','collect_paper_medical.py','check_paper_medical.py']:ast.parse(Path('tools',name).read_text())
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-started,report_only=True,
        medical_images_opened=0,current_worker_or_validation_access=False,gpu_processes=0,formal_updates=0,test_accessed=False,
        synthetic_results_never_saved_as_scientific_results=True,main_rows=16,external_project_contrasts=32,total_signed_contrasts=40,
        final_class_signed_rows=496,stage_class_recall_rows=1008,missing_resources_preserved_not_zero=True,
        AIR_analytic_cost_not_zero_claim=True,resource_pretraining_steps_and_replay_counts=True,
        all_positive_negative_retained=True,fixed_primary_and_denominator_rejection=True,
        frozen_training_source_unchanged=spec['training_source_unchanged'])


if __name__=='__main__':
    result=check('docs/dat_hao_isic_hk2_20261012')
    Path('docs/dat_hao_isic_hk2_20261012/CPU_REPORT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
