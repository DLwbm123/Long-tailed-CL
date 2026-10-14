"""Anonymous report fixture; no images, checkpoints or inference."""
import copy
import json
import os
from pathlib import Path
import tempfile
import run_paired_transport_cycle as cycle
from check_lca_production import rejected


def check(config):
    from summarize_paired_transport import report,NEW,MODULES
    from run_paired_transport_worker import paired
    import numpy as np
    yy=np.array([0,0,1,1]);off=np.array([0,1,1,0]);on=np.array([1,0,1,0]);ss=np.ones((4,2))
    pp=paired(yy,[0,1],on,off,ss,ss,1)
    assert pp['classes'][0]['rescued']==1 and pp['classes'][0]['damaged']==1
    assert all(r['correct_on']-r['correct_control']==r['rescued']-r['damaged'] for r in pp['classes'])
    snapshot=dict(program=dict(status='COMPLETE',active={},failures=[],elapsed_seconds=0.,costs=[dict(returncode=0,gpu=i%2,residence_seconds=0.,operation=op,job=prefix+ds+'_'+arm) for i,(ds,arm,prefix,op) in enumerate((ds,arm,prefix,op) for ds in cycle.ORDERS for arm in NEW for prefix,op in [('pre_','train'),('','train'),('','evaluate')])]),
        source_commit='synthetic',lock=json.loads(Path(os.environ['Q215_CPU_PROTOCOL']).read_text()),runs={},preflights={},
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
                test_accessed=False,source_commit='synthetic',pmt_version=cycle.VERSION,module=arm,checkpoint_head_rebuild_verified=True,paired_control_inference_recall_verified=True)
            diagnostics=[dict(epoch=ep,max_batch_solve_residual=0.,paired_transport=dict(fit_only=True,detached=True),boundary_paired_transport=dict(current_train_including_meta=True)) for t in cycle.SIZES[ds] for ep in (1,2)]
            snapshot['runs'][ds+'_'+arm]=dict(status=dict(status='COMPLETE',steps=sum(cycle.STEPS[ds]),stages=len(cycle.SIZES[ds]),policy_updates=0,max_solve_residual=0.,synthetic_highest_bank=dict(synthetic_only=True,dimension=1536)),metrics=metrics,scalar_diagnostics=diagnostics)
            if arm in NEW:snapshot['preflights'][ds+'_'+arm]=dict(receipt_status='PASS',receipt_complete=True,optimizer_updates=0,policy_updates=0,parameters_unchanged=True,test_accessed=False,gradient_norm=1.,max_solve_residual=0.,synthetic_highest_bank=dict(synthetic_only=True,dimension=1536))
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
        report(snapshot,Path(tmp)/'good')
        bad=copy.deepcopy(snapshot);bad['runs']['ISIC_pmt']['metrics']['stages'][0]['per_class_n']['4']+=1
        rejected(lambda:report(bad,Path(tmp)/'bad'))
    return dict(all_methods_and_contrasts=True,paired_counts_identity=True,bad_denominator_refused=True)
