"""Read sealed scores after full acceptance; never run or refit a model."""
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from data import read_rows

def main(cfg):
    start=time.time();root=Path(cfg['output']);receipt=root.parent
    assert json.loads((root/'VERIFICATION.json').read_text())['state']=='PASS'
    assert json.loads((receipt/'verify_terminal.private.json').read_text())['returncode']==0
    assert (receipt/'guardian_terminal.private.json').exists(), 'Wait for automatic group retirement'
    target=root/'SAVED_SCORE_DIAGNOSTICS.json';assert not target.exists(), 'Do not repeat the diagnostic'
    rows=read_rows(cfg,'val');labels=np.array([r['label'] for r in rows]);groups=np.array([r['group'] for r in rows]);base=cfg['order'][:10];arms=cfg['arms'];results=json.loads((root/'RESULTS.json').read_text());base_labels=labels[np.isin(labels,base)];assert len(base_labels)==150
    stages=[];initial={};last={};score_rows=0
    for task in range(6):
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];mask=np.isin(labels,seen);yy=labels[mask];base_mask=np.isin(yy,base);assert np.array_equal(yy[base_mask],base_labels)
        arrival=cfg['order'][sum(cfg['task_sizes'][:task]):sum(cfg['task_sizes'][:task+1])];aa=np.isin(yy,arrival);record=dict(task=task,arrival_classes=arrival,arrival_rows=int(aa.sum()),arms={});predictions={}
        for arm in arms:
            scores=torch.load(root/f'scores_{task}_{arm}.private.pt',map_location='cpu',weights_only=False).numpy();assert scores.shape==(len(yy),len(seen));score_rows+=len(scores)
            pred=np.array(seen)[scores.argmax(1)];predictions[arm]=pred;correct=pred==yy
            expected=results['stages'][task]['metrics'][arm];assert int(correct.sum())==expected['total_correct']
            all_correct=int(correct[base_mask].sum());old_pred=np.array(base)[scores[base_mask,:10].argmax(1)];old_correct=int((old_pred==base_labels).sum())
            if task==0:initial[arm]=all_correct
            internal_change=initial[arm]-old_correct;competition_loss=old_correct-all_correct;assert competition_loss>=0 and initial[arm]-all_correct==internal_change+competition_loss
            record['arms'][arm]=dict(base_stage_correct=initial[arm],base_rows=150,base_old_candidates_correct=old_correct,base_all_seen_correct=all_correct,base_internal_head_change=internal_change,added_candidate_competition_loss=competition_loss,current_arrival_correct=int(correct[aa].sum()),perclass_correct={str(c):int(correct[yy==c].sum()) for c in seen})
            if task==5:last[arm]=pred
        record['prediction_disagreement_vs_native']={a:int((predictions[a]!=predictions[arms[0]]).sum()) for a in arms[1:]};record['CE_RC_disagreement']=int((predictions[arms[1]]!=predictions[arms[2]]).sum());stages.append(record)
    paired={};native=last[arms[0]]==labels
    for arm in arms[1:]:
        correct=last[arm]==labels;rescued=correct&~native;harmed=native&~correct;rescue_patients=set(groups[rescued]);harm_patients=set(groups[harmed]);paired[arm]=dict(rescued_images=int(rescued.sum()),harmed_images=int(harmed.sum()),net_correct_gain=int(correct.sum()-native.sum()),rescue_patient_count=len(rescue_patients),harm_patient_count=len(harm_patients),patients_with_both=len(rescue_patients&harm_patients),perclass_rescued={str(c):int(rescued[labels==c].sum()) for c in cfg['order']},perclass_harmed={str(c):int(harmed[labels==c].sum()) for c in cfg['order']});assert paired[arm]['net_correct_gain']==paired[arm]['rescued_images']-paired[arm]['harmed_images']
    curves=json.loads((root/'TRAINING_CURVES.json').read_text());learning={};init=torch.load(root/'initial_prompts.private.pt',map_location='cpu',weights_only=False)
    for arm in arms[1:]:
        prompt=torch.load(root/f'prompts_{arm}.private.pt',map_location='cpu',weights_only=False);sampled=[r for r in curves if r['arm']==arm];learning[arm]=dict(prompt_maxdelta=float((prompt-init).abs().max()),prompt_l2delta=float((prompt-init).double().norm()),first_logged_batch=sampled[0],last_logged_batch=sampled[-1],logged_batches=len(sampled))
    out=dict(scope='One post-acceptance read-only aggregation of saved18score tables. Old-only candidate restriction is diagnostic, not deployable performance. Signed internal changes are not a unique causal attribution.',stages=stages,paired_final_vs_native=paired,existing_training_evidence=learning,temporary_CE_head_comparison='UNAVAILABLE: discarded by frozen protocol; no substitute trained',gradient_decomposition='UNAVAILABLE: not persisted; coefficient alone does not measure its effect',saved_score_rows_read=score_rows,new_model_forwards=0,new_encodes=0,new_optimization=0,internal_seconds=time.time()-start,process_RSS_highwater_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(target)
if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
