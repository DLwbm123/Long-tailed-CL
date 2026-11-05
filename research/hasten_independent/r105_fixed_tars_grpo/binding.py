"""Read-only R104 FULL assets; no selection from historical VAL outcomes."""
import json,csv
from pathlib import Path
import torch
from common import write
from objectives import tailness

def paths(c):
    p=Path(c['frozen_output']);return [p/'strong_heads/head0.private.pt',p/'s0/FULL/fusion1_step512.private.pt',p/'s0/FULL/fusion2_step512.private.pt',p/'s0/FULL/table.private.pt',p/'s0/FULL/features.private.pt']
def snapshot(c):return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths(c)}
def unchanged(c):assert snapshot(c)==json.loads((Path(c['output'])/'FROZEN_STAT.private.json').read_text()),'FROZEN_SOURCE_CHANGED'
def bind(c):
    r=Path(c['output']);p=Path(c['frozen_output']);assert json.loads((p.parent/'CONTROL_EXIT.json').read_text())['exit_code']==0 and json.loads((p.parent/'VERIFICATION.json').read_text())['passed']
    rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()));assert len(rows)==6821;y=torch.tensor([c['order'].index(int(x['original_label'])) for x in rows]);assert torch.bincount(y,minlength=23).tolist()==c['train_counts']
    head=torch.load(p/'strong_heads/head0.private.pt',map_location='cpu',weights_only=False);assert head['seen']==23 and head['fit_ids']==list(range(6821)) and head['train_counts'].tolist()==c['train_counts']
    pack=torch.load(p/'s0/FULL/table.private.pt',map_location='cpu',weights_only=False);assert pack['ids']==list(range(6821)) and pack['source_bank']=='FULL';assert pack['features'].shape==(6821,3,19) and pack['scores'].shape==(6821,3,23)
    assert torch.equal(pack['correct'],pack['scores'].argmax(-1)==y[:,None]);table={k:pack[k].cuda().detach() for k in ['features','scores','correct']};nc=torch.bincount(y,minlength=23);correct=torch.bincount(y[pack['correct'][:,0]],minlength=23);weights=tailness(nc,correct);assert torch.isfinite(weights).all()
    write(r/'FROZEN_STAT.private.json',snapshot(c));write(r/'TARS_WEIGHTS.json',{'gamma':.75,'median':'midpoint','source':'fixed FULL teacher TRAIN recognition error, no VAL','counts':nc.tolist(),'teacher_correct':correct.tolist(),'weights':weights.tolist(),'computed_once_before_policy_fit':True,'query_sampling':'natural TRAIN rows, shared across all arms'})
    with (r/'TRAIN_CLASS_WEIGHTS.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['class','original_label','legacy_index','TRAIN_count','teacher_TRAIN_correct','teacher_TRAIN_error','weight'])
        for k in range(23):w.writerow([k,c['order'][k],c['legacy_indices'][k],int(nc[k]),int(correct[k]),float(1-correct[k]/nc[k]),float(weights[k])])
    write(r/'FROZEN_BINDING.json',{'passed':True,'source_experiment':'R104','source_final_commit':'27f6b25edd0a12fb54671dbbaff17781fcc3d950','same_FULL_teacher_prototypes_and_both_fusion_models_for_feedback_and_deployment':True,'source_train_rows':6821,'TRAIN_in_sample_to_upstream_fit':True,'whole_chain_crossfit':False,'upstream_refit_new_teacher_fusion_optimizer':0,'no_historical_VAL_arrays_loaded':True,'immutable_source_checked_by_size_mtime_not_checksum':True})
    return rows,y,head,table,weights.cuda()
