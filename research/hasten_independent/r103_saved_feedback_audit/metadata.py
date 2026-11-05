"""Inventory archived lineage and shift aggregates, without model reconstruction."""
import json,csv,time,argparse
from pathlib import Path

def run(source,out):
    start=time.time();cpu=time.process_time();source=Path(source);out=Path(out)
    with (out/'ASSET_AVAILABILITY.csv').open() as f:assets=list(csv.DictReader(f))
    provenance=[]
    for stream in ['a','b']:
        static=json.loads((source/stream/'FOLD_EXCLUSION_AUDIT.json').read_text());runtime=json.loads((source/stream/'LINEAGE_RUNTIME_AUDIT.json').read_text());full=json.loads((source/stream/'FULL_TEACHER_BINDING.json').read_text())
        assert static['passed'] and runtime['passed'] and full['passed'] and static['identity_component_intersection']==0
        for rec in runtime['records']:
            assert rec['history_fit_components_intersect_feedback'] is False and rec['optimizer_fit_rows_only']
            provenance.append(dict(stream=stream,stage=rec['stage'],bank=rec['bank'],fit_rows=rec['fit_rows'],feedback_rows=rec['feedback_rows'],historical_fit_feedback_component_intersection=0,precision='archived_runtime_and_binding_audit_not_training_reexecution'))
        for stage in range(5):
            for field,precision,limit in [('OOF_source_bank','endpoint_only','TRAIN_OOF IDs mapped to opposite outer bank; current components independently disjoint'),('historical_bank_supervision_lineage','aggregate_only','archived runtime/static lineage assertions checked; no training replay or model load'),('step0_256_512_outputs','unavailable','checkpoints exist per original verification but per-row router outputs were not saved; no forward allowed'),('OOF_FULL_router_probability_shift','aggregate_only','only original aggregate statistics can be cited, not independently reconstructed')]:
                assets.append(dict(stream=stream,stage=stage,role='TRAIN_OOF_and_FULL',field=field,precision=precision,available=precision!='unavailable',limitation=limit))
    for v in assets:
        stage=v['stage'];role=v['role'];tab=f's{stage}/OOF.private.pt' if role=='TRAIN_OOF' else f's{stage}/IN.private.pt' if role=='TRAIN_FULL' else f'VAL_predictions_s{stage}.private.pt'
        v['source_science_commit']='ec5502cf886e7e453e98609f9fcd6dde0df6d5fd'
        v['source_artifact']={'stable_query_identity':tab+' ids + frozen manifest','candidate_logits':tab+' scores' if role!='VAL_FULL' else 'not_saved','candidate_correctness':tab+' correct/candidate_correct','learned_router_predictions_actions':tab+' predictions/actions' if role=='VAL_FULL' else 'TRAIN_FEEDBACK_DIAGNOSTICS.json aggregates only','learned_router_probabilities':'not_saved; checkpoint weights are not outputs','rollout_query_plan':f's{stage}/QUERY_PLAN.private.pt' if role!='VAL_FULL' else 'not_applicable','rollout_old_probabilities_actions':'not_saved','identity_component':'frozen manifests; historical saved VAL component metadata','observed_rollout_zero_group_totals':f's{stage}/QUERY_PLAN.private.pt + GRPO_*_TRAIN_DIAGNOSTICS.json','OOF_source_bank':f's{stage}/OOF.private.pt ids + stable S/B/F source binding','historical_bank_supervision_lineage':'FOLD_EXCLUSION_AUDIT.json + LINEAGE_RUNTIME_AUDIT.json','step0_256_512_outputs':'only checkpoints saved, not per-row outputs','OOF_FULL_router_probability_shift':'FOLD_TO_FULL_SHIFT.csv original aggregate'}[v['field']]
    with (out/'ASSET_AVAILABILITY.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(assets[0]),lineterminator='\n');w.writeheader();w.writerows(assets)
    with (out/'SOURCE_BANK_PROVENANCE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(provenance[0]),lineterminator='\n');w.writeheader();w.writerows(provenance)
    with (source/'FOLD_TO_FULL_SHIFT.csv').open() as f:shift=list(csv.DictReader(f))
    for row in shift:row['verification_level']='source_aggregate_only_no_saved_row_probabilities'
    with (out/'ROUTER_SHIFT_RECORDED_AGGREGATE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(shift[0]),lineterminator='\n');w.writeheader();w.writerows(shift)
    # This is the pre-existing readout caveat, not a new replacement or score.
    (out/'SOURCE_BASELINE_READOUT_LIMITATION.json').write_text((source/'BASELINE_READOUT_AUDIT.json').read_text())
    counts=json.loads((out/'ANALYSIS_COUNTS.json').read_text());counts['saved_reads']['historical_VAL_metadata_rows_loaded']=counts['saved_reads']['saved_VAL_query_rows'];counts['historical_predictions_used_as_new_reference']=False;(out/'ANALYSIS_COUNTS.json').write_text(json.dumps(counts,indent=2))
    (out/'METADATA_CHECK.json').write_text(json.dumps({'passed_archived_metadata_checks':True,'lineage_records':len(provenance),'independent_model_history_reexecution':False,'new_optimizer_forward_image_encoder_VAL_scoring_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
