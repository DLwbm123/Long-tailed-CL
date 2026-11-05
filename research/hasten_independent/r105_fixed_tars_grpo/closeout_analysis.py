"""Aggregate-only reconstruction and tiny sensitivity; never a new model evaluation."""
import csv,json,time
from pathlib import Path

def run(r):
    r=Path(r);start=time.perf_counter();cpu=time.process_time();s=json.loads((r/'RESULT_SUMMARY.json').read_text())['final'];rows=list(csv.DictReader((r/'CLASS_STAGE.csv').open()));mapping={int(x['classifier_index']):int(x['TRAIN_count']) for x in csv.DictReader((r/'CLASS_MAPPING.csv').open())};stats={};err=0
    for mode,m in s.items():
        rr=[x for x in rows if x['mode']==mode];assert len(rr)==23 and sum(int(x['count']) for x in rr)==1698
        assert all(0<=int(x['correct'])<=int(x['count']) for x in rr);tail=[x for x in rr if mapping[int(x['class'])]<250];non=[x for x in rr if x['tiny']!='True']
        values={'micro':100*sum(int(x['correct']) for x in rr)/1698,'macro':100*sum(int(x['correct'])/int(x['count']) for x in rr)/23,'tail':100*sum(int(x['correct'])/int(x['count']) for x in tail)/len(tail)}
        err=max(err,max(abs(values[k]-m[k]) for k in values));assert err<1e-10
        stats[mode]={'tiny':{x['legacy_index']:[int(x['correct']),int(x['count'])] for x in rr if x['tiny']=='True'},'non_tiny_macro':100*sum(int(x['correct'])/int(x['count']) for x in non)/len(non)}
    effects={}
    for reference in ['T','GRPO','CENTER','TARS_RAW_EXACT','TARS_GROUP_EXACT','GROUP_EXACT','TARS_SUP']:
        tiny=sum(stats['TARS_GRPO']['tiny'][k][0]/stats['TARS_GRPO']['tiny'][k][1]-stats[reference]['tiny'][k][0]/stats[reference]['tiny'][k][1] for k in stats[reference]['tiny'])*100/23
        effects[reference]={'delta':{k:s['TARS_GRPO'][k]-s[reference][k] for k in ['micro','macro','tail']},'tiny_macro_contribution_pp':tiny,'non_tiny_macro_delta_pp':stats['TARS_GRPO']['non_tiny_macro']-stats[reference]['non_tiny_macro']}
    out={'passed':True,'scope':'saved public class counts only; no forward or selection','max_reconstruction_error_pp':err,'static_class_count':23,'tail_classes':11,'VAL_denominator':1698,'per_mode':stats,'TARS_GRPO_minus_reference':effects,'formal_query_update_exposures':4096*128,'actual_GRPO_sampled_groups':32768,'GRPO_group_update_exposures':32768*4,'actual_categorical_draws':262144,'GRPO_action_update_exposures':262144*4,'new_encoder_optimizer_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    out.update(derived_cache_parity_TRAIN_candidate_scoring_rows=6821*3,derived_formal_TRAIN_router_forward_query_rows=4096*128+8*128*128+8*4,derived_engineering_router_forward_query_rows=64*128+8*2*128,derived_VAL_router_forward_query_rows=8*1698,derived_forward_counts_are_code_path_accounting_not_independent_cases=True)
    (r/'SAVED_CLOSEOUT_ANALYSIS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');return out
if __name__=='__main__':run(Path(__file__).resolve().parent)
