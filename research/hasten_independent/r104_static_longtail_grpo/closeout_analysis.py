"""Saved aggregate tables only: no model, tensors, images or evaluation forward."""
import csv,json,time,math
from pathlib import Path

def run(r):
    r=Path(r);start=time.perf_counter();cpu=time.process_time();s=json.loads((r/'RESULT_SUMMARY.json').read_text());rows=list(csv.DictReader((r/'CLASS_STAGE.csv').open()));mapping={int(x['classifier_index']):x for x in csv.DictReader((r/'CLASS_MAPPING.csv').open())};out={};worst=0
    for mode,m in s['final'].items():
        rr=[x for x in rows if x['mode']==mode];assert len(rr)==23 and sum(int(x['count']) for x in rr)==1698
        for x in rr:assert 0<=int(x['correct'])<=int(x['count'])
        rec=[int(x['correct'])/int(x['count']) for x in rr];tt=[j for j,x in enumerate(rr) if int(mapping[int(x['class'])]['TRAIN_count'])<250]
        rebuilt={'micro':100*sum(int(x['correct']) for x in rr)/1698,'macro':100*sum(rec)/23,'tail':100*sum(rec[j] for j in tt)/len(tt)}
        worst=max(worst,max(abs(rebuilt[k]-m[k]) for k in rebuilt));assert worst<1e-10
        tiny=[x for x in rr if x['tiny']=='True'];out[mode]={'tiny_correct_counts':{x['legacy_index']:[int(x['correct']),int(x['count'])] for x in tiny},'non_tiny_macro':100*sum(int(x['correct'])/int(x['count']) for x in rr if x['tiny']!='True')/20}
    t=s['final']['T'];pair={mode:{k:x[k]-t[k] for k in ['micro','macro','tail']} for mode,x in s['final'].items()}
    for mode in ['CENTER','MULTI','UNIFORM','GRPO_IN']:
        tiny_gain=sum(out[mode]['tiny_correct_counts'][k][0]/out[mode]['tiny_correct_counts'][k][1]-out['T']['tiny_correct_counts'][k][0]/out['T']['tiny_correct_counts'][k][1] for k in out['T']['tiny_correct_counts'])*100/23
        out[mode]['tiny_macro_contribution_vs_T_pp']=tiny_gain;out[mode]['tiny_share_of_net_macro_gain_vs_T']=tiny_gain/pair[mode]['macro'] if pair[mode]['macro'] else None
        out[mode]['non_tiny_macro_vs_T_pp']=out[mode]['non_tiny_macro']-out['T']['non_tiny_macro']
    aux={k:s['final']['GRPO_IN'][k]-s['final']['CENTER'][k] for k in ['micro','macro','tail']}
    result={'scope':'saved public aggregate CSV/JSON arithmetic only','passed':True,'max_reconstruction_error_pp':worst,'class_count':23,'VAL_denominator':1698,'tail_class_count':len(tt),'per_mode':out,'vs_T_pp':pair,'GRPO_IN_minus_CENTER_pp':aux,'primary_GRPO_CF_has_no_macro_tail_gain_vs_teacher':pair['GRPO_CF']['macro']<0 and pair['GRPO_CF']['tail']<0,'matched_IN_SUP_RAW_GROUP_controls_present':False,'new_encoder_optimizer_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    (r/'SAVED_CLOSEOUT_ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':run(Path(__file__).resolve().parent)
