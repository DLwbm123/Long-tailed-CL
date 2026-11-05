"""Complete post-run metrics/denominator/rescue audit from sealed predictions only."""
import csv,json,time
from pathlib import Path
import torch
from common import write

def run(root):
    root=Path(root);cpu=time.process_time();wall=time.time();readouts=0;teacher_diffs={};group_diffs={}
    for stream in ['a','b']:
        r=root/stream;c=json.loads((r/'RUN_CONFIG.private.json').read_text());record=json.loads((r/'STAGE_METRICS.json').read_text());summary=json.loads((r/'RESULT_SUMMARY.json').read_text())
        groups=list(csv.DictReader((r/'GROUP_STAGE.csv').open()));harms=list(csv.DictReader((r/'RESCUE_HARM.csv').open()));classes=list(csv.DictReader((r/'CLASS_STAGE.csv').open()));oracles=list(csv.DictReader((r/'ORACLE_DIAGNOSTICS.csv').open()));teacher_diffs[stream]=[];group_diffs[stream]=[]
        for s in range(5):
            p=torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False);y=p['labels'];C=sum(map(len,c['stages'][:s+1]));den=torch.bincount(y,minlength=C);assert (den>0).all();old=torch.tensor([b<s for b,cl in enumerate(c['stages'][:s+1]) for _ in cl]);tail=torch.tensor(c['train_counts'][:C])<250;T=p['predictions']['T']==y
            teacher=torch.load(Path(c['reuse_root'])/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False);assert torch.equal(y,teacher['labels']);teacher_diffs[stream].append(int((p['predictions']['T']!=teacher['predictions']['T']).sum()));group_diffs[stream].append(int((p['predictions']['GRPO_CF']!=p['predictions']['GROUP_EXACT_CF']).sum()))
            oracle=p['candidate_correct'].any(-1)
            for cl in range(C):
                o=next(v for v in oracles if int(v['stage'])==s and int(v['class'])==cl);mask=y==cl;assert int(o['teacher_wrong_candidate_rescue'])==int((mask&~T&oracle).sum()) and int(o['count'])==int(den[cl])
            for mode,pred in p['predictions'].items():
                good=pred==y;cc=torch.bincount(y[good],minlength=C);rec=cc.double()/den;raw={'micro':float(good.double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None,'old_tail':float(rec[old&tail].mean()*100) if (old&tail).any() else None}
                for key,v in raw.items():
                    target=record[s]['modes'][mode][key];assert target is None if v is None else abs(target-v)<1e-10
                for cl in range(C):
                    mask=y==cl;h=next(v for v in harms if int(v['stage'])==s and v['mode']==mode and int(v['class'])==cl);z=next(v for v in classes if int(v['stage'])==s and v['mode']==mode and int(v['class'])==cl)
                    assert int(h['harm'])==int((mask&T&~good).sum()) and int(h['rescue'])==int((mask&~T&good).sum())
                    assert int(z['correct'])==int(cc[cl]) and int(z['count'])==int(den[cl]) and int(z['original_label'])==c['order'][cl]
                for age,am in [('old',old),('new',~old),('all',torch.ones(C,dtype=torch.bool))]:
                    for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones(C,dtype=torch.bool))]:
                        mask=am&fm;g=next(v for v in groups if int(v['stage'])==s and v['mode']==mode and v['group']==age+'_'+freq);assert int(g['class_count'])==int(mask.sum())
                        if mask.any():assert abs(float(g['macro'])-float(rec[mask].mean()*100))<1e-10
                        else:assert g['macro']==''
                readouts+=1
        for mode in summary['mean']:
            for key in ['micro','macro','tail','old_tail']:
                values=[z['modes'][mode][key] for z in record if z['modes'][mode][key] is not None];assert len(values)==summary['mean_valid_stages'][mode][key]
                assert abs(sum(values)/len(values)-summary['mean'][mode][key])<1e-10
    assert readouts==90
    result={'passed':True,'readouts_recomputed':readouts,'micro_macro_tail_oldtail_groups_class_counts_rescue_harm_oracle_verified':True,'all_empty_groups_and_mean_denominators_checked':True,'R102_teacher_prediction_differences':teacher_diffs,'GRPO_GROUP_EXACT_prediction_differences':group_diffs,'strict_R102_teacher_VAL_identity':False,'baseline_readout_limitation_reported':True,'new_encoder_optimizer_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall}
    write(root/'SAVED_VERIFICATION.json',result);return result
