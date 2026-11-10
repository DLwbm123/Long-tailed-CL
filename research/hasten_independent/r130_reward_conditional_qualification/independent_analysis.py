"""Exact rational arithmetic on public class counts; no torch or model import."""
import csv,json,time
from pathlib import Path
from fractions import Fraction as F
P=Path(__file__).parent

def run():
 start=time.time();cpu=time.process_time();arms=['RAPO','ACC_CTAN','ACC_FULL_KL','ACC_OLD_COND','CE_OLD_COND'];names={};counts={}
 for r in csv.DictReader((P.parent/'r125_seen_label_interface/CLASS_RESULTS.csv').open()):
  c=int(r['class']);n=int(r['TRAIN_count']);assert c not in counts or counts[c]==n;counts[c]=n;names[c]=r['name']
 by={}
 for r in csv.DictReader((P/'CLASS_RESULTS.csv').open()):
  a=r['arm'];c=int(r['class_id']);good=int(r['correct']);den=int(r['denominator']);assert 0<=good<=den and den>0;assert (a,c) not in by;by[a,c]=(good,den)
 assert len(by)==60
 means={};evals={r['arm']:r for r in json.loads((P/'EVALUATIONS.json').read_text())}
 for a in ['PREFIX_CURRENT_MENU']+arms:
  def mean(ids):return sum((F(*by[a,c]) for c in ids),F(0))/len(ids)
  means[a]={'macro':mean(range(10)),'old_macro':mean(range(5)),'new_macro':mean(range(5,10)),'micro':F(sum(by[a,c][0] for c in range(10)),sum(by[a,c][1] for c in range(10)))}
  for k,ids in [('head',[c for c in range(10) if counts[c]>=250]),('tail',[c for c in range(10) if counts[c]<250]),('tiny',[c for c in range(10) if counts[c]<25])]:means[a][k]=mean(ids)
  assert sum(by[a,c][1] for c in range(10))==964
  for k in ['macro','micro','head','tail','tiny']:assert abs(float(means[a][k])-evals[a][k])<1e-12
 d=means['ACC_OLD_COND'];c=means['ACC_FULL_KL'];b=means['ACC_CTAN'];e=means['CE_OLD_COND']
 gates={'D_minus_C_macro_ge_1pp':d['macro']-c['macro']>=F(1,100),'D_minus_C_tail_ge_2pp':d['tail']-c['tail']>=F(2,100),'D_old_macro_ge_C':d['old_macro']>=c['old_macro'],'D_new_macro_ge_B_minus1pp':d['new_macro']-b['new_macro']>=F(-1,100),'D_macro_over_matched_CE_positive':d['macro']>e['macro'],'D_tail_over_matched_CE_positive':d['tail']>e['tail']}
 final=json.loads((P/'FINAL_RESULTS.json').read_text());assert gates==final['gates'];curves=json.loads((P/'TRAINING_CURVES.json').read_text());assert len(curves)==160
 integrity=all(sum(x['arm']==a and x['nonzero_update'] and x['parameter_L2']>0 for x in curves)==32 for a in arms);assert integrity
 group=[]
 for a in arms:
  for c0 in range(5,10):
   sums={k:0 for k in ['queries','all_wrong','mixed','all_correct','correct_responses']}
   for x in curves:
    if x['arm']!=a:continue
    g=x['class_groups'].get(str(c0),{})
    for k in sums:sums[k]+=g.get(k,0)
   group.append({'arm':a,'class_id':c0,'name':names[c0],'TRAIN_count':counts[c0],**sums})
  assert sum(x['queries'] for x in group if x['arm']==a)==256
 with (P/'GROUP_DIAGNOSTICS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(group[0]));w.writeheader();w.writerows(group)
 pairs=[('ACC_OLD_COND','RAPO'),('ACC_OLD_COND','ACC_CTAN'),('ACC_OLD_COND','ACC_FULL_KL'),('ACC_OLD_COND','CE_OLD_COND'),('RAPO','ACC_CTAN'),('ACC_FULL_KL','ACC_CTAN')];contributions=[];diff={}
 for a,b0 in pairs:
  diff[a+'__minus__'+b0]={k:float(100*(means[a][k]-means[b0][k])) for k in means[a]}
  for c0 in range(10):
   assert by[a,c0][1]==by[b0,c0][1];den=by[a,c0][1];delta=F(by[a,c0][0]-by[b0,c0][0],den)
   contributions.append({'arm':a,'control':b0,'class_id':c0,'name':names[c0],'role':'old' if c0<5 else 'new','TRAIN_count':counts[c0],'correct':by[a,c0][0],'control_correct':by[b0,c0][0],'denominator':den,'recall_delta_pp':float(100*delta),'seen_macro_contribution_pp':float(10*delta)})
 with (P/'PAIRED_CLASS_CONTRIBUTIONS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(contributions[0]));w.writeheader();w.writerows(contributions)
 result={'passed':True,'gates_from_exact_integer_fractions':gates,'performance_gates_passed':sum(gates.values()),'performance_gate_count':6,'integrity_gate_passed':integrity,'all_seven_qualification_conditions_passed':all(gates.values()) and integrity,'metrics_percent':{a:{k:float(100*v) for k,v in m.items()} for a,m in means.items()},'metrics_exact_fractions':{a:{k:str(v) for k,v in m.items()} for a,m in means.items()},'paired_delta_pp':diff,'class_rows':len(by),'paired_class_rows':len(contributions),'new_model_calls':0,'CPU_body_seconds':time.process_time()-cpu,'wall_body_seconds':time.time()-start}
 (P/'INDEPENDENT_INTEGER_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'performance_passed':sum(gates.values()),'integrity_passed':integrity,'D_delta_pp':{k:v for k,v in diff.items() if k.startswith('ACC_OLD_COND')},'current_tiny_query_opportunities':[r for r in group if counts[r['class_id']]<25]},ensure_ascii=False))
if __name__=='__main__':run()
