"""Frozen CF_BHV A/B/C gates and all matched positive/negative comparisons."""
import csv,json,torch
from pathlib import Path
from common import write
from evaluate import MODES
PRIMARY='CF_BHV'

def table(c,pack,s):
 n=sum(map(len,c['stages'][:s+1]));y=pack['labels'];den=torch.bincount(y,minlength=n);assert (den>0).all();tail=torch.tensor(c['train_counts'][:n])<250;old=torch.arange(n)<sum(map(len,c['stages'][:s]));out={}
 for a,pred in pack['predictions'].items():
  cc=torch.bincount(y[pred==y],minlength=n);rec=cc.double()/den;groups={}
  for age,mask in [('old',old),('new',~old),('all',torch.ones(n,dtype=torch.bool))]:
   for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones(n,dtype=torch.bool))]:
    keep=mask&fm;groups[age+'_'+freq]=float(rec[keep].mean()*100) if keep.any() else None
  out[a]={'micro':float(cc.double().sum()/den.sum()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100),'counts':den.tolist(),'correct':cc.tolist(),'recall':(rec*100).tolist(),'groups':groups}
 return {'stage':s,'classes':n,'arms':out}

def summarize(c,records,packs):
 out={'primary':PRIMARY,'mean':{},'final':{},'primary_comparisons':{},'tiny_harm':{}}
 for arm in MODES:
  out['mean'][arm]={k:sum(v['arms'][arm][k] for v in records)/6 for k in ['micro','macro','tail']};out['final'][arm]={k:records[-1]['arms'][arm][k] for k in ['micro','macro','tail']}
 def diff(a,b):return {phase+'_delta_pp':{k:out[phase][a][k]-out[phase][b][k] for k in ['micro','macro','tail']} for phase in ['mean','final']}
 for ref in MODES:
  if ref!=PRIMARY:out['primary_comparisons'][ref]=diff(PRIMARY,ref)
 for ref in ['T','SUP0']:
  out['tiny_harm'][ref]={str(cl):sum(int(((p['labels']==cl)&(p['predictions'][ref]==p['labels'])&(p['predictions'][PRIMARY]!=p['labels'])).sum()) for p in packs) for cl in [1,2,12]}
 d=diff(PRIMARY,'T');both=sum(v['arms'][PRIMARY]['macro']>=v['arms']['T']['macro']-1e-8 and v['arms'][PRIMARY]['tail']>=v['arms']['T']['tail']-1e-8 for v in records);A=d['mean_delta_pp']['macro']>=.5 and d['final_delta_pp']['macro']>=.5 and d['mean_delta_pp']['tail']>=-1e-8 and d['final_delta_pp']['tail']>=1 and both>=4 and all(v==0 for rr in out['tiny_harm'].values() for v in rr.values());B={}
 for ref in ['SUP0','SC_BHV']:
  d=diff(PRIMARY,ref);B[ref]=d['mean_delta_pp']['macro']>=.3 and d['final_delta_pp']['macro']>=.3 and d['mean_delta_pp']['tail']>=-1e-8 and d['final_delta_pp']['tail']>=-1e-8
 d=diff(PRIMARY,'CF_ACT');C=d['mean_delta_pp']['macro']>1e-8 and d['final_delta_pp']['macro']>1e-8 and d['mean_delta_pp']['tail']>=-1e-8 and d['final_delta_pp']['tail']>=-1e-8
 out['constraint_replacement']={o:{'BHV_minus_ACT':diff(o+'_BHV',o+'_ACT'),'BHV_minus_NONE':diff(o+'_BHV',o+'_NONE')} for o in ['SC','CF','PATH']};out['CF_SC_constraint_interaction_pp']={phase:{k:(out[phase]['CF_BHV'][k]-out[phase]['SC_BHV'][k])-(out[phase]['CF_ACT'][k]-out[phase]['SC_ACT'][k]) for k in ['micro','macro','tail']} for phase in ['mean','final']};out.update(gate_A=bool(A),gate_B={k:bool(v) for k,v in B.items()},constraint_gate_C=bool(C),stages_macro_tail_nonnegative=both,passed=bool(A and all(B.values())));out['conclusion']='COMPLETE_PRIMARY_DEVELOPMENT_GATE_PASS' if out['passed'] else 'COMPLETE_PRIMARY_MECHANISM_NEGATIVE';return out

def csvout(p,header,rows):
 with p.open('w') as f:w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(rows)

def run(c):
 r=Path(c['output']);packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(6)];records=[table(c,p,s) for s,p in enumerate(packs)];F=[table(c,torch.load(r/f'F_predictions_s{s}.private.pt',map_location='cpu',weights_only=False),s) for s in range(6)];a=summarize(c,records,packs);write(r/'RESULT_SUMMARY.json',a);write(r/'STAGE_METRICS.json',records);write(r/'F_TRANSFER.json',F)
 csvout(r/'STAGE_METRICS.csv',['stage','arm','micro','macro','tail'],[[v['stage'],arm,*[m[k] for k in ['micro','macro','tail']]] for v in records for arm,m in v['arms'].items()]);csvout(r/'CLASS_STAGE.csv',['stage','arm','class','count','correct','recall'],[[v['stage'],arm,i,n,m['correct'][i],m['recall'][i]] for v in records for arm,m in v['arms'].items() for i,n in enumerate(m['counts'])]);csvout(r/'GROUP_STAGE.csv',['stage','arm','group','macro'],[[v['stage'],arm,g,x] for v in records for arm,m in v['arms'].items() for g,x in m['groups'].items()]);contrib=[];harm=[];sensitivity=[]
 for arm in MODES[1:]:
  for ref in ['T','SUP0']:
   influence={}
   for s,(v,p) in enumerate(zip(records,packs)):
    n=v['classes'];nt=sum(x<250 for x in c['train_counts'][:n]);y=p['labels'];delta=(p['predictions'][arm]==y).double()-(p['predictions'][ref]==y).double();den=torch.bincount(y,minlength=n)
    for cl in range(n):
     dr=v['arms'][arm]['recall'][cl]-v['arms'][ref]['recall'][cl];contrib.append([s,arm,ref,cl,'tiny' if cl in [1,2,12] else 'other',dr/(6*n),dr/(6*nt) if c['train_counts'][cl]<250 else 0]);mask=y==cl;rp=p['predictions'][ref]==y;mp=p['predictions'][arm]==y;harm.append([s,arm,ref,cl,int(mask.sum()),int((mask&rp&~mp).sum()),int((mask&~rp&mp).sum())])
    for j,k in enumerate(p['components']):influence[k]=influence.get(k,0)+float(delta[j])*100/(6*n*int(den[y[j]]))
   sensitivity.append({'arm':arm,'reference':ref,'components':len(influence),'max_component_contribution_pp':max(influence.values()),'min_component_contribution_pp':min(influence.values()),'sum_contribution_pp':sum(influence.values())})
 csvout(r/'CONTRIBUTION_DECOMPOSITION.csv',['stage','arm','reference','class','group','mean_macro_contribution_pp','mean_tail_contribution_pp'],contrib);csvout(r/'RESCUE_HARM.csv',['stage','arm','reference','class','VAL_count','harm_exposures','rescue_exposures'],harm);write(r/'COMPONENT_SENSITIVITY.json',{'scope':'Additive influence with original denominators, same component grouped across stages; no patient bootstrap or new independent cases','records':sensitivity})
 diag=json.loads((r/'GRADIENT_DIAGNOSTICS.json').read_text());keys=sorted(set().union(*(set(v) for v in diag)));csvout(r/'GRADIENT_DIAGNOSTICS.csv',keys,[[v.get(k,'') for k in keys] for v in diag]);agg={}
 for line in (r/'TRAIN_DIAGNOSTICS.private.jsonl').open():
  z=json.loads(line);v=agg.setdefault((z['stage'],z['arm']),{'updates':0,'clipped':0,'norm':0.,'episodes':0,'stats':{}});v['updates']+=1;v['clipped']+=z['gradient_clipped'];v['norm']+=z['grad_norm_before_clip']
  for ep in z['episodes']:
   v['episodes']+=1
   for k,x in ep.items():v['stats'][k]=v['stats'].get(k,0)+x
 flat=[{'stage':s,'arm':arm,'updates':v['updates'],'clipping_fraction':v['clipped']/v['updates'],'grad_norm_before_clip_mean':v['norm']/v['updates'],**{k:x/v['episodes'] for k,x in v['stats'].items()}} for (s,arm),v in agg.items()];keys=sorted(set().union(*(set(v) for v in flat)));csvout(r/'POLICY_DIAGNOSTICS.csv',keys,[[v.get(k,'') for k in keys] for v in flat]);lines=['# R99 动作KL与分类行为保持：完整匹配结果','',a['conclusion'],'',f"主候选CF_BHV。A={a['gate_A']}；B={a['gate_B']}；约束机制C={a['constraint_gate_C']}。",'','|arm|平均macro|最终macro|平均tail|最终tail|','|---|---:|---:|---:|---:|']
 for arm in MODES:lines.append('|'+arm+'|'+'|'.join(f"{a[k][arm][m]:.6f}" for k,m in [('mean','macro'),('final','macro'),('mean','tail'),('final','tail')])+'|')
 lines+=['','九臂从同一R98 SUP0固定参照重新开始fresh optimizer。本轮新增13824更新，R98 warmup3072为借用历史成本，不是本轮新增或免费全生命周期。强R67表示冻结，全历史TRAIN预算，非严格小记忆CL或患者独立确认。','', 'BHV尺度由固定SUP0/S/B输出小扰动校准，不看F/VAL；仅匹配起点局部尺度，不保证全程等强。NONE用于识别保持项整体作用；不事后挑NONE为主候选。PATH若相当/更强，不支持采样RL独立优势；SC同样改善不能归给RL。差分中的差分及全部对照正负见RESULT_SUMMARY。', '', '全部54最终和54中间端点先封存，再S-full→F读出，再全TRAIN部署并一次VAL，TEST0。tiny最终VAL分母1/1/2；正确保护不等于已修好。逐类、原分母贡献、T/SUP0救回损害及component影响均保存，不发布身份或逐例。','', 'gradient诊断拆分CE/纯RL/加权保持，不将辅助+KL合并当纯RL。实际分类翻转与utility并列报告，分数变化或奖励上涨不当分类成功。', '', '完整成本见COSTS；native/profile已含worker，不重复相加。历史借用教师/特征和SUP0的费用不算本轮新增但必须披露。准备/SSH/发布CPU非0未全采样。监测关闭，完成后不自动追加实验。'];(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return a
