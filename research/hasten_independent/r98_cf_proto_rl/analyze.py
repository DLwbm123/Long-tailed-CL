"""Saved predictions only: frozen A/B gates, all signs and additive attribution."""
import csv,json,torch
from pathlib import Path
from common import write
from evaluate import MODES

def table(c,pack,s):
 n=sum(map(len,c['stages'][:s+1]));y=pack['labels'];den=torch.bincount(y,minlength=n);assert (den>0).all();tail=torch.tensor(c['train_counts'][:n])<250;old=torch.arange(n)<sum(map(len,c['stages'][:s]));result={}
 for a,pred in pack['predictions'].items():
  cc=torch.bincount(y[pred==y],minlength=n);rec=cc.double()/den;groups={}
  for age,mask in [('old',old),('new',~old),('all',torch.ones(n,dtype=torch.bool))]:
   for freq,fm in [('head',~tail),('tail',tail),('all',torch.ones(n,dtype=torch.bool))]:
    keep=mask&fm;groups[age+'_'+freq]=float(rec[keep].mean()*100) if keep.any() else None
  result[a]={'micro':float(cc.double().sum()/den.sum()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100),'counts':den.tolist(),'correct':cc.tolist(),'recall':(rec*100).tolist(),'groups':groups}
 return {'stage':s,'classes':n,'arms':result}

def summarize(c,records,packs):
 out={'mean':{},'final':{},'CF_comparisons':{},'tiny_harm':{},'conclusion':None}
 for a in MODES:
  out['mean'][a]={k:sum(v['arms'][a][k] for v in records)/6 for k in ['micro','macro','tail']};out['final'][a]={k:records[-1]['arms'][a][k] for k in ['micro','macro','tail']}
 for ref in [a for a in MODES if a!='CF']:
  out['CF_comparisons'][ref]={'mean_delta_pp':{k:out['mean']['CF'][k]-out['mean'][ref][k] for k in ['micro','macro','tail']},'final_delta_pp':{k:out['final']['CF'][k]-out['final'][ref][k] for k in ['micro','macro','tail']}}
 for ref in ['T','SUP0']:
  out['tiny_harm'][ref]={str(cl):sum(int(((p['labels']==cl)&(p['predictions'][ref]==p['labels'])&(p['predictions']['CF']!=p['labels'])).sum()) for p in packs) for cl in [1,2,12]}
 d=out['CF_comparisons']['T'];both=sum(v['arms']['CF']['macro']>=v['arms']['T']['macro']-1e-8 and v['arms']['CF']['tail']>=v['arms']['T']['tail']-1e-8 for v in records);A=d['mean_delta_pp']['macro']>=.5 and d['final_delta_pp']['macro']>=.5 and d['mean_delta_pp']['tail']>=-1e-8 and d['final_delta_pp']['tail']>=1 and both>=4 and all(v==0 for r in out['tiny_harm'].values() for v in r.values());B={}
 for ref in ['SUP0','SC']:
  d=out['CF_comparisons'][ref];B[ref]=d['mean_delta_pp']['macro']>=.3 and d['final_delta_pp']['macro']>=.3 and d['mean_delta_pp']['tail']>=-1e-8 and d['final_delta_pp']['tail']>=-1e-8
 out.update(gate_A=bool(A),gate_B={k:bool(v) for k,v in B.items()},stages_macro_tail_nonnegative=both,passed=bool(A and all(B.values())));out['conclusion']='COMPLETE_CF_DEVELOPMENT_GATE_PASS' if out['passed'] else 'COMPLETE_CF_MECHANISM_NEGATIVE';return out

def csvout(p,header,rows):
 with p.open('w') as f:w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(rows)

def run(c):
 r=Path(c['output']);packs=[torch.load(r/f'VAL_predictions_s{s}.private.pt',map_location='cpu',weights_only=False) for s in range(6)];records=[table(c,p,s) for s,p in enumerate(packs)];F=[table(c,torch.load(r/f'F_predictions_s{s}.private.pt',map_location='cpu',weights_only=False),s) for s in range(6)];a=summarize(c,records,packs);write(r/'RESULT_SUMMARY.json',a);write(r/'STAGE_METRICS.json',records);write(r/'F_TRANSFER.json',F)
 csvout(r/'STAGE_METRICS.csv',['stage','arm','micro','macro','tail'],[[v['stage'],arm,*[m[k] for k in ['micro','macro','tail']]] for v in records for arm,m in v['arms'].items()]);csvout(r/'CLASS_STAGE.csv',['stage','arm','class','count','correct','recall'],[[v['stage'],arm,i,n,m['correct'][i],m['recall'][i]] for v in records for arm,m in v['arms'].items() for i,n in enumerate(m['counts'])]);csvout(r/'GROUP_STAGE.csv',['stage','arm','group','macro'],[[v['stage'],arm,g,x] for v in records for arm,m in v['arms'].items() for g,x in m['groups'].items()])
 contrib=[];harm=[];sensitivity=[]
 for arm in MODES[1:]:
  for ref in ['T','SUP0']:
   influence={}
   for s,(v,p) in enumerate(zip(records,packs)):
    n=v['classes'];nt=sum(x<250 for x in c['train_counts'][:n]);y=p['labels'];diff=(p['predictions'][arm]==y).double()-(p['predictions'][ref]==y).double();den=torch.bincount(y,minlength=n)
    for cl in range(n):
     dr=v['arms'][arm]['recall'][cl]-v['arms'][ref]['recall'][cl];contrib.append([s,arm,ref,cl,'tiny' if cl in [1,2,12] else 'other',dr/(6*n),dr/(6*nt) if c['train_counts'][cl]<250 else 0]);mask=y==cl;right=p['predictions'][ref]==y;new=p['predictions'][arm]==y;harm.append([s,arm,ref,cl,int(mask.sum()),int((mask&right&~new).sum()),int((mask&~right&new).sum())])
    for j,component in enumerate(p['components']):influence[component]=influence.get(component,0)+float(diff[j])*100/(6*n*int(den[y[j]]))
   sensitivity.append({'arm':arm,'reference':ref,'available_components':len(influence),'maximum_component_macro_contribution_pp':max(influence.values()),'minimum_component_macro_contribution_pp':min(influence.values()),'summed_contribution_pp':sum(influence.values()),'no_component_ID_published':True})
 csvout(r/'CONTRIBUTION_DECOMPOSITION.csv',['stage','arm','reference','class','group','mean_macro_contribution_pp','mean_tail_contribution_pp'],contrib);csvout(r/'RESCUE_HARM.csv',['stage','arm','reference','class','VAL_count','harm_exposures','rescue_exposures'],harm);write(r/'COMPONENT_SENSITIVITY.json',{'scope':'Additive component contributions under original denominators, identical component grouped across stages. No patient-independence claim or bootstrap CI; tiny denominators prevent strong inferential claims.','records':sensitivity})
 diag=json.loads((r/'GRADIENT_DIAGNOSTICS.json').read_text());est=json.loads((r/'ESTIMATOR_DIAGNOSTICS.json').read_text());csvout(r/'GRADIENT_ESTIMATOR_DIAGNOSTICS.csv',list(est[0]),[[x[k] for k in est[0]] for x in est]);csvout(r/'GRADIENT_DIAGNOSTICS.csv',list(diag[0]),[[x[k] for k in diag[0]] for x in diag])
 # Aggregate private step records; no sample IDs or class-risk values are published.
 agg={}
 for line in (r/'TRAIN_DIAGNOSTICS.private.jsonl').open():
  z=json.loads(line);key=(z['stage'],z['arm']);v=agg.setdefault(key,{'updates':0,'clipped_updates':0,'grad_norm_before_clip_sum':0.,'stats':{},'episodes':0});v['updates']+=1;v['clipped_updates']+=z['gradient_clipped'];v['grad_norm_before_clip_sum']+=z['grad_norm_before_clip']
  for ep in z['episodes']:
   v['episodes']+=1
   for k,x in ep.items():v['stats'][k]=v['stats'].get(k,0)+x
 flat=[]
 for (s,arm),v in agg.items():flat.append({'stage':s,'arm':arm,'updates':v['updates'],'clipping_fraction':v['clipped_updates']/v['updates'],'grad_norm_before_clip_mean':v['grad_norm_before_clip_sum']/v['updates'],**{k:x/v['episodes'] for k,x in v['stats'].items()}})
 keys=sorted(set().union(*(set(v) for v in flat)));csvout(r/'POLICY_DIAGNOSTICS.csv',keys,[[v.get(k,'') for k in keys] for v in flat]);lines=['# R98 CF-PROTO-RL 完整结果','',a['conclusion'],'','单教师seed74002、全历史TRAIN信息预算；不是严格小记忆CL/患者独立验证。CF为事前唯一主候选，全部正负保留。','',f"方法门A={a['gate_A']}；RL增量门B={a['gate_B']}。",'','|arm|平均macro|最终macro|平均tail|最终tail|','|---|---:|---:|---:|---:|']
 for arm in MODES:lines.append('|'+arm+'|'+'|'.join(f"{a[k][arm][m]:.6f}" for k,m in [('mean','macro'),('final','macro'),('mean','tail'),('final','tail')])+'|')
 lines+=['','CF相对全部控制差值见RESULT_SUMMARY；逐类/阶段/old-new-head-tail、原分母贡献和T/SUP0救回损害均公开聚合。tiny最终VAL分母1/1/2，重复stage不是独立病例。','', 'CF/GLOBAL同一随机期望目标，仅估计器不同；梯度方差比较在同一SUP0参数点、32个配对G8基础噪声上进行，使用裁剪前梯度。PATH是MC重参数化控制，不是exact积分。若PATH相当或更优，不能证明score-function RL有独立优势。连续utility提升不代替准确率门。','', 'F仅为已历史开发且教师见过的TRAIN模块迁移读出；先S-only/F后全TRAIN部署，未按F改配置。全部模型/规则封存后一次VAL，TEST0。','', '完整费用见COSTS.json。已含worker的native/profile/训练/评估子项不重复加；教师历史成本为借用资产，不计本轮新增也不冒充全生命周期免费。准备/SSH/发布CPU非0且未全采样。监测关闭，完成后不自动追加实验。'];(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return a
