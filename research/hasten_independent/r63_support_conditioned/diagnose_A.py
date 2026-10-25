from pathlib import Path
import json,time,statistics,math
r=Path('/tmp/q1093');o=r/'post_diagnostics';clock=json.loads((o/'FIRST_ACTUAL.json').read_text());assert time.time()<clock['deadline_unix'];d=json.loads((o/'SAVED_DERIVATIONS.json').read_text());cur=json.loads((r/'results/FULL_TRAIN_CURVES.json').read_text());cl=json.loads((r/'results/ALL_CLASS_COUNTS.json').read_text());rows=[];summary={}
for ds,ss in d.items():
 finals=[]
 for z in ss:
  seed=z['seed'];final=z['rows'][-1];nc=next(v['train_counts'] for v in cl if v['dataset']==ds and v['seed']==seed and v['stage']==final['stage']);tail=[k for k,n in enumerate(nc) if n<250];birth={};prev=0
  for v in z['rows']:
   for c in v['classes'][prev:]:birth[c['class_index']]=(v['stage'],c)
   prev=v['seen']
  curve=max([v for v in cur if v['dataset']==ds and v['seed']==seed],key=lambda x:x['stage'])['curves'][-1]
  for c in final['classes']:
   k=c['class_index'];b,abc=birth[k];den=c['count'];assert abc['C']==c['C'];tr=curve['query_counts'][k];tc=curve['crossfit_query_correct'][k]
   rows.append(dict(dataset=ds,seed=seed,class_index=k,birth_stage=b,TRAIN_count=tr,TRAIN_crossfit_correct=tc,TRAIN_crossfit_BA=100*tc/tr,VAL_count=den,VAL_correct=c['C'],VAL_BA=100*c['C']/den,TRAIN_VAL_gap_pp=100*(tc/tr-c['C']/den),is_tail=k in tail,tail_single_query_step_pp=100/(len(tail)*den) if k in tail else None,**{f'birth_{kk}':vv for kk,vv in abc.items() if kk not in ['class_index','count']}))
  cells=[v for v in rows if v['dataset']==ds and v['seed']==seed and v['is_tail']]
  finals.append(dict(seed=seed,tail_A=statistics.mean(100*v['birth_A']/v['VAL_count'] for v in cells),tail_B=statistics.mean(100*v['birth_B']/v['VAL_count'] for v in cells),tail_C=statistics.mean(v['VAL_BA'] for v in cells),TRAIN_tail=statistics.mean(v['TRAIN_crossfit_BA'] for v in cells),tail_AB_rescue=sum(v['birth_AB_rescue'] for v in cells),tail_AB_harm=sum(v['birth_AB_harm'] for v in cells),tail_BC_rescue=sum(v['birth_BC_rescue'] for v in cells),tail_BC_harm=sum(v['birth_BC_harm'] for v in cells),mean_tail_birth_final_proto_cos=statistics.mean(v['birth_prototype_cos'] for v in cells)))
 summary[ds]={'seeds':finals,'mean':{k:statistics.mean(v[k] for v in finals) for k in finals[0] if k!='seed'},'tail_classes':len(tail),'final_class_count':len(nc)}
(o/'ALL_CLASS_TRAIN_VAL_ABC.json').write_text(json.dumps(rows,indent=2));(o/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(o/'LIMITATIONS.json').write_text(json.dumps({'H0_R28_per_query_paired':'not computed: this finite input bundle contains aggregate reference counts, no aligned reference per-query responses; no rescoring','fold_prototypes':'not saved; full prototype drift is not fold inconsistency','TRAIN_crossfit_not_independent_holdout':True,'A_to_B_mixes_encoder_prototype_and_training_history':True,'B_to_C_fixed_final_scores_candidate_set_intervention':True,'no_patient_independence':True,'no_conflict_gradient_evidence':True},indent=2))
lines=['# R63结案后A段：已有输出的有限诊断','','0新模型前向、0backward、0optimizer。仅保存分数的argmax/候选列限制、计数和小原型内积，正式90端点不重评分。全域全seed/class结果保留。','','|域|尾类出生A|最终限出生类B|最终全类C|最终TRAIN跨折尾|A→B变化|B→C变化|','|---|---|---|---|---|---|']
for ds,z in summary.items():
 m=z['mean'];lines.append(f"|{ds}|{m['tail_A']:.4f}|{m['tail_B']:.4f}|{m['tail_C']:.4f}|{m['TRAIN_tail']:.4f}|{m['tail_B']-m['tail_A']:.4f}|{m['tail_C']-m['tail_B']:.4f}|")
lines+=['','A是各类出生模型/出生类别集合，B是最终模型/出生类别集合，C是最终全类集合；同类查询ID/标签对齐。A→B混合编码器和原型更新，非纯遗忘；B→C是固定最终分数下后来竞争类别影响，非训练机制因果占比。出生各类阶段不同，汇总须按逐类等权解释。','','训练到开发的巨大落差在三域均存在；限制后到类是否恢复以表为准，不把支持/查询组分离称独立holdout。HK部分VAL类1/2行，报告离散步长，不用三seed替代独立样本。全支持方向变化不证明两折证据不稳定，亦不支持直接梯度投影。','','当前A段已经支持：固定R63没有解决跨折训练到开发迁移；不得由其高TRAIN或单次冲突假设直接启动PCGrad。先读A/B/C与逐类证据，必要的来源/分布可实施性核对优先于计算全梯度。B段不自动执行整套清单。若新实验不能解决独立查询问题，继续反复全TRAIN调参无根据。','','H0/R28参考输入仅聚合计数，未虚构逐例配对；没有患者级身份保证，不做患者bootstrap。未测两折原型和梯度，不能把缺失项说成阴性。']
(o/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');elapsed=time.time()-clock['start_unix'];size=sum(p.stat().st_size for p in o.iterdir() if p.is_file());assert size<64000000 and elapsed<1800;(o/'CONTROL_EXIT.json').write_text(json.dumps({'exit_code':0,'elapsed_seconds':elapsed,'allocated_logical_bytes':size,'new_model_forwards':0,'new_backward':0,'new_optimizer':0},indent=2));print(json.dumps(summary,indent=2))
