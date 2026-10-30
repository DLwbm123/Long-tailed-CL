import os,json
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);rows=json.loads((r/'RECORDS.private.json').read_text());assert json.loads((r/'VERIFICATION.json').read_text())['passed'];held=[v for v in rows if v['query']=='heldout'];controls=['U','CE','E','H'];metrics=['expected_macro','expected_tail'];eps=1e-8
def mean(x):return sum(x)/len(x) if x else None
def gains(rr,ref):return {m:mean([v['arms']['G'][m]-v['arms'][ref][m] for v in rr]) for m in metrics}
stages=[]
for stage in range(6):
 rr=[v for v in held if v['stage']==stage];gg={c:gains(rr,c) for c in controls+['NCM','balanced']};stages.append(dict(stage=stage,gains=gg,all8positive=all(gg[c][m]>eps for c in controls for m in metrics),arms={a:{m:mean([v['arms'][a][m] for v in rr]) for m in metrics+['modal_macro','modal_tail']} for a in ['G']+controls+['NCM','balanced']}))
global_gain={c:gains(held,c) for c in controls+['NCM','balanced']};directions=[dict(direction=d,gains={c:gains([v for v in held if v['direction']==d],c) for c in controls}) for d in [0,1]]
classes=[]
for c in range(23):
 rr=[v for v in held if c<len(v['arms']['NCM']['counts']) and v['arms']['NCM']['counts'][c]>0];dd={}
 for ref in controls+['NCM','balanced']:
  dif=[100*(v['arms']['G']['expected_correct'][c]-v['arms'][ref]['expected_correct'][c])/v['arms'][ref]['counts'][c] for v in rr];dd[ref]=dict(mean_expected_recall_gain_pp=mean(dif),positive=sum(x>eps for x in dif),negative=sum(x<-eps for x in dif),same=sum(abs(x)<=eps for x in dif))
 classes.append(dict(class_index=c,cells=len(rr),class1_fold1_unsupported=c==1,gains=dd))
tiny=[classes[c]['gains']['NCM']['mean_expected_recall_gain_pp'] for c in [1,2,12]];tinygate=all(v>=-eps for v in tiny) and sum(v>eps for v in tiny)>=2
final=stages[-1];finalgate=all(final['gains']['NCM'][m]>eps for m in metrics) and all(final['arms']['G'][m]-final['arms']['NCM'][m]>eps for m in ['modal_macro','modal_tail'])
dgate=all(v['gains'][c][m]>eps for v in directions for c in ['U','CE'] for m in metrics);sg=sum(v['all8positive'] for v in stages);primary=all(global_gain[c][m]>eps for c in controls for m in metrics) and sg>=5 and dgate and finalgate and tinygate
result=dict(global_heldout_gain_pp=global_gain,all8positive_stages=sg,directions=directions,tiny_gate=tinygate,final_expected_and_modal_NCM_gate=finalgate,GRPO_specific_necessary_gate=primary,not_full_CIL_efficacy=True,not_independent_confirmation=True,new_VAL_TEST_reads=0)
write(r/'ANALYSIS.json',result);write(r/'STAGE_METRICS.json',stages);write(r/'CLASS_RESPONSES.json',classes)
text='# R87：两步分类仲裁策略与GRPO额外价值\n\n固定R81/R84保存CLS/NCM分数，双fold、双方向、3条件stream、六阶段；288policy snapshots/36864controlleroptimizersteps，0视觉优化/编码/VALTEST。分类动作不是生成，最多两个不重复竞争对的分数交换；真实类不在top2不能纠正。\n\n'
text+=f'GRPO特异必要门 **{primary}**；全8项同正阶段 **{sg}/6**；tiny门 **{tinygate}**；最终期望与modal对NCM门 **{finalgate}**。\n\n|G减参照|heldout期望宏pp|heldout期望尾pp|\n|---|---:|---:|\n'
for c,v in global_gain.items():text+=f"|{c}|{v['expected_macro']:+.9f}|{v['expected_tail']:+.9f}|\n"
text+='\nU为同legal tree的祖先uniform，CE为同网络精确监督CE目标，E为同网络精确终点reward梯度，H为reward循环打乱GRPO。均同128step/stage，实际policy计算量不等；没有GPU效率声明。若只胜CE/U不胜E，只能支持目标差异，不能宣称GRPO特异收益。\n\n原Q有历史开发暴露，本次heldout只指未进该方向policy loss的component；另一方向独立参数。class1fold1缺双slice，必要少类门仅fold0class1覆盖，未补独立证据。期望准确率是随机策略的精确期望，非实际独立正确图像数；modal为最大joint概率轨迹，不按测试reward选。全Q含反馈行仅描述。全部stage/fold/direction/stream/类方向完整保留，不挑方向、阶段或stream。\n'
(r/'RESULTS_ZH.md').write_text(text)
