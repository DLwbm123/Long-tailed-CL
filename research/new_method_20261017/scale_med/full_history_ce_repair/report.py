"""Render aggregate evidence, including missing or unresolved units."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
def read(name,default=None):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else default
r=read('aggregate.json');ex=read('EXIT.json');co=read('COORDINATOR_EXIT.json');partial=read('partial.json',{})
evidence=dict(result=r,exit=ex,coordinator=co,partial=None if r else partial,selection_seal=read('SELECTION_SEAL.json'),model_seal=read('MODEL_SEAL.json'),cost_receipt=read('COST_RECEIPT.json'),failure=read('FAILURE.json'))
(out/'aggregate.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
lines=['# 完整历史CE独立优化修复结果','', '同原冻结G/P/C表示、原永久训练与验证划分；CE是特权完整历史参照，不满足退休部署约束，不是BA上界或新方法。原内部测试已研究暴露。','',f"退出回执：worker={ex.get('exit_code') if ex else None}；coordinator={co}。",'']
if r:
 lines += [f"科学判断：`{r['decision']}`；完整收敛臂：{r['model_seal']['complete_arms']}；通过臂：{r['passing_arms']}。",'', '|臂|初始选择lambda|初始BA%|+类4 BA%|+类5 BA%|平均J%|类4出生%|类4转旧%|类5出生%|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for a in ['G','P','C']:
  for mode in ['ridge','CE']:
   n=a+'_'+mode;m=r['summary'][n]
   if not all(str(s) in m for s in range(3)):
    lines.append(f'|{n}|{r["chosen"].get(a)}|未决|未决|未决|未决|未决|未决|未决|');continue
   vals=[100*m[str(s)]['BA'] for s in range(3)];J=(vals[1]+vals[2])/2
   row=[n,str(.01 if mode=='ridge' else r['chosen'][a])]+[f'{v:.4f}' for v in vals+[J,100*m['1']['recall'][4],100*m['2']['recall'][4],100*m['2']['recall'][5]]]
   lines.append('| '+' | '.join(row)+' |')
 lines+=['','|CE臂减ridge控制|J差pp|95%区间|9项调整区间|','|---|---:|---|---|']
 for a,v in r['comparisons'].items():
  for b,z in v['vs_ridge'].items():lines.append(f"|{a}_CE−{b}_ridge|{z['J_gain_pp']:+.4f}|{z['CI95_pp']}|{z['CI9_adjusted_pp']}|")
 lines+=['','## 优化认证','', '|臂|阶段|lambda|状态|原坐标梯度max|目标|closure次数|wall秒|','|---|---:|---:|---|---:|---:|---:|---:|']
 for f in r['fits']:lines.append(f"|{f['arm']}|{f['stage']}|{f['lambda_value']}|{f['status']}|{f['gradient_max_abs']:.3e}|{f['final_objective']:.8f}|{f['closure_evaluations']}|{f['wall_seconds']:.2f}|")
 lines+=['','## 逐类风险','', '|臂|阶段|图像召回%按类顺序|组等权BA%|','|---|---:|---|---:|']
 for n,ms in r['summary'].items():
  for st,m in ms.items():lines.append(f"|{n}|{st}|{[round(v*100,4) for v in m['recall']]}|{100*m['group_equal_BA']:.4f}|")
 lines+=['','## 预设门','']+[f'- {a}: {v}' for a,v in r['gates'].items()]
 lines+=['','## 成本和访问边界','','```json',json.dumps(r['costs'],ensure_ascii=False,indent=2),'```','','协调墙钟包含worker正文，不相加。GPU驻留为保守阶段墙钟上界，不是精确硬件计费。原global缓存成本未重新发生；旧训练逐图特征在本轮被特权离线复用，不能称可部署无回放。', '', '所有训练统计及原ridge输出parity见aggregate；不发布原图、身份、逐例分数、头或私有路径。']
else:
 lines+=['科学未决。未以缺失结果、未收敛或工程失败充当CE无效证据；不追加预算或重试。','',f'本轮真实数据已完成优化单元数：{len(partial.get("fits",[]))}。', '', '合成自检未通过认证，原7个已收敛头尚未复核，新增最多8头均未执行。原完整历史CE优化未决结论保留；本轮是此前置工程未决。', '', '## 失败及成本回执', '', '```json', json.dumps(dict(failure=read("FAILURE.json"),cost=read("COST_RECEIPT.json")),ensure_ascii=False,indent=2), '```']
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
