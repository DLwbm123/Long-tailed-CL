"""Render aggregate evidence, including missing or unresolved units."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
def read(name,default=None):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else default
r=read('aggregate.json');r=r.get('result') if isinstance(r,dict) and 'result' in r else r;ex=read('EXIT.json');co=read('COORDINATOR_EXIT.json');partial=read('partial.json',{})
evidence=dict(result=r,exit=ex,coordinator=co,partial=None if r else partial,selection_seal=read('SELECTION_SEAL.json'),model_seal=read('MODEL_SEAL.json'),cost_receipt=read('COST_RECEIPT.json'),failure=read('FAILURE.json'),worker_resource=read('WORKER_RESOURCE_RECEIPT.json'),selfcheck=read('SELFCHECK.json'),initialization=read('INITIALIZATION.json'),environment_check=read('ENVIRONMENT_CHECK.json'),reuse=read('REUSE_RECEIPT.json'))
evidence.update(verification=read('VERIFICATION.json'),budget_verification=read('BUDGET_VERIFICATION.json'),decision='UNRESOLVED_ENGINEERING_NO_EXTENSION' if not r else r['decision'],method_success=False)
(out/'aggregate.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
lines=['# 完整历史CE接受点验收与Newton-CG修复独立运行结果','', '同原冻结G/P/C表示、原永久训练与验证划分；CE是特权完整历史参照，不满足退休部署约束，不是BA上界或新方法。原内部测试已研究暴露。','',f"退出回执：worker={ex.get('exit_code') if ex else None}；coordinator={co}。",'']
if r:
 lines += [f"科学判断：`{r['decision']}`；完整收敛臂：{r['model_seal']['complete_arms']}；通过臂：{r['passing_arms']}。",'', '|臂|初始选择lambda|初始BA%|+类4 BA%|+类5 BA%|平均J%|类4出生%|类4转旧%|类5出生%|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for a in ['G','P','C']:
  for mode in ['ridge','CE']:
   n=a+'_'+mode;m=r['summary'][n]
   def metric(stage,key,index=None):
    if str(stage) not in m:return None
    value=m[str(stage)][key];return 100*(value if index is None else value[index])
   vals=[metric(stage,'BA') for stage in range(3)];J=(vals[1]+vals[2])/2 if vals[1] is not None and vals[2] is not None else None
   row=[n,'0.01' if mode=='ridge' else ('未决' if r['chosen'][a] is None else str(r['chosen'][a]))]+['未决' if v is None else f'{v:.4f}' for v in vals+[J,metric(1,'recall',4),metric(2,'recall',4),metric(2,'recall',5)]]
   lines.append('| '+' | '.join(row)+' |')
 lines+=['','|CE臂减ridge控制|J差pp|95%区间|9项调整区间|','|---|---:|---|---|']
 for a,v in r['comparisons'].items():
  for b,z in v['vs_ridge'].items():lines.append(f"|{a}_CE−{b}_ridge|{z['J_gain_pp']:+.4f}|{z['CI95_pp']}|{z['CI9_adjusted_pp']}|")
 lines+=['','## 优化认证','', '|臂|阶段|lambda|复用|状态|停止分支|原坐标梯度max|目标|值梯度次数|wall秒|','|---|---:|---:|---|---|---|---:|---:|---:|---:|']
 for f in r['fits']:lines.append(f"|{f['arm']}|{f['stage']}|{f['lambda_value']}|{f.get('reused',False)}|{f['status']}|{f['stop_reason']}|{f['gradient_max_abs']:.3e}|{f['final_objective']:.8f}|{f['closure_evaluations']}|{f['wall_seconds']:.2f}|")
 lines+=['','## 本轮新增求解工作量','','|臂|阶段|lambda|点类型|总算子|HVP|Newton步|转换原因|','|---|---:|---:|---|---:|---:|---:|---|']
 for f in r['fits']:
  if not f.get('reused'):
   lines.append(f"|{f['arm']}|{f['stage']}|{f['lambda_value']}|{f.get('point_kind')}|{f.get('operator_evaluations')}|{f.get('operator_counts',{}).get('HVP')}|{f.get('newton_steps')}|{f.get('phase_transition')}|")
 lines+=['','复用行的优化次数和原认证梯度继承旧回执，wall为本轮复核时间；不算本轮重新拟合。主J比较缺失时不生成区间，不将缺失当零收益。','', '## 逐类风险','', '|臂|阶段|图像召回%按类顺序|组等权BA%|','|---|---:|---|---:|']
 for n,ms in r['summary'].items():
  for st,m in ms.items():lines.append(f"|{n}|{st}|{[round(v*100,4) for v in m['recall']]}|{100*m['group_equal_BA']:.4f}|")
 lines+=['','## 预设门','']+[f'- {a}: {v}' for a,v in r['gates'].items()]
 lines+=['','## 成本和访问边界','','```json',json.dumps(r['costs'],ensure_ascii=False,indent=2),'```','','协调墙钟包含worker正文，不相加。GPU驻留为保守阶段墙钟上界，不是精确硬件计费。原global缓存成本未重新发生；旧训练逐图特征在本轮被特权离线复用，不能称可部署无回放。', '', '所有训练统计及原ridge输出parity见aggregate；不发布原图、身份、逐例分数、头或私有路径。']
else:
 lines+=['科学未决。未以缺失结果、未收敛或工程失败充当CE无效证据；不追加预算或重试。','',f'本轮真实数据已完成优化单元数：{len(partial.get("fits",[]))}。', '', '本轮未完整形成科学比较。实际阶段、已完成单元和失败原因以partial、FAILURE及退出回执为准；不把工程失败当CE无效。', '', '## 失败及成本回执', '', '```json', json.dumps(dict(failure=read("FAILURE.json"),cost=read("COST_RECEIPT.json")),ensure_ascii=False,indent=2), '```']
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

if not r:
 lines += ['', '## 已保存的优化结果', '', '|臂|阶段|lambda|复用|状态|原W梯度max|Newton步|总算子（新增）|', '|---|---:|---:|---|---|---:|---:|---:|']
 for f in partial.get('fits',[]):
  lines.append(f"|{f['arm']}|{f['stage']}|{f['lambda_value']}|{f.get('reused')}|{f['status']}|{f['gradient_max_abs']:.8e}|{f.get('newton_steps','继承')}|{f.get('operator_evaluations','继承')}|")
 lines += ['', '15/15优化单元认证并不代表完整科学交付成功：最终aggregate序列化失败；主风险表、区间、门未保存，禁止填为通过或失败。', '', '## 原协议对齐', '', 'G/P/C原完整特征维度1535 / 1375+128 / 1318+128；训练前缀2555/4367/4395；初始681验证选择，各臂lambda均0.0001。九项训练统计parity最大2.84217e-14。MODEL_SEAL在测试之前形成。最终测试得分parity数值未保存，不声称有完整逐项回执。', '', '## 完整成本', '', '```json', json.dumps(read('BUDGET_VERIFICATION.json'),ensure_ascii=False,indent=2), '```', '', '失败前通用资源回执和父进程wait4保留。正文、boot与协调墙钟是包含关系，不相加。累计成本见CUMULATIVE_COSTS.json；历史失败CPU缺失，累计总CPU仍未知。']
 (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
