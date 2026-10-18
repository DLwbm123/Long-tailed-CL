"""Aggregate quota audit and bounded feasibility outcomes, without private rows."""
import json
from pathlib import Path
def generate(root,out):
    root,out=Path(root),Path(out);out.mkdir(exist_ok=True)
    def read(n):return json.loads((root/n).read_text())
    quota=read('QUOTA_SEAL.json');results={a:read(a+'.result.json') for a in ['G','C','U']}
    public=dict(quotas=quota,results=results,exit=read('EXIT.json'),prepare_costs=read('prepare.costs.json'),feature_parity=read('FEATURE_PARITY.json'),dispatch_seal=read('DISPATCH_SEAL.json'),pre_dispatch_audit=read('PRE_DISPATCH_AUDIT.json'))
    (out/'aggregate.json').write_text(json.dumps(public,indent=2,allow_nan=False))
    lines=['# 三空间训练配额可行性诊断','', '训练支持上的有限样本存在性诊断；不使用测试集定目标，不进行验证/测试评测，也不属于合法部署记忆或等预算方法比较。','', '| 类别 | 训练样本 | G正确 | C正确 | 冻结配额 |','|---|---:|---:|---:|---:|']
    for c in range(6):lines.append(f"| {c} | {quota['training_counts'][c]} | {quota['reference']['G']['certified_correct'][c]} | {quota['reference']['C']['certified_correct'][c]} | {quota['quotas'][c]} |")
    lines+=['','| 空间 | 特征维数 | 结论 | 见证逐类正确数 |','|---|---:|---|---|']
    for a,r in results.items():lines.append(f"| {a} | {quota['dimensions'][a]} | {r['classification']} | {r.get('verification',{}).get('certified_correct','无已核验见证')} |")
    feasible=[a for a,r in results.items() if r['classification']=='FEASIBLE_WITNESS']
    lines+=['','## 解释与停止','']
    if 'G' in feasible:lines+=['全局空间已找到达到全部训练配额的分类头。针对这些训练配额，额外中间表示不是必要条件；原ridge参考点不能代表全局线性函数族的全部能力。这不证明该头可泛化或能从合法历史记忆中学到。','']
    if 'C' in feasible:lines+=['冻结C参考头本身就达到全部训练配额，其各类训练正确数均不低于G；独立核验后直接记录见证，没有重复求解C。训练支持不存在这两个参考点所定义的必要取舍，不能据查询类别损失认定C训练信息不足。','']
    if 'U' in feasible:lines+=['联合空间存在可核验见证。若G/C仅未决，不能据此声称被删全局坐标是必要信息；必须有可信的负面证明才能作此比较。','']
    if not feasible:lines+=['本次预算内没有得到可核验见证，三个空间的可行性仍未决。超时、LP不成功或普通浮点infeasible均不证明目标不能兼得。','']
    lines+=['每空间300秒，LP最多90秒并计入其中；CPU最多4线程，RSS16GiB。时间不跨空间转移，没有追加预算或重试。固定正确样本LP只用于寻找充分见证，其失败不否定完整配额问题。完整指示模型允许自己选择判对的样本，没有任意权重界或big-M。','','独立检查在完整训练矩阵上重新argmax，对边界比较用精确二进制有理数运算。普通SCIP浮点不可行不是数学证明；本轮未配置精确不可行证书核验链。训练可行也不代表测试收益，更不代表原无旧样本方法已成立。所有结果结束在本轮，不自动产生新方法或后续实验。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return public
