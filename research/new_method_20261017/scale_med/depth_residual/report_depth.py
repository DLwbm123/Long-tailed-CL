"""Publish aggregate evidence only, never image identities or learned states."""
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
def read(n):return json.loads((root/n).read_text())
exitinfo=read('EXIT.json');assert exitinfo['exit_code']==0
r=read('results.json');costs={k:read(k+'.costs.json') for k in ['prepare','fit','evaluate']}
evidence=dict(result=r,costs=costs,exit=exitinfo,budget=read('BUDGET.json'),data_audit=read('DATA_AUDIT.json'),base_selection=read('BASE_SELECTION.json'),retirement=read('RETIREMENT.json'),model_seal=read('MODEL_SEAL.json'),transform_seal=read('TRANSFORM_SEAL.json'),selector=read('SELECTOR.json'),feature_parity={k:read(k+'.FEATURE_PARITY.json') for k in ['fit','evaluate']},reparameterization=read('TEST_REPARAMETERIZATION.json'))
(out/'aggregate.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
a=str(r['common_alpha']);lines=['# 固定预算跨层残差最小实验','',f"完成退出码0，墙钟{exitinfo['wall_seconds']:.2f}秒。停止判断：`{r['decision']}`。",'',f"初始G验证选定共同alpha={a}；未来阶段没有重新选参。所有投影只使用初始训练类，三阶段全部头封存后才打开测试。",'', '现有测试历史暴露，患者隔离未知；以下为单序列探索性机制证据。','', '| 方法 | 初始BA % | +类4 BA % | +类5 BA % | 两次转换平均BA % | 类4出生召回 % | 类4转旧召回 % | 类5出生召回 % |','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ['G','R','P','C','S']:
    m=[r['metrics'][str(s)][arm][a] for s in range(3)];avg=(m[1]['BA']+m[2]['BA'])/2
    lines.append(f"| {arm} | {100*m[0]['BA']:.4f} | {100*m[1]['BA']:.4f} | {100*m[2]['BA']:.4f} | {100*avg:.4f} | {100*m[1]['recall'][4]:.4f} | {100*m[2]['recall'][4]:.4f} | {100*m[2]['recall'][5]:.4f} |")
lines+=['','G=全局；R=随机中间层；P=普通中间层PCA；C=条件残差PCA；S=Head2Toe式监督坐标选择（非论文完整复现）。','','| C减控制 | 平均BA差 pp | 配对95%区间 pp | 类4出生差 pp | 类4转旧差 pp | 类5出生差 pp |','|---|---:|---|---:|---:|---:|']
for k,v in r['contrasts_C_minus_control'].items():
    ci=v['CI95_pp'];lines.append(f"| {k} | {v['mean_BA_gain_pp']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {v['class4_birth_gain_pp']:+.4f} | {v['class4_retired_gain_pp']:+.4f} | {v['class5_birth_gain_pp']:+.4f} |")
lines+=['',f"四个控制最小平均增益的联合bootstrap下界：{r['minimum_control_gain_lower95_pp']:+.4f}pp。",'', '## 预设门与停止','']
for k,v in r['gates'].items():lines.append(f'- {k}: {v}')
lines+=['','任一科学门失败就停止，不追加层、秩、顺序、种子、正则或救援规则。残差化本身只是坐标变换；与正确变换惩罚后的直接坐标logit一致，不能将它写成新增信息或新函数族。','','## 状态和成本','','| 方法 | 输出维数 | 六类持久状态 bytes | cap内 |','|---|---:|---:|---|']
for arm,v in evidence['budget']['final_bytes'].items():
    width=evidence['budget']['width'][arm]+(128 if arm in ['R','P','C'] else 0);lines.append(f"| {arm} | {width} | {v} | {v<=evidence['budget']['cap_bytes']} |")
lines+=['','cap=19,058,800bytes，float64；包含训练统计、一个部署头、投影/选择索引、全部归一化和固定标量余量。另两正则头与历史快照在诊断成本单列。共有冻结编码器不计入方法间cap，但其字节与额外计算实测报告。旧全局缓存生成成本排除，本次中间层提取计入。','','## 结果解释','']
failed=[k for k,v in r['gates'].items() if not v]
lines.append('未通过：'+', '.join(failed)+'。本轮不支持该固定秩、固定层和初始类学习规则的后续扩展。' if failed else '预设探索性门通过，但不能替代独立患者隔离确认；不自动启动下一实验。')
lines+=['','仅初始类收益不足以支持未来迁移；类4出生收益与转旧后收益须分开看。未来类别下降反对本轮子空间迁移假设，不证明所有中间层无用。监督控制的500步特征排名不等同精确优化，见记录的proximal residual。','']
(out/'REPORT_ZH.md').write_text('\n'.join(lines))
