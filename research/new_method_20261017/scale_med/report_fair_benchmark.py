"""Export aggregate-only tables from the prespecified benchmark results."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = json.loads((root / 'results.json').read_text())
assert json.loads((root / 'status.json').read_text())['state'] == 'COMPLETE'
summary, classes, primary = [], [], {}
lines = ['# 固定共同支持方案：同划分公平比较', '',
         '冻结源代码提交：97f0201b03ab6211ba0bbe715125f9468084147c。主候选为 consensus，主参照为 B；没有训练、重新拟合尺度、扫描参数或根据测试结果选择新主方案。', '',
         '| 数据集 | 方法 | BA % | 旧类 BA % | 新类 BA % | 相对 B pp | 95% CI pp |',
         '|---|---|---:|---:|---:|---:|---|']
order = ['B', 'consensus', 'B_density', 'B_distance', 'A', 'ridge', 'uniform', 'local_shrink', 'distance_local']
for result in results:
    name, metrics = result['dataset'], result['metrics']
    base = metrics['B']['summary']
    for arm in order:
        value = metrics[arm]
        score = value['summary']
        ci = result['intervals']['arms'][arm]['all']['delta_BA_pp_CI95']
        delta = 100 * (score['all']['BA'] - base['all']['BA'])
        lines.append(f"| {name} | {arm} | {100*score['all']['BA']:.4f} | {100*score['old']['BA']:.4f} | {100*score['new']['BA']:.4f} | {delta:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        for group, values in score.items():
            interval = result['intervals']['arms'][arm][group]
            lo, hi = interval['delta_BA_pp_CI95']
            summary.append(dict(dataset=name, arm=arm, group=group, **values,
                                delta_BA_pp=100*(values['BA']-base[group]['BA']),
                                CI95_low_pp=lo, CI95_high_pp=hi,
                                group_equal_BA=interval['group_equal_BA']))
        classes.extend(dict(dataset=name, arm=arm, **row) for row in value['classes'])
    deltas = {g:100*(metrics['consensus']['summary'][g]['BA']-base[g]['BA']) for g in base}
    ci = result['intervals']['arms']['consensus']['all']['delta_BA_pp_CI95']
    primary[name] = dict(delta_pp=deltas, CI95_pp=ci,
                         descriptive_positive=deltas['all'] > 0,
                         old_or_new_loss_over_1pp=min(deltas['old'],deltas['new']) < -1,
                         positive_lower_CI=ci[0] > 0)
lines += ['', '## 主比较与机制', '']
for result in results:
    name = result['dataset']
    p = primary[name]
    pair = result['paired']['B']['all']
    bound = result['retention']['all']['harm_bound']
    assert bound['subset_verified'] and bound['observed'] <= bound['upper_bound']
    lines.append(f"- {name}：主方案相对 B 总体 {p['delta_pp']['all']:+.4f} pp，旧类 {p['delta_pp']['old']:+.4f} pp，新类 {p['delta_pp']['new']:+.4f} pp；纠正 {pair['rescue']} 张、损害 {pair['harm']} 张。共同损害集合上界 {bound['upper_bound']}，实测 {bound['observed']}，子集约束通过。旧/新类下降超过 1 pp：{p['old_or_new_loss_over_1pp']}。")
    for arm in ['B_density','B_distance']:
        r = result['retention']['all'][arm]
        gap = 100*(result['metrics']['consensus']['summary']['all']['BA']-result['metrics'][arm]['summary']['all']['BA'])
        lines.append(f"  相对 {arm} 总体 {gap:+.4f} pp；保留其纠正 {r['retained']}/{r['reference_rescues']}，丢失 {r['lost']}，额外纠正 {r['additional_rescues']}。")
strong = all(p['positive_lower_CI'] and not p['old_or_new_loss_over_1pp'] for p in primary.values())
lines += ['', '本次未显示共同支持方案稳定优于 B：ISIC 点估计为负，HK 点估计接近零，两个主区间均跨零。约束成立只说明损害来自两种加法融合共同出错的集合，不能保证该集合为空，也不能保证收益超过损害。ISIC 仍损害 14 张，HK 仍损害 1 张；同时相对 density 两个数据集各丢失 7 个纠正。该保守机制没有在测试上形成稳定净收益。', '',
          'HK 的 4 次纠正集中于样本数 89/130 的类，而 1 次损害来自样本数 27 的类；宏平均对后者赋予更高的单图权重，所以净纠正 3 张仍只增加 0.0291 pp。不能用纠正总数替代 BA。B 在 ISIC 的九项中点估计最高；HK 的 density 融合点估计最高，但相对 B 的区间仍跨零，不能据此改选主方案或宣称确定优势。']
lines += ['', f"预先规定的双数据集较强正面证据条件：{'满足' if strong else '不满足'}。该条件仅适用于本次复用基准。主方案结果不改变预先指定的主比较，不从其余八项中另选主方法。", '',
          '## 数据与解释边界', '',
          'ISIC：586 张、500 个已知身份组，首转换 4+2 类；HK：1271 张、1271 个精确图像身份组，首转换 13+2 类。当前支持集及开发划分与测试在路径、已知身份组及既有内容标识上的重叠为零，测试图像全部存在；没有重新计算哈希。HK 身份组不是已验证患者/检查组，患者级隔离未知。', '',
          'ISIC 存在此前完成的测试批次；HK 暴露记录含历史 train/test 成员关系且不满足独立确认。成员关系不等于已证明每张图都用于训练。本轮公式直接依据 Qfit/Qgate 开发，未以该独立于开发划分的测试成绩调参，但不能排除继承的设计选择受到历史测试的间接影响。因此这是已有划分上的公平比较，不是全新盲测、完整持续学习序列、完整 8/23 类榜单或外部临床验证。', '',
          '区间使用类别内已知身份组的配对 bootstrap，10000 次、seed 74003，各方法共用抽样。主指标仍为图像级类别平均召回率；group_equal_BA 为次要敏感性指标。HK 有每类仅 2/3 个样本的类别，区间可能脆弱或退化；区间不校正历史反复使用及方法选择。CE 只作描述，各方法尺度不同，不能据此声称校准改善。', '',
          'B 与主方案使用相同支持行，主方案额外使用局部统计和第二条特征路径，并非等计算/等存储。A 压缩旧类历史，ridge 使用不同表示和权重；局部方法为消融。这里没有复现新的外部 SOTA 方法。', '',
          '## 成本与交付', '',
          'costs.json、B_FEATURE_COST.json 和 total_cost.json 记录本次两阶段成本；既有支持集统计/缓存构建成本不包含在内，不从总耗时推断逐方法延迟。results.json 含不计编码器的逻辑模型状态字节数。两条特征路径均通过冻结容差下的历史分数/特征一致性检查。', '',
          '公开内容限代码、协议、聚合指标、类别召回/混淆矩阵、成本和检查摘要。原始图像、身份信息、逐样本分数、特征及模型权重不公开。固定测试完成后不追加种子、阈值扫描或公式修补。']
cost = json.loads((root / 'total_cost.json').read_text())
lines += ['', f"本次两阶段总墙钟耗时 {cost['wall_seconds']:.2f} 秒（含启动开销）。"]
for result in results:
    state = result['logical_state_bytes_excluding_encoders']
    additional = sum(v for k,v in state.items() if k.startswith('additional_'))
    lines.append(f"- {result['dataset']}：B 头 {state['B']} 字节，局部均值/协方差、参考协方差及投影额外 {additional} 字节；不含编码器、运行时激活和历史缓存。")
for filename, rows in [('summary.csv', summary), ('classes.csv', classes)]:
    with (root / filename).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
(root / 'PRIMARY_CONCLUSION.json').write_text(json.dumps(dict(datasets=primary, stronger_positive_evidence=strong), indent=2)+'\n')
(root / 'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
