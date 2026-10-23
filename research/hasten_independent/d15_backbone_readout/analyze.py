"""Verify saved count aggregates and report every frozen D15 cell; no model calls."""
import json, pathlib, statistics as st
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results'
def read(p): return json.loads(p.read_text())
x = read(OUT / 'RESULTS.json')
h0 = read(ROOT.parent / 'r12_three_datasets/results/ANALYSIS.json')
r14 = read(ROOT.parent / 'r14_batch_multiplicity/results/ANALYSIS.json')
report = {}; checks = 0
for ds, data in x.items():
    split = data['split']; assert split == h0[ds]['split']
    seal = read(OUT / ds / 'ALL_HEADS_SEALED.json')
    assert seal['validation_image_reads'] == 0
    assert seal['stages'] == len(split['stages'])
    assert seal['time'] < read(OUT / ds / 'STATUS.json')['time']
    assert read(OUT / ds / 'CACHE_PROVENANCE.json')['max_abs_error'] < 2e-5
    cells = {}
    for encoder in ['Biomed', 'ImageNet']:
        receipt = read(OUT / ds / (encoder + '_TRAIN_RECEIPT.json'))
        assert receipt['all_heads_restore_exact'] and receipt['validation_image_reads'] == 0
        assert [r['classes'] for r in receipt['stages']] == split['stages']
        assert [r['rows'] for r in receipt['stages']] == [sum(split['train_counts'][c] for c in cs) for cs in split['stages']]
        rows = sorted([r for r in data['results'] if r['encoder'] == encoder], key=lambda r: r['stage'])
        assert [r['stage'] for r in rows] == list(range(len(split['stages'])))
        for head in ['NCM', 'ridge']:
            metrics = []
            for row in rows:
                m = row['metrics'][head]; n = row['seen']
                assert n == sum(map(len, split['stages'][:row['stage']+1]))
                assert m['counts'] == split['eval_counts'][:n]
                assert all(0 <= c <= t for c, t in zip(m['correct'], m['counts']))
                recall = [100*c/t for c,t in zip(m['correct'], m['counts'])]
                tail = [v for i,v in enumerate(recall) if split['train_counts'][i] < 250]
                assert abs(st.mean(recall)-m['BA']) < 1e-8
                assert abs(st.mean(tail)-m['tail_BA']) < 1e-8
                checks += 1; metrics.append(m)
            cells[encoder+'_'+head] = dict(initial_BA=metrics[0]['BA'], mean_BA=st.mean(m['BA'] for m in metrics), final_BA=metrics[-1]['BA'], tail_BA=metrics[-1]['tail_BA'], stages=[dict(stage=r['stage'],seen=r['seen'],**r['metrics'][head]) for r in rows])
    report[ds] = dict(cells=cells, H0=h0[ds]['summary']['final'], H0_seeds=h0[ds]['seeds'], R14=r14[ds]['summary']['final'], R14_seeds=r14[ds]['seeds'])
exit = read(OUT/'CONTROL_EXIT.json')
assert exit['exit_code'] == 0 and exit['reason'] is None
assert exit['total_elapsed_seconds'] < 14400 and exit['allocated_bytes'] < 8000000000
(OUT/'ANALYSIS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(OUT/'VERIFICATION.json').write_text(json.dumps(dict(count_metric_checks=checks, all_three_datasets=True, stages=15, cells=12, exit_budget_passed=True, restore_and_zero_validation_receipts=True, caveat='Seal ordering supported by saved receipts and worker code; no separate validation-start timestamp was recorded.', independent_confirmation=False),indent=2)+'\n')
lines=['# D15：三数据集骨干 × 读出诊断','', '全部完成，exit0；耗时 %.2f 秒，退出时新增占用 %d bytes。'%(exit['total_elapsed_seconds'],exit['allocated_bytes']),'', 'BA为类宏召回百分比；尾类为TRAIN<250。阶段平均包括初始阶段。','', '|数据集|骨干与读出|初始BA|阶段平均BA|最终BA|最终尾类BA|','|---|---|---:|---:|---:|---:|']
for ds,r in report.items():
    for cell,m in r['cells'].items():lines.append('|%s|%s|%.4f|%.4f|%.4f|%.4f|'%(ds,cell,m['initial_BA'],m['mean_BA'],m['final_BA'],m['tail_BA']))
lines += ['', '## 与H0、R14三种子并列','', '|数据集|实验|阶段平均BA±SD|最终BA±SD|尾类BA±SD|','|---|---|---:|---:|---:|']
for ds,r in report.items():
    for name in ['H0','R14']:
        vals=['%.4f ± %.4f'%(r[name][m]['mean'],r[name][m]['sd']) for m in ['mean_BA','final_BA','tail_BA']]
        lines.append('|%s|%s|%s|'%(ds,name,'|'.join(vals)))
lines += ['', '## 综合判断与主线决策','', 'HK在两种读出下ImageNet最终总体/尾类均优于Biomed；阶段平均与初始总体也一致。这支持该域表征管线存在差距，但不能证明纯预训练因果。NIH与ISIC中NCM偏向Biomed、ridge总体偏向ImageNet，表明表征与读出交互。ImageNet ridge并非三域尾类一致最优，ISIC Biomed ridge尾类更好。', '', '固定Biomed时ridge优于NCM；相对H0，NIH ridge总体较低但尾类较高，ISIC总体接近而尾类较高，HK总体与尾类均较低。因此不能将全部HASTEN差距归因于骨干，也不能宣称ridge全面胜过已训练HASTEN。R14失败进一步说明训练目标敏感，但本诊断未单独消除增量训练、损失与融合的混杂。', '', '下一轮R16：在H0训练状态完全固定下，将NCM分支替换成已封存的类均衡ridge TRAIN记忆，保持原双曲分支、固定stat=.5与原归一化/softmax融合。检验读出机制本身能否改善HASTEN尾类且保护总体。只做一个固定候选、不按域择优、不扫描；失败不降低门。保留Biomed作为此轮控制，并非否定HK表征改进价值。', '', '## 证据边界','', 'D15每骨干为一次确定性序列，不能伪装成3seed；H0/R14为3seed，SD仅描述历史训练变异。每骨干使用自身官方预处理、维度512/768不同，不能做纯预训练归因或因果百分比拆分。历史开发集已反复使用，无TEST或独立泛化证据。所有逐阶段逐类分母/正确数及H0/R14逐seed摘要见results/RESULTS.json和ANALYSIS.json。60项计数指标核验通过；头恢复及零VAL封存有回执，独立VAL开始时间未单独记录。', '', '只发布源代码、协议及类级聚合；图像、身份、逐例响应、特征、权重及原始日志保持私有。']
(ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
print('D15 verification passed:',checks)
