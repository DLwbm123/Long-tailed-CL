"""Audit existing anonymous PMT aggregates and render all-sign supplementary tables."""
import csv
import json
from pathlib import Path
import sys


def analyze(round_dir):
    r=Path(round_dir);out=r/'results';s=json.loads((out/'RESULTS.json').read_text());lock=json.loads((r/'PROTOCOL_LOCK.json').read_text())
    new=('pmt','local_transport_pmt','local_transport_pmt_global_only');methods=('base','local_transport','AIR',*new)
    final=[];stage=[];conf=[];gates=[]
    for ds in ('ISIC','HK'):
        for arm in methods:
            for t in s['runs'][ds+'_'+arm]['metrics']['stages']:
                stage.append(dict(dataset=ds,method=arm,stage=t['task'],n=t['validation_n'],BA_percent=100*t['balanced_accuracy'],Acc_percent=100*t['accuracy']))
        for label in lock['order'][ds]:
            row=dict(dataset=ds,label=label,train_n=lock['data'][ds]['train_counts'][str(label)],val_n=lock['data'][ds]['val_counts'][str(label)],role='new' if label in lock['order'][ds][-2:] else 'old')
            for arm in methods:
                t=s['runs'][ds+'_'+arm]['metrics']['stages'][-1];row[arm+'_correct']=round(row['val_n']*t['per_class_recall'][str(label)])
                if arm in new:
                    paired=next(a for a in t['paired']['classes'] if a['label']==label)
                    assert paired['correct_on']==row[arm+'_correct']
                    for key in ('rescued','damaged','true_margin_mean_on','true_margin_mean_control'):row[arm+'_'+key]=paired[key]
            final.append(row)
        for arm in new:
            m=s['runs'][ds+'_'+arm]['metrics'];control='base' if arm=='pmt' else 'local_transport';c=s['runs'][ds+'_'+control]['metrics']
            fields=('final_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall');d={k:100*(m[k]-c[k]) for k in fields}
            checks=dict(BA=d[fields[0]]>=1,tail=d[fields[1]]>=-.5,forget=d[fields[2]]<=1,old=d[fields[3]]>=-1,new=d[fields[4]]>=-1)
            gates.append(dict(dataset=ds,method=arm,control=control,delta_pp=d,gates=checks,all_passed=all(checks.values())))
            for t in m['stages']:
                old=len(t['seen'])-lock['task_sizes'][ds][t['task']-1]
                for name,cm in t['paired']['confusion'].items():
                    conf.append(dict(dataset=ds,method=arm,stage=t['task'],state=name,
                        old_to_old_wrong=sum(cm[i][j] for i in range(old) for j in range(old) if i!=j),old_to_new=sum(sum(cm[i][old:]) for i in range(old)),
                        new_to_old=sum(sum(cm[i][:old]) for i in range(old,len(cm))),new_to_new_wrong=sum(cm[i][j] for i in range(old,len(cm)) for j in range(old,len(cm)) if i!=j)))
    for name,rows in [('FINAL_CLASS_COMPARISON.csv',final),('STAGE_SUMMARY.csv',stage),('CONFUSION_GROUPS.csv',conf)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'DEVELOPMENT_GATES.json').write_text(json.dumps(gates,indent=2)+'\n')
    lines=['# PMT 结果分析：旧类改善与新类代价','','固定主方法为LT。P=Core+PMT，F=LT+global/regional PMT，G=LT+global PMT（原区域平移）。全部六个自身控制开发门均未全过，不能称主方法取得可靠净提升。',
        '', 'ISIC：P−Core BA+1.375046pp、tail+9.429933pp、forget−6.286622pp、old+5.924303pp；new−12.272727pp。F−LT BA+.768540pp、tail+7.402906pp、forget−5.111534pp、old+4.661083pp；new−10.909091pp。降低遗忘和提高旧类召回有真实配对计数支持，但代价明确违反新类保护门。',
        '', 'HK：P−Core BA−.100031pp；F/G−LT BA+.054924pp，两组合全stage分类recall完全相同，区域映射没有可观测分类增益。新类均−.645269pp，两类各损失1例；极少旧类5/6/19/21仍无救回，未解决稀少旧类内部混淆。',
        '', '区域贡献：ISIC F−G BA+.337838pp，对应最终旧类5少破坏1例，新类计数不变；HK F/G全stage类别recall相同，只有分数/间隔微小变化，不能当净效用。ISIC主交互BA−.606506pp，HK+.154955pp，不能仅凭组合绝对成绩声称全面互补。',
        '', 'AIR：ISIC P把差距3.047766缩至1.672720pp，F把LT差距1.757783缩至.989243pp；均伴随显著新类下降。HK P差距扩大到1.585998pp，F/G仅缩至1.417739pp；两数据集均未超过AIR，训练曝光/解析批数/资源不同，不能称等预算胜负。',
        '', '## 最终逐类：全部正负及支持数','','Core/LT/AIR与P/F/G列为正确数；分母见n，精确recall=正确数/n。R/D为相对各自控制的救回/破坏。',
        '', '|数据|类|train n|val n|角色|Core/LT/AIR正确|P/F/G正确|P R/D|F R/D|G R/D|','|---|---:|---:|---:|---|---|---|---|---|---|']
    for row in final:
        lines.append('|'+ '|'.join([row['dataset'],str(row['label']),str(row['train_n']),str(row['val_n']),row['role'],'/'.join(str(row[a+'_correct']) for a in methods[:3]),'/'.join(str(row[a+'_correct']) for a in new),*[str(row[a+'_rescued'])+'/'+str(row[a+'_damaged']) for a in new]])+'|')
    lines+=['','## 全stage BA/Acc（%）','','|数据|stage|Core|LT|P|F|G|AIR|','|---|---:|---|---|---|---|---|---|']
    for ds in ('ISIC','HK'):
        for t in range(1,len(lock['task_sizes'][ds])+1):
            values=[]
            for arm in ('base','local_transport',*new,'AIR'):
                v=next(a for a in stage if a['dataset']==ds and a['method']==arm and a['stage']==t);values.append(f"{v['BA_percent']:.6f}/{v['Acc_percent']:.6f}")
            lines.append('|'+ds+'|'+str(t)+'|'+'|'.join(values)+'|')
    lines+=['','## 可确定的混淆变化及不能确定的原因','',
        'ISIC P：旧→新错误53→33，但旧→旧错误55→62，新→旧错误15→26。F：旧→新51→34，旧→旧53→60，新→旧16→25。G：旧→新51→34，旧→旧53→61，新→旧16→25。机制主要改变旧新竞争；“当前新类统计没有运输”不保证新类分类边界不变。',
        'HK P旧→旧错误221→222、旧→新27→26、新→旧25→27；F/G对应222→220、27→26、25→27。新类8对旧类19的误判净增1，类3对旧类4净增1；类19自身仍0/2。这里只是匿名聚合混淆变化，不能从两张边际混淆矩阵捏造个体反事实因果。',
        'ISIC P最终旧类17救回/4破坏、新类0救回/10破坏；F为14/4、0/9；G为14/5、0/9。HK P旧5/5、新0/2；F/G旧5/2、新0/2。ISIC平均新类margin分别由.131117降至.081534（P）、.127550至.082078（F）、.127550至.081960（G）；HK也均下降。完整每类margin与所有stage配对计数见PAIRED_CORRECTIONS。',
        '', '## 映射证据与假设边界','',
        'ISIC后两task boundary global输入rank1536，最终P/F/G线性变化Frobenius约3.984/3.380/3.399；HK小task rank116、47，fit配对残差很小，但不能识别旧域外推误差。最终旧component半径上界最大ISIC约14.692/8.975/9.125，HK约4.464/4.440/4.440。这些是保守上界，不是实际协方差膨胀估计；不能据此断言某旧类真实漂移。',
        '精确矩pushforward、解析求解和checkpoint重建通过，支持实现了冻结的运输机制；不能证明旧域仿射假设正确，也不能单凭本轮数据完全排除机制设计失配。新类统计原样加入但解析头的共享G、competition及训练梯度仍会改变所有列。现有结果没有把bank运输与编码器轨迹的贡献分离，不能宣称已确定唯一原因。',
        '', '## 下一步：先CPU证伪，不按val挑类或调强度','',
        '保留Core与LT。本轮P/F/G不是新主方法，不机械放大/缩小ridge或扫rank。下一步仅独立CPU研究：用固定数据生成的反例核对“旧矩精确运输、当前新类统计不动仍损害新类边界”；再检验只由current-fit真实标签和同状态OFF头定义的逐样本边界约束是否存在非identity可行更新，并检查约束活跃时的Lagrangian/envelope梯度。全fit约束只保护已观察训练支持，不保证val或旧旧类；旧域推广不可识别仍须明确。',
        '不使用此次val混淆挑类别对/阈值，不新增旧图、seed、预算档位或GPU轮。高维完整native算子/可行性/关闭精确/原梯度与checkpoint未验收前，不称GPU-ready；若约束在非退化fixture也只能identity，先报告失败而不启动另一轮。',
        '', '单seed/order与反复development val，非test、患者独立或临床证据。全stage逐类n/recall共756行、配对计数378行；完整正零负、代价、六个开发门和18次驻留记录一并公开。']
    (out/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
    return dict(status='PASS',class_rows=len(final),stage_rows=len(stage),paired_stage_confusions=len(conf),development_gates_passed=sum(x['all_passed'] for x in gates),additional_inference=False)


if __name__=='__main__':print(json.dumps(analyze(sys.argv[1])))
