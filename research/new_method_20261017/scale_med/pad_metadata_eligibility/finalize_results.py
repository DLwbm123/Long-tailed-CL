from pathlib import Path
import json
b=Path(__file__).resolve().parent;w=b.parent/'autonomy_20261019_12h';x=json.loads(Path('/tmp/q980_receipts.json').read_text());c=x['COORDINATOR_EXIT.json'];assert x['ps']=='' and c['exit_code']==1;r=b/'results';r.mkdir(exist_ok=True)
def save(n,v):(r/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
for n,v in x.items():
 if not n.endswith('.private.json') and n!='ps':save(n,v)
f=json.loads(Path('/tmp/q980_failure_counts.json').read_text());save('POST_FAILURE_METADATA_FORENSICS.json',dict(**f,posthoc_forensic_only=True,model_scores_used=False,no_cleaning_or_split=True))
a=dict(decision='UNRESOLVED_GLOBAL_LESION_ID_NAMESPACE',metadata_download_complete=True,metadata_bytes=316209,reported_rows=2298,reported_patients=1373,observed_global_lesion_ids=1641,observed_patient_lesion_pairs=1891,integrity_gate_passed=False,patient_split_executed=False,external_validation_complete=False,method_success=False,actual_fits=0,actual_solves=0,images_downloaded=0,query_scored=False,source_commit='bd23478c226d9351ed5b0c88a559717bf8b7a0b3',no_auto_retry=True);save('aggregate.json',a)
cost=dict(activity_wall_seconds=c['wall_seconds'],known_worker_wait4_CPU_seconds=c['worker_CPU_seconds'],coordinator_CPU_seconds=c['coordinator_CPU_seconds'],peak_RSS_bytes=c['worker_peak_RSS_bytes'],round_remote_bytes=c['round_bytes'],round_with16MiB_reserve=c['round_bytes']+16*2**20,window_remote_bytes=c['window_bytes'],window_total_conservative=c['window_bytes']+1001040384,unmetered='Reading/network preparation/postfailure metadata forensic/report CPU unmetered; body worker not added twice.');save('COST_RECEIPT.json',cost)
s=json.loads((w/'STATE.private.json').read_text());total=dict(activity_wall_seconds=s['current_activity_wall_total']+cost['activity_wall_seconds'],known_worker_wait4_CPU_seconds=s['known_worker_CPU_total']+cost['known_worker_wait4_CPU_seconds'],coordinator_CPU_seconds=s['quantified_coordinator_CPU_total']+cost['coordinator_CPU_seconds']);save('CUMULATIVE_COSTS.json',total)
save('VERIFICATION.json',dict(download_HTTP=200,download_bytes_exact=True,expected_rows_patients_global_lesion_counts_passed=True,unique_image_nonempty_required_fields_and6labels_passed=True,global_lesion_single_patient_single_label_failed=True,split_not_reached=True,phase_exit1_honestly_reported=True,no_resource_guard=True,retired_pids=list(x['LAUNCH.private.json'].values()),active_neutral_argv_captured=True,firstactual=x['FIRST_ACTUAL.private.json']['start_unix'],scientific_complete=False))
t=f'''# PAD-UFES-20外部验证准备：身份命名空间未决

用户建议用其它数据集验证，已开展独立候选核查，并在读取元数据前封存协议。候选为PAD-UFES-20 v1，DOI10.17632/zr7vgbcyr2.1，官方CC BY4.0；有患者与病灶标识，六类手机临床图像。它能补充皮肤镜ISIC的患者分组和成像域，但不是原四类标签的同义复制；原class1皮肤纤维瘤不在此数据中，后续MEL只能是明确的外部tail sentinel。

## 实际完成与停止

科学源/协议bd23478c226d9351ed5b0c88a559717bf8b7a0b3先push、核ref与匿名PROTOCOL200后才actual。NFS仅下载官方metadata.csv，HTTP200、316209bytes匹配。2298行/1373患者/1641全局lesion_id，非空标识、唯一img_id、六类标签检查通过；但预设“一个lesion_id对应一个患者和诊断”断言失败。worker与协调均exit1；不是资源超限、模型性能负结论或下载缺失。

**UNRESOLVED_GLOBAL_LESION_ID_NAMESPACE**。患者两折划分尚未执行，所有per-fold资格/少数类风险为NA，没有图像下载、encoder、模型fit/solve/GPU/评分/bootstrap。原停止门不放宽、不改源码重跑、不删除冲突行凑通过。

## 失败后的只读定位

元数据仍留在NFS。为定位失败，仅在remote对已下载字段做聚合，未返回身份或逐例：250个全局lesion_id跨患者，226个跨诊断。使用(patient_id,lesion_id)得1891个组合，组合内诊断冲突0。该结果提示病灶ID可能具有患者命名空间；尚未认证1891个组合就是独立真实病灶，不能偷偷把1641改成1891后宣称原协议通过。

作者公开sanity_check检查缺失、图像路径、合法诊断等，但所查脚本没有认证lesion_id全局唯一。它不是已执行清洗或独立临床身份认证；本轮没有运行第三方代码。

下一步先核官方标识生成/命名空间依据及跨患者合并含义，再另封不同身份验证协议。保留本轮失败，禁止自动重试、换数据集性能挑选或患者/病灶混称。尚未验证外部病灶内预测变化机制，更未测试删项、GLS或新头。新数据的患者隔离、少数类/全类/两fold门、原冻结encoder和F强对照均须在任何编码前封存。图像官方总下载约3.35GB，archive+unpack+cache峰值须另预算，不能塞入旧17/20GB窗口或忽略副本成本。

## 成本与交付

firstactual={x['FIRST_ACTUAL.private.json']['start_unix']}，原600s/CPU300s/RSS4GiB/file16MiB门未触发；wall{cost['activity_wall_seconds']:.6f}s/wait4CPU{cost['known_worker_wait4_CPU_seconds']:.6f}s/coord{cost['coordinator_CPU_seconds']:.6f}s一次并账。RSS{cost['peak_RSS_bytes']}；round{cost['round_remote_bytes']}+16MiB<128MiB，window{cost['window_remote_bytes']}+1001040384+16MiB<17/20GB。正文worker CPU已含，不重加；准备/读取/网络/事后元数据整理成本未知，累计见CUMULATIVE_COSTS。

父1095201/子1095202 exit1退休，活动中性argv已捕获、完成ps为空，不再监测/启动。NFS4容量与写读probe过，无旧资产删除/像素模型环境复制。CSV及patient/lesion/img ID只在remote私有存储；split没有生成。公开仅协议源码、聚合异常/退出/NA/成本，DELIVERY工程交付可complete而scientific_complete=false，不能写外部验证完成。

来源：[官方数据页](https://data.mendeley.com/datasets/zr7vgbcyr2/1)，[原始数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/)，[作者sanity_check](https://github.com/labcin-ufes/PAD-UFES-20/blob/master/sanity_check/sanity_check.py)。官方许可与样本说明不认证本项目pretraining无重叠；本地有界搜索未见PAD旧实验不是全历史证明。
'''
for n in ['ANALYSIS_ZH.md','REPORT_ZH.md']:(r/n).write_text(t)
print(json.dumps(dict(cost=cost,total=total)))
