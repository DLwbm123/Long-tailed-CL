"""Saved-only closure of a time-rejected native attempt; no scientific finalization."""
import json
from pathlib import Path
from .core import ARMS,write


def run(cfg):
    root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    native=load('NATIVE_ADMISSION.json');cost=load('COSTS.json');exit=load('CONTROL_EXIT.json');launch=load('LAUNCH.private.json')
    assert not native['passed'] and native['original_forecast_seconds']>=native['remaining_seconds']
    assert native['single_hidden_parity']<=.05 and native['same_shape_last_head_parity']<=.05
    assert native['generic_and_reference_restored'] and native['simulated_old_role_flags_use_only_legal_first_task_images']
    assert [r['arm'] for r in native['native_training_paths']]==ARMS
    assert exit['exit_code']==1 and exit['worker']['exit_code']==1
    assert launch['original_start']==cfg['start']==1794271927.5872638
    assert launch['deadline']==cfg['deadline']==1794343927.5872638
    c=cost['counts'];expected_nonzero={'engineering_optimizer_attempts':5,'engineering_teacherforced_queries':16,'engineering_responses':80,'generation_responses':80,'generation_attempts':80,'TRAIN_encoder_rows':4}
    for name,value in c.items():assert value==expected_nonzero.get(name,0),(name,value)
    assert load('TRAINING_CURVES.json')==[] and load('EVALUATIONS.json')==[] and load('STATE_BINDINGS.json')==[]
    assert not list(root.glob('BUFFER_BIRTH_*.private.pt')) and not list(root.glob('STATE_*.private.pt'))
    assert not (root/'COMMON_PREFIX_SFT.private.pt').exists() and not (root/'FINAL_RESULTS.json').exists()
    assert all(r['split']=='TRAIN' and r['stage']==0 for r in load('ACCESS.private.json'))
    write(root/'NATIVE_TIME_DECISION.json',{'status':'NOT_TIME_ADMITTED_NO_SCIENCE','saved_only_verification_passed':True,
        'original_start':cfg['start'],'frozen_deadline':cfg['deadline'],'forecast_seconds_including_margin':native['original_forecast_seconds'],
        'remaining_seconds_at_native':native['remaining_seconds'],'shortfall_seconds':native['original_forecast_seconds']-native['remaining_seconds'],
        'all_five_real_optimizer_paths_executed':True,'all_engineering_updates_discarded':True,
        'same_hidden_parity':native['single_hidden_parity'],'same_shape_head_parity':native['same_shape_last_head_parity'],
        'raw_full_head_difference':native['raw_full_head_vs_last_head_difference'],
        'sampling_TF_max_difference':max(r.get('sampled_vs_TF_logp_max_difference',0) for r in native['native_training_paths']),
        'unupdated_TF_logp_difference':max(r.get('unupdated_behavior_logp_difference',0) for r in native['native_training_paths']),
        'native_profile_input_tokens':native['conservative_longest_prompt_profile']['input_tokens'],
        'science_SFT_updates':0,'science_post_updates':0,'science_VAL_responses':0,'true_old_buffer_accesses':0,
        'engineering_counts':c,'original_clock_and_margin_not_relaxed':True,'no_scientific_efficacy_conclusion':True,
        'future_action':'No automatic same-run retry without concrete new capacity evidence under original deadline. Do not shorten matrix, relax margin, or reset clock merely to launch.'})
    write(root/'ALL_ATTEMPT_COSTS.json',{'attempts':1,'scientific_training_started':False,'native_time_rejected_attempt':cost,
        'controller_exit':exit,'CPU_software_preflight':load('CPU_SOFTWARE_RUN_COSTS.json'),
        'full_protocol_normal_expected_generation_responses':182226,'actual_generation_responses':80,
        'actual_engineering_optimizer_attempts':5,'actual_scientific_optimizer_attempts':0,
        'historical_cache_creation_costs_not_free_or_repeated':True,'original_first_preparation_and_deadline_unchanged':True})
    lines=['# R126 原生时间准入未通过','',
        '真实原生模型的五条反向/optimizer路径均完成，matched hidden和同形末token head差均0；原始full-sequence head差0.0625、sampling/TF差0.187709保留，0.05阈值未放宽，同优化路径未更新ratio为1。工程更新全部丢弃并还原generic/reference。',
        f"冻结完整矩阵（1.5倍实测外推+1800秒预留）需{native['original_forecast_seconds']:.3f}秒，但原钟仅剩{native['remaining_seconds']:.3f}秒，不满足时间准入；worker断言退出1。该退出不是科学效果否决，也不说明数值路径失效。", 
        '共同SFT、正式RL/CE、VAL和真实旧buffer访问全部0；只有工程80响应、4TRAIN编码、5 optimizer尝试和16 teacherforced queries。没有启动科学训练，未运行要求完整成功数据的finalize_saved。',
        'CPU-only保存核对已确认全部计数/空科学结果/还原状态与原首次准备20h钟。保留原源码、原日志和全部费用；不删除历史，不延长20h或放宽外推裕量，不重置first preparation。',
        '当前五臂协议在此次实际吞吐下不可准入。旧正确奖励因果问题仍未被测量；不能把缺少训练说成机制无效或成功。当前无活跃本任务GPU实验。',
        '不为启动而删掉对照、缩短更新数、改变提示/seed/SFT/参数。后续只有具体的新资源/吞吐证据且仍满足原钟，或另有事先冻结、科学问题和费用明确的不同方案，才能重新准入；不得盲轮询/重跑native等待偶然变快。']
    (root/'ADMISSION_REPORT_ZH.md').write_text('\n\n'.join(lines)+'\n')
