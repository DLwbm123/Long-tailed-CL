# 保留完整图，不重算的执行资格

MLP checkpoint通过等价与显存但预算失败，其结果不改。剩余可消除的实际工作是MLP反向重算；无需再改变attention实现。使用原NativeEvidence逐块forward，micro8各输出拼接后计算一次完整S/Q耦合目标，保留整个图直接backward。完全沿用原attention、GELU、FP32、双dual求解器、梯度裁剪和AdamW，没有新近似或fused内核。代价是更多保留激活；是否小于原8GiB必须实测，不能凭估计启动bulk。

候选固定为retained_full_graph一项，非microbatch/精度/超参搜索。CPU和真实GPU各三臂完整梯度与实际更新对照、真实64/62(micro8末6)资格与peak、有限state独立FP64响应和旧完整class保存链组合证据不减。若GPU峰或原预算门失败就停止该执行，不重复测速赌速度。

原first1792727818.659005/deadline1792785418.659005、9400步/四臂/17门/6GB5.4GB/GPU8GiB/RSS40GiB与1.5×9400×三臂step+10800全不改。所有新资格和入口纳入原根/钟和窗口30GB，保留全部旧失败。科学bulk仍独立于工程资格，满足所有门才可启动；源码不是运行结果。
