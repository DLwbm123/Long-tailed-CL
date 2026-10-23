# 原生完整图执行资格通过

未改attention、目标、9400步、四臂与17门，去除完整body cache/VJP重算与MLP checkpoint重算，保留micro8全S/Q图。CPU小图与真实GPU完整12层三臂目标/梯度/实际AdamW均逐项通过，真实64/62和micro8末6各臂2步完整完成，有限state保存加载及独立FP64/NumPy响应通过。完整类矩保存/8+2链仍明确组合使用repair2旧PASS；不声称本次重跑所有类保存。

实际三臂配对步2.884269476秒；冻结公式1.5×9400×step+10800=51468.199611秒，小于当时原剩余54316.800144秒（仍需bulk启动时重查）。GPU完整图峰6,796,869,632B<8GiB。先前两执行预算失败均保留；新结果不是重新测同一实现，不将旧FAIL改PASS。

原first1792727818.659005/deadline1792785418.659005、6GB/5.4GB/RSS40GiB/GPU8GiB及窗口30GB不重置。新78prefix、18丢弃AdamW(CPU6/GPUparity6/GPU62-64六步)，比较forward/back计数CPU48/GPU48；timed378全body前反向、0重算；teacher78与有限GPU/FP64/NumPy各6响应另列。资格无科学编码/val/正式head或性能。原q680全组已退，辅助驻留含加载/证明，缺失分段未知不记0。独立guard采样root+entries1,531,185,161B/RSS2,590,187,520B，不作连续峰保证。

full_graph_run.py保留原科学逻辑，跳过已经封存的GPU资格副本，组合读取旧完整class链和本次变更执行证据，原剩余预算复查后才bulk。为保持原prompt随机初始化的随机数状态，保留原CPU数学检查（新增6丢弃CPU更新单列）；不重复GPU资格。full_graph_verify.py保留24系统/全12层NumPy、患者隔离、完整paired draws与原科学验收，按新执行明确检查forward/back实际行和零cache/重算计数。所有历次费用合并交付；原截止不足则如实部分结束。
