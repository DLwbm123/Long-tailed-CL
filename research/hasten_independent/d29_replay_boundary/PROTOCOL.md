# D29：Gaussian生成类与NCM边界的TRAIN统计诊断

R28修复精度后固定GELU未通过，NIH初始总体较好而最终更差，下一研究用途为增量回放机制。D20 teacher不一致不是NCM不一致，因此先检验这一必要条件，不直接训练。

只读R28 linear64 seed74002各阶段已封存TRAIN均值/协方差；每个增量阶段s>0从其前stage的旧类Gaussian统计生成每旧类固定128行，drawseed=20261024+100*dataset_index+stage（域顺序NIH/ISIC/HK），无新训练seed。比较使用当stage已到达全部类NCM均值，原replace_fc在该stage训练前已计算它们，因此不含未来类别；协方差原1e-5I不变，Gaussian输入不归一化。

记录原native cosine-NCM生成类一致性。对越界样本g，沿g→所属类原始均值m_c的线段，解析计算进入其原型Voronoi半空间交集的最小t：对所有竞争类j，用单位原型差d_j，若g·d_j<0，需t>=−g·d_j/(m_c·d_j−g·d_j)，取最大；其它样本不变。无拟合/阈值/强度扫描；只做可行性投影候选审查，不宣称生成标签是实际图像真值。若有不同原型却所属均值不在本类cell则失败。FP64求t后向1方向取一次FP32 nextafter，使边界向内；若转换FP32后native argmax仍非生成类，则退回所属均值，单独完整记录fallback，不能悄悄丢弃。该线段入口不等于全空间最小欧式投影，fallback可能损害多样性，必须披露。

每域全部增量阶段、全部旧类分母保留，单独记录samples/offclass/moved/fallback/t分布和相对位移范数。原class-consistent样本逐元素不变，矫正后生成类全部一致；3域均有非零越界且总越界率至少1%时，才认为该操作有足够触发量可进入新训练协议；这仅可行性规则，不是科学效果门，不读取VAL/TEST或择域选参。若未满足则不启动该机制训练，不改门以救援。

CPU-only，无模型前向/优化/原图/骨干/VAL特征响应/TEST；只读取上述已封存TRAIN统计。预计新增<2MB，共享checkpoint只读不复制。独立600秒含重试、300MB含16MiB余量、CPU2400s/RSS6GiB/单文件64MiB，CPU4，中性入口与独立guard。退出后全正负结果公开；若可行，另立训练检验因果效果，不能把本诊断当实际性能改善。
