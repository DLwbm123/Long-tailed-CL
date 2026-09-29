# DFD-T4-P-v1：P0 资产准入结果

**状态：BLOCKED_ASSET。P1/P2 未启动；NEXT_DECISION=STOP。**

这不是方法的科学负结果。本轮仅核查 beta10 前缀条件下的 Task4 方向 FD 试验资产，未训练或评价 D/R。

## 已完成

- 原计划字节 SHA256 与 prompt 一致：`0e91b213cae6d9483622e9c08a88e3b9bb05377ed32af58b298baa9095b1ae0d`。
- 固定提交中的 CT9 协议、执行代码、谱系，以及 CT4、CT10 路由更正、CT11 结论已读取；未重跑已有实验。
- 六个 CT3-P beta10 Task3 父文件 SHA 均匹配 CT9 父谱系，CPU 反序列化、阶段/顺序元数据通过。**这不等于完整模型 ordinary restore 通过**，后者尚未运行。
- 共享 AugReg 文件 SHA 与父记录一致；train/val manifest 字节 SHA 一致。仅检查 Task4 fit 路径存在和 val component 元数据，图像读取为零。
- 父内嵌 T bank 的 S/mu/v/e 有限，mu 为1536×6，space_version=3，n 与锁定训练计数一致。六例方向秩均为5，随机方向由指定局部PCG64生成，FP32正交误差均≤1e-5。Q文件保留在私有目录。
- CPU 合成检查通过：正交、两种损失式、空Q/满Q、同系数回归、解析梯度、同谱、公共平移跨度不变、确定性随机方向、初始零漂移与非法bank拒绝。

## 实际布局与步数

原入口 batch48、drop_last=False、10epoch；只读 manifest 计数重算。

|数据|seed|Task4 fit数|每arm steps|Q秩|
|---|---:|---:|---:|---:|
|HK|1993|667|140|5|
|HK|1994|29|10|5|
|HK|1995|570|120|5|
|ISIC|1993|6098|1280|5|
|ISIC|1994|2879|600|5|
|ISIC|1995|2862|600|5|

单arm六例共2750 steps，D/R共5500，符合计划；未执行这些更新。

## 阻塞项

1. **18份历史 Task4 sealed score NPZ 未找到**：HK/ISIC×3seed 的 CT5 F1、CT6 C3、CT9 B1T。公开锁和逐类表存在，但不能替代样本级分数、身份与component配对。具体文件名及原SHA见 `CONTROL_COMPATIBILITY.json`。
2. CT3 Task3 原始 C2/C3 bank/readout NPZ 引用文件未找到，共12份。父checkpoint内的T bank存在；缺少的是 CT9 ordinary fork 用来复核原W的独立锁定参考文件，未将它们混为一谈。
3. 当前CT9基线与父源码绑定只有历史传输服务 `tools/serve_ct3p.py` 有差异；该项须在恢复时使用原绑定源码，不得修改父元数据绕过检查。已在本地Git对象 `d1eaa599a23679d025659a6d1b45a43996291b86` 找到原字节并核对SHA，因此这一源码差异可恢复；当前决定性阻塞仍是私有历史资产缺失。

检索覆盖当前获准 my-gpu 的两个项目资产树、已知 jiangsuiyang 备份，以及本地历史工作区/已知临时路径。NAS的旧ct3p目录为空。只说明这些位置未找到，不宣称所有其他备份永久丢失。旧租期服务器在交接中已记为不可访问，本轮未连接该已结束租期主机。

## 未执行与资源

工程optimizer steps=0，正式steps=0，epoch=0，新checkpoint=0，GPU驻留=0，图像/新val预测/test/reserved访问=0。成功CPU资产审计用时约19.24秒；初步只读核查及首次失败导入未完整计时，不能将19.24秒称为整轮总耗时。初次访问守卫误拦numpy.testing已修复并保留工程记录。

真实小批次反传、教师不可变性、真实旧FD/完整real loss回归、模型普通恢复、新checkpoint roundtrip/拒绝测试和全矩阵吞吐/存储准入均未运行。尚未生成允许正式训练的 PROTOCOL/SOURCE/ASSET 锁，不将CPU数学检查当作P0全面通过。未创建周期监测。

## 产物与恢复条件

分支 `exp/dfd-t4-p-v1`；独立工作区 `/Users/bominwang/Desktop/codes/_worktrees/dfd-t4`。远程输出 `/remote-home/wangbomin/LongTailedCL/dfd_t4_p_v1_20260929`；独立备份目标 `/data_nas/jiangsuiyang/LongTailedCL/dfd_t4_p_v1_20260929`，同步与复读校验结果另见交付回执。

已完成的代码仅为最小固定数学实现和CPU资产审计，不冒称完整训练器。原始图像、Q、bank、checkpoint和逐样本分数均不进入公开Git。依本轮prompt不自动push或新建共享。

要解除阻塞，需要找回原锁定的18份sealed预测及原C2/C3 Task3参考bank/readout，随后重新做完整P0准入；不能从公开均值补造，不能通过重训Task1–3、重评历史控制或替换父状态救援。30/240指标、120epoch/梯度记录及12终态均未产生。科学门M/U/S未评价。
