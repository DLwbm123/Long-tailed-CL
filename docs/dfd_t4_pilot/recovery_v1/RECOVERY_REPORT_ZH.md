# DFD-P0R-v1：有限资产恢复报告

**最终状态：BLOCKED_UPSTREAM_STATE；formal_training_started=false；NEXT_DECISION=STOP。**

原 `9a35764aeebe173593272cab9407cb3edb2aca0a` 中的 BLOCKED_ASSET 报告与审计保持原样。本轮只恢复资产，没有D/R工程或正式训练，也不代表完整P0通过。

## 30项恢复结果

|对象|原缺项|本轮结果|证据边界|
|---|---:|---|---|
|Task3 C2/C3参考W|12|全部 REBUILT_FROM_VERIFIED_PARENT|从原父合法A/T bank按原ridge派生；独立历史W匹配 NOT_AVAILABLE|
|CT6 C3 Task4控制预测|6|全部 REBUILT_FIXED_CONTROL|固定原Task4模型/合法T bank/原val；逐类离散计数回归通过|
|CT9 B1T Task4控制预测|6|全部 REBUILT_FIXED_CONTROL|固定原Task4模型/合法T bank/原val；逐类离散计数回归通过|
|CT5 F1 Task4控制预测|6|全部 UNRECOVERABLE|未定位原Task4 W或合法累计八类bank，不能用Task1两类bank代替|

按历史SHA找回原始30项文件数为0，派生或固定重评恢复24项，剩余6项。未找到不等于永久丢失。完整逐项旧SHA、新SHA和缺口见 `RECOVERY_INVENTORY.json`。

## 先保护父状态与源码

已检查已知独立备份与项目备份索引，未找到六父独立副本。本轮新增6份只读父checkpoint副本，共 1,002,762,434 字节；独立端逐文件复读SHA均与原历史谱系一致。原有P0 DELIVERY_RECEIPT不覆盖父状态，本轮未沿用该回执作父备份证明。

从锁定Git对象恢复 `serve_ct3p.py` 原字节，完整核验父绑定的137份源码文件。源checkpoint元数据、原权重、manifest、源参数和原协议均未修改。原权重路径通过指向既有已核验文件的兼容符号链接恢复，没有另复制权重。恢复适配入口单独绑定SHA。

6个Task3父和12个CT6/CT9 Task4末状态均调用原ordinary restore，严格state_dict、完整网络指纹、原args/source/shared-weight/manifest绑定，以及RNG/loader/synthesis状态检查全部通过。这与外部历史W回归是两项检查：前者通过，后者因原独立W缺失仍为NOT_AVAILABLE。FZ1的6个Task1模型文件SHA与历史锁一致，但因其Task4读出不准入，未做该控制的完整模型恢复或前向。

## Task3读出与固定控制重建

已分别检查A和T的S/mu/v/e/n/arrival/space_version/类别映射及有限性。共24次原float64 ridge求解：12次用于Task3参考W，12次用于CT6/CT9 Task4 T-W。固定lambda=0.001，没有jitter或重新拟合统计。原实现残差门槛保持1e-10；独立未归一化法方程残差最大约 1.257e-15。求解前后bank指纹不变。未把重新求出的W与自身比较来冒充历史W匹配。

12个可重建控制的模型、W来源、原val身份/component/列顺序、源码及精度/路由先统一锁定，再每控制执行一次全量val。沿用原提取入口、batch48、4个worker、shuffle=False、drop_last=False、pointwise probe，关闭AMP/TF32；全程无optimizer和autograd训练。

12个控制共96条逐类记录的原标签、n_images、n_correct、n_components与历史公开离散记录逐项一致。模型、bank、读出文件及RNG核验通过。新预测有真实新SHA，**历史逐样本一致性仍为UNKNOWN**；计数一致不能证明逐样本分数或原NPZ字节相同。这是恢复用固定重评，不是新的独立验证，不产生D/R科学门结论。

## 精确剩余缺项

HK和ISIC，各seed 1993/1994/1995，均缺CT5冻结Task1编码器累计至Task4的原读出：`{dataset}_{seed}_t04.npz`；相应缺失预测为 `{dataset}_{seed}_F1_t04.npz`。每个预期W与预测SHA见 `FZ1_UPSTREAM_AUDIT.json` 和逐项清单。

已定位的原Task1模型仅携带2类bank，所需Task4为8类；没有合法累计统计就不能重建。未读取旧fit补算，没有减掉FZ1控制、替换模型或启动D/R。要继续需找回这6个原Task4 W/合法完整bank，或匹配历史SHA的6份原sealed预测，并完成相应来源与布局验收。

## 有界溯源、资源与修复

补查了归档索引、3个历史源码包的成员目录、CT11诊断模型/读出索引及后续项目备份报告；包内未解包图像、旧fit特征或禁止资产。依据CT5状态锁另核对了其W文件名，它们不同于F1预测文件名。未连接已结束租期主机，未扩大权限或租用资源。搜索保守计时上界约 466.5 秒，含部分非搜索准备，低于30分钟。

- D/R工程steps=0、正式steps=0、新增训练checkpoint=0。
- 固定控制预测12单元，val图像读取5464次；旧fit、未来fit、test/reserved、oracle读取均0。
- 恢复工作进程计费时长 674.48 秒，约 11.24 分钟，含该进程CPU求解和恢复时间，低于1小时。
- 运行期新增产物与独立备份合计约 1.007 GiB，低于3GiB；该数含运行文件，不含随后少量报告文字。两端空闲均超过1GiB。

首次启动在旧源码导入链缺sklearn处终止，未构造模型/评价。已在本轮独立依赖目录安装scikit-learn1.5.2、joblib1.4.2，不改共享环境或旧源码；保留失败日志与 `ENGINEERING_REPAIR_01.json`。没有数值回归失败、放宽容忍、重复评价择优或方法修订。

## 备份与停止

新私有资产已持久化：24份重建W和12份sealed预测全部在独立端复读SHA通过；另核验 156 份源代码/配置/审计文件。六父备份另有逐文件回执。依赖安装目录和pip缓存可再生成，不作为科学资产备份。认证信息、NAS路径、样本身份、图像、W和checkpoint不纳入本次公开材料。

当前不是READY_FOR_FULL_P0。即使后续补齐FZ1，仍需继续原P0的真实梯度、teacher、checkpoint roundtrip与资源准入；本轮不执行这些D/R工程或5500步训练。新报告本地提交，本补充授权未扩展GitHub推送范围，未自动推送。
