# R77 后的核均值归一化候选审查

本轮只读取R77已保存的全部六stage、SFT/prototypeGRPO/distributionGRPO共18份真实TRAIN支持特征。没有图像读取、模型forward、query-class打分、optimizer或VAL；没有重跑D76或任何已闭合训练。只计算class内部支持pair kernel标量，不形成效益门/试训练/选择最佳stage。

一个可能的数学扩展是把经验核均值在RKHS中单位化。对单位向量 k(x,y)=exp(20*x·y)，令Q_c=(1/n_c²) sum_ij k(f_ci,f_cj)。query映射范数平方为exp20，故其与单位化class核均值的log cosine（忽略对所有类相同的常数）对应R77 raw score加 b_c=(20-log Q_c)/2。它有明确核均值解释，不能称新的RL创新。

但这个修正不是纯几何、无类别先验作用的模块。含对角项时可精确写成：

r_c=log Q_c-20+log n_c，
b_c=0.5 log n_c-0.5 r_c，
0<=r_c<=log n_c，0<=b_c<=0.5 log n_c。

定义n_kernel=exp(20-log Q_c)，b_c=0.5 log n_kernel，对角项占Q总和的比例=n_kernel/n_c。这是核集中程度的描述量，不是独立患者数或统计有效样本量。类样本完全重复时，完整核均值及其归一化不变，count项与geometry项相互补偿；不能只看一项叫它“纯count”，也不能隐藏其count依赖。

实际支持pair计算已核上述恒等式及数值界限（容纳FP32单位范数舍入）。最终stage结果：

|端点|b最小|b最大|b类间标准差|geometry项标准差|b与log n相关系数|
|---|---:|---:|---:|---:|---:|
|sft_reference|0.531037|3.023732|0.699867|0.319528|0.935586|
|prototype_grpo|0.541725|3.035308|0.715659|0.380258|0.898526|
|distribution_grpo|0.544413|3.021060|0.711686|0.275964|0.954978|

同stage纯0.5logn项标准差0.857342。三种端点的修正均与classcount强相关，但geometry项也非零，不能断言完全等同一个count prior。SFT最大对角份额约0.997696，distributionGRPO约0.992613，说明部分小类的该norm估计几乎由自身配对项支撑；不是已证临床罕类结构。全stage/class标量保存在KERNEL_MEAN_NORM.json/SUMMARY.json，不挑最终stage当效果结论，不计算显著性或因果解释。

决策：目前不直接冻结/启动“原kernel对归一化kernel”两arm性能试验。那样无法区分类别count项与geometry项，可能退化成对已测试prior路径的隐蔽再加权；R68等完整负结果和R77原SFT最终/尾类代价均须保留。若以后有独立依据提出此机制，至少须事前明确count-only对照、相同SFT读出控制、同目标exact及额外计算预算，不能把某个归一化公式当作已经解决原因。也不因本审查改用去对角/带宽/强度来救援；本次没有候选训练或参数选择。

此次证据只界定一个未采用候选的混杂，不否定全部核方法或未来表示学习机制。持续授权有效，当前无新FIRST_ACTUAL/新GPU实验。下一路线应有不依赖小分母VAL增量、也不只是类score偏移/权重/融合的直接主线依据。

只读pair分析实测wall2.117822s、进程CPU6.858435s、峰RSS475574272B；自设CPU600s/RSS4GB/file8MiB，沿原R77未重置deadline约束，CUDA不可见。该开销独立记录，不篡改已封存CONTROL_EXIT或冒称已计入原run费用。类内pair kernel运算不是新增模型/验证预测；原支持特征/权重/私有moments保持原样。
