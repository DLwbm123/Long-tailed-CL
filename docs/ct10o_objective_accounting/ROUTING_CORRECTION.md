# 路由说明更正

发现时点：准备尚未启动的固定读出对照。

原错误：将“解析评价batch48”写成沿用batchwise路由，并在监测提示中暗示FD与解析评价路由不同。

证据：tools/run_ct1.py:164–166创建独立eval probe、安装既有batched linear，并将main/few pool.batchwise_prompt设为False；:110–111仅load_state_dict和eval，不重置此普通Python布尔属性。tools/run_ct3p.py:118–120继承该构造器；CT6/CT9没有再次修改probe路由。tools/ct3p_core.py:17–30在FD上下文中同样将两pool.batchwise_prompt=False，随后恢复原状态。

正确解释：原CE训练网络的配置为True；FD与解析probe都使用逐样本路由。batch48与逐样本路由可同时成立。神经训练头与ridge不共享分类器目标的结论仍成立，但不能用FD/probe之间的批次路由差别解释结果。

影响：文字及监测提示更正。既有模型、预测、表格、锁和数值不变。新forward/训练/预测/test=0，无需重跑。原报告Git历史保留。
