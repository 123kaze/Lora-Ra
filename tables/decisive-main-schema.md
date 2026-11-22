# 第0步正式结果数据契约

| 文件 | 行单位 | 聚合与边界 |
|---|---|---|
| main.csv | 宽度/新基础/adapter/预算/方法/定步数 | 1440行，含等价Width别名；不是1440个独立实验 |
| conditions.csv | 逻辑训练条件 | 720条件，至多640条实际优化轨迹；reused_from显式记录复用 |
| by_base_summary.csv | 宽度/基础/预算/方法/步数 | 先平均2个adapter，不把adapter当基础样本 |
| summary.csv | 宽度/预算/方法/步数 | 跨10个基础均值，SD为基础均值样本SD；ShuffledMean先每基础平均5分组 |
| paired_differences.csv | 宽度/基础/预算/步数 | 对Full/随机均值的配对差、真实组在6方案中的名次 |
| bootstrap_input.csv | 新基础ID | 两预算各6列：long/short两对比及两交互，共12列 |
| confidence_intervals.csv / bootstrap.json | 12个预定对比 | 200000次基础配对百分位bootstrap，8个预定CI下界均>0才进入A |
| verdict.json | 整体预定门 | 两预算联合判断；非等效性检验，不因某一项阳性走A |
| worker_links.json | 两个计算进程的来源 | 不同宽度的独立进程；并发下的wall time仅描述性 |
| shards/ | 原始子运行非tensor副本 | 完整日志、分配、校准NPZ、检查点审计与代码/协议快照 |

容量复核另有36条旧基础轨迹、72条定步记录；单组机制探针另有84条旧基础轨迹、168行，参数192/416不混作同预算主方法对比。Spearman按矩阵成本族分层、以基础分开报告；3个发现基础不支持机制规律的确认性结论。

各基础config.json沿用源预训练/评价配置，包含未调用的pilot adapter/sweep字段。真实适配参数由子运行根config.json、协议、单位分配和保存adapter共同定义，使用600步、lr0.001、alpha/sqrt(rank)，不读取旧字段执行180步或lr0.003。
