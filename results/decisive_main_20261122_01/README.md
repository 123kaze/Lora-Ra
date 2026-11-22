# 第0步主实验实际证据

固定规则判定 **B**。完整报告见[report.md](report.md)，机器判定见[verdict.json](verdict.json)。20个新基础模型，长短各10；2个adapter种子、5个随机分组、2个预算。720逻辑条件包含80个明确复用的WidthRankProportional别名，实际640条600步训练、1440条验证记录。

- [main.csv](main.csv)：实际定步验证；[conditions.csv](conditions.csv)：实际训练与别名来源。
- [summary.csv](summary.csv)：先平均adapter，再跨基础的均值和样本SD；随机组均值单列。
- [paired_differences.csv](paired_differences.csv)、[confidence_intervals.csv](confidence_intervals.csv)、[bootstrap_input.csv](bootstrap_input.csv)：基础级配对差、全部12项边际95%区间及重采样输入。主门8项不可事后挑预算。
- [allocation_summary.csv](allocation_summary.csv)、[allocation_distances.csv](allocation_distances.csv)：实际参数/rank/零组/覆盖率及非退化性。
- shards/long与shards/short：每基础的spectra.csv、assignments.json、校准NPZ、逐单位result.json、原始训练日志、源码快照、冻结身份以及完整模型独立重载审计。

1216的FullBudgetBalanced是整数预算稀疏均衡，不能冒称严格Uniform；2464才是末层七矩阵全rank1。GroupedBudgetBalanced也不是全组统一rank。WidthRankProportional在该范围与GroupedBudgetBalanced数学等价，显式复用而不伪造训练。基础子目录config.json中的旧pilot adapter/sweep字段未被调用；实际600步、lr0.001等使用根config.json及protocol.md。

复核：`python scripts/verify_decisive_main.py results/decisive_main_20261122_01`，只需NumPy，无需torch、GPU、原runs或检查点。原.pt保留本地runs/，未随轻量包复制。统计/谱/分配可独立重算；无权重不等于可重新训练。

[主门图](../../figures/decisive_main_20261122_01/joint_gate.png)、[实测方法比较](../../figures/decisive_main_20261122_01/comparison.png)。本包在本地完成，未自动提交或推送GitHub。
