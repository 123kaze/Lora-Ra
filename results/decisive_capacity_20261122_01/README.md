# 第0步真实证据包

运行`decisive_capacity_20261122_01`，类型`capacity`。主表和逐单位日志来自完整运行；所有.pt只保留本地。protocol.md及源快照记录训练前约定，不以代码、计划或smoke作为实验结果。

FullBudgetBalanced是整数预算均衡的整矩阵LoRA；2464恰为末层7矩阵全rank1。GroupedBudgetBalanced不等同于无法在这些预算上实现的严格全组统一rank。WidthRankProportional是已证明等价的别名，复用来源在conditions.csv/reused_from中明确记录。

标准差先平均adapter，再跨基础均值。主试验10个新基础；容量与单组探针仅3个发现基础，分开报告。基础config.json中的pilot adapter/sweep字段未调用，实际适配使用运行根config.json、协议和单位分配。联合CI规则不按中途结果改动，阴性不是等效性证明。

主分析可用`python scripts/verify_decisive_main.py results/decisive_capacity_20261122_01`无torch复核。容量/机制阶段可用`python scripts/verify_decisive_phase.py results/decisive_capacity_20261122_01`。有完整报告时请查看`report.md`。

[完整报告](report.md)。
