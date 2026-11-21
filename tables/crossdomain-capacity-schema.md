# 跨域容量筛查数据契约

| 表 | 每行 | 主要字段 | 来源 | 解释边界 |
|---|---|---|---|---|
| main.csv | 宽度/基础seed/adapter seed/rank/固定步数 | budget, val_loss, train_eval_loss, val_bytes, base_val_loss, data_trace_sha256 | 实际训练与独立split评价 | 24轨迹、72行；非方法比较 |
| summary.csv | 宽度/基础seed/rank/固定步数 | mean与sample_std, n_adapter_seeds | main.csv按adapter seed聚合 | 适配器波动，不是基础模型波动 |
| trajectories.csv | 一条完整训练轨迹 | 初始化/冻结/合并误差、耗时、显存、文件SHA | 实际训练审计 | 末步，不是最佳步 |
| capacity_gate.json | 每宽度/基础的rank1减rank4配对差 | 所有配对差、mean、sample_std、pass | 预先固定协议 | 两种宽度全部满足才通过；不证明标签优势 |
| checkpoint_audit.json | 各末步独立重载 | 验证误差、合并误差、旧数据SHA检查 | 独立进程 | 内部一致性，不是独立重训 |

图`capacity.png/svg`：全部180/360/600步容量曲线，均值±3个adapter seed样本标准差。图`paired_gate.png/svg`：600步rank1/rank4逐种子配对。只绘实际数据。
