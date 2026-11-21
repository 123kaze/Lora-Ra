# 跨域容量筛查真实数据

运行：`crossdomain_capacity_20261122_01`。容量门结果：`True`。本轮没有比较CarryRank方法，没有读取WikiText test。

`main.csv`包含全部定步记录，`summary.csv`按单个基础、宽度、rank、步数聚合adapter seeds，std为样本标准差。`capacity_gate.json`保留预定rank1/rank4配对结果；`checkpoint_audit.json`为独立末步重载；各基础目录保存逐次训练日志和adapter result.json，tensor未包含在此轻量包。

无torch复核：`python scripts/verify_capacity.py results/crossdomain_capacity_20261122_01`。从数据源重建缓存：`python scripts/prepare_wikitext.py`（需要pyarrow）。训练：`PYTHONPATH=src python -m carryrank.capacity --output runs/<fresh-id>`。

[完整运行报告](report.md)。
