# 跨域真实C/R与五个随机分组

84条600步轨迹、168条180/600步验证记录。长/短旁路、3个基础seed、2个adapter seed、预算384、7种方法。本轮没有读取test，不能视作确认性测试。

`main.csv`是全部原始记录；`by_base_summary.csv`先平均adapter；`summary.csv`中的标准差只跨3个基础均值；`ShuffledMean`先对每个基础的5个随机分组取平均。`base_comparisons.csv`和`verdict.json`保留全部基础配对差。不要把随机分组seed乘adapter seed当独立基础样本。

`allocation_summary.csv`和各基础`groups.json/allocations.json`保留分配与列索引。随机组中的C/R键只是槽位名称；真实语义仅属于true组。`calibration_arrays.npz`不含权重，NumPy可读取；`checkpoint_audit.json`给出独立重载；逐步JSONL和源码快照均保留。

无torch完整统计/SVD复核：`python scripts/verify_labels.py results/crossdomain_labels_20261122_01`。

[完整运行报告](report.md)。
