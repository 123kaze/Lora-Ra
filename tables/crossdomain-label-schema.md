# 跨域标签对比的数据契约

| 文件 | 行单位 | 统计/验收 |
|---|---|---|
| main.csv | 宽度、基础、adapter seed、方法、固定步数 | 168个原始验证loss；预算384 |
| by_base_summary.csv | 宽度、基础、方法、固定步数 | 先聚合2个adapter seed，SD不用于基础推断 |
| summary.csv | 宽度、方法、步数 | 均值±3个基础均值的样本SD；ShuffledMean先每基础平均5个分组 |
| base_comparisons.csv | 宽度、基础、步数 | shuffled平均减真实、VBalanced减真实，正数有利真实分组 |
| verdict.json | 预定600步、长旁路 | 标签门和方法门分别判断，均要求3基础同向且平均≥0.01 |
| allocation_summary.csv | 宽度、基础、方法 | Q/V参数、rank、真实方案是否等于VBalanced |
| groups.json / calibration_arrays.npz | 每基础的完整列分组与G/Hin/Hout | 保留5个随机seed，无torch可重算SVD/DP |
| checkpoint_audit.json | 84个末步adapter | 独立重载loss、预算、冻结、合并和配对trace |

图只使用实际CSV。方法均值图的误差条为基础间SD；配对差图逐一呈现3个基础，不把adapter×shuffle数作为基础样本量。
