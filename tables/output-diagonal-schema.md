# 输出侧对角校准数据契约

| 文件 | 一行代表什么 | 用途 |
|---|---|---|
| moments.csv | 基础×缩放×投影 | 实测输入/输出梯度二阶矩及G/H变换误差 |
| invariance.csv | 基础×缩放×分组×评分 | M、谱、rank前缀收益相对c=1的误差 |
| spectra.csv | 上述组合的一个奇异方向 | 实测奇异值、前缀收益、rank成本 |
| allocation_summary.csv | 基础×缩放×评分×预算 | Q/V实际参数量、精确预算与字典一致性 |
| allocations.json | 同上 | 完整rank配置，供NumPy独立重算 |
| base_*/scale_*.npz | 一次完整校准 | float64 G、Hin、Hout，不含模型权重 |

不生成性能对比图；本阶段没有训练、验证或测试指标。相对阻尼与固定绝对阻尼是度量一致性对照，不是方法性能比较。
