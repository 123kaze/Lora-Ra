# 第0步离线证据契约

| 文件 | 行单位与用途 | 边界 |
|---|---|---|
| shapes.csv | 宽度×范围，模块数/组数/严格rank1成本 | 静态结构，不是性能 |
| budget_feasibility.csv | 宽度×范围×预算，严格/整数基线可行性与成本族可达性 | 不可行≠方法负结果 |
| allocation_scan.csv | 宽度×发现基础×范围×分组×预算 | 仅训练集校准，没有微调结果 |
| nondegeneracy_summary.csv | 范围×预算，6个发现基础的TV最小/最大值 | 参数分配距离，不是主指标 |
| selection.json | 原严格规格状态、替代定义候选点 | formal_budgets_locked为空；没有A/B决定 |
| spectra.csv | 20160条双侧谱前缀 | G/Hin/Hout经验矩，非精确Hessian |
| calibration_arrays.npz | 每基础所有扩张层7类矩阵的G/Hin/Hout | NumPy可读，不含模型权重 |
| previous_reversal_diagnostic/ | 旧末层实验12组矩、4608条谱、6个基础/12个adapter配对结果 | 已知结果的事后诊断，非因果/预测规律验证 |
| power_reference.json | 规划用配对t近似 | n10不等于达到80% bootstrap/联合门功效 |

TV使用同实际预算的模块参数向量，归一化为`L1/(2B)`。V仅为预算集中到末层V的几何参考；在替代预算上它不是可实现的同rank VBalanced训练方法。严格Uniform、整数均衡、宽度启发式等定义必须在报告和协议中分开命名。
