# 第二轮诊断实测证据

完整解读：[为什么梯度谱舍弃Q](../../reports/diagnostic_20261122.md)。

- [原始218条固定步数记录](diagnostic.csv)，来自109条轨迹。
- [每个基础内部聚合](by_base_summary.csv)、[基础内部配对效应](paired_effects.csv)、[自动报告](report.md)。
- [运行审计](diagnostic_verification.json)、[109个检查点重载审计](checkpoint_audit.json)。
- [末步梯度probe](late_gradient_probe.csv)、[分析边界](late_gradient_probe_provenance.json)、[末步输出变化](final_output_shifts.csv)。
- [Q/K等价参数化probe](../parameterization_probe_20261122_01/parameterization.csv)，不是训练性能比较。
- [冻结基础测试指标](../calibration_probe_20261122_01/frozen_base_metrics.json)。
- [小预算图](../../figures/diagnostic_20261122_02/small_budget.png)、[容量图](../../figures/diagnostic_20261122_02/capacity.png)、[集中分配图](../../figures/diagnostic_20261122_02/concentration.png)，均另有SVG。

每个`base_<seed>/`目录含配置、预训练记录、校准谱和`diagnostic_adapters/`逐条结果。基础2026继承第一轮，2027/2028重新预训练；两个adapter种子不能当作更多独立基础重复。

## 不装torch也能看校准

每个基础的`calibration_arrays.npz`只含float64平均权重梯度与输入E[x²]，不含模型权重；键名映射在相邻JSON。

```python
import numpy as np
arrays = np.load('results/diagnostic_20261122_02/base_2026/calibration_arrays.npz', allow_pickle=False)
print(arrays.files)
gradient = arrays['gradient__blocks_4_attn_v_proj']
moment = arrays['moment__blocks_4_attn_v_proj']
```

只用标准库核验散列和汇总；加`--spectra`用NumPy重新计算所有谱，均不需要torch、CUDA或`runs/`：

```powershell
python scripts/verify_diagnostic.py results/diagnostic_20261122_02
python scripts/verify_diagnostic.py results/diagnostic_20261122_02 --spectra
```

`artifact-manifest.json`是原始正式导出的391个文件清单，`supplement-manifest.json`记录随后只读分析、可移植校准与解读。源代码快照在`source_snapshot/`。完整检查点重载另外需要原始tensor。
