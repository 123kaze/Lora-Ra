# 第二轮诊断数据契约

固定协议：`plan/diagnostic-protocol.md`；正式run：`diagnostic_20261122_02`。

| 文件 | 一行代表什么 | 统计/比较规则 |
|---|---|---|
| diagnostic.csv | 一条训练轨迹在固定180或600步的验证结果 | 不按最优验证步数挑选；不使用测试集调参 |
| by_base_summary.csv | 同基础seed、方法、scale、步数下的两个adapter seed均值 | 基础seed是重复层级，不把六个adapter运行当六个独立基础 |
| paired_effects.csv | 每个基础seed上的方法相对Gradient验证loss优势 | 正值表示方法更好；报告三个基础seed各自差值 |
| base_*/normalized_spectra.csv | 一个校准组的一个rank前缀收益 | 同训练校准样本，原谱与对角输入归一化谱并列 |
| base_*/diagnostic_adapters/*/updates.csv | 固定步数的每个实际适配因子更新范数 | 记录rank、scale、delta范数及相对原投影权重范数 |
| base_*/diagnostic_adapters/*/result.json | 一条完整600步轨迹 | 保存rank、两处固定验证指标和工程检查 |

图只从上述真实CSV生成，保存为`figures/diagnostic_20261122_02/*.png,*.svg`。完整轻量证据自动导出到同run-id的`results/`目录；权重留在`runs/`。

`scripts/probe_late_gradients.py`是另行标注的事后只读分析：重载固定末步适配器，重测同一训练校准集上的全投影梯度，不更新权重、不重新分配rank。它只能用于探索时间变化，不能当作独立确认性结果。
