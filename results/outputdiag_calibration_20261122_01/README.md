# 双侧对角校准证据

本轮训练步数为0，只使用目标训练校准集。

- [完整解读](../../reports/output_diagonal_20261122.md)
- [验收结果](verification.json)、[配置](config.json)、[事前协议](protocol.md)
- [测得二阶矩与变换误差](moments.csv)、[逐组不变性](invariance.csv)、[全部谱](spectra.csv)
- [分配汇总](allocation_summary.csv)、[完整rank字典](allocations.json)
- [旁路年龄与预算成本](structure_and_budget_feasibility.json)

每个`base_<seed>/`中含三个缩放值实际采集的NPZ及键名映射JSON：`G`为平均任务梯度，`Hin/Hout`为输入/输出梯度的未中心化二阶矩。c的三个值都实际执行了前反向。输出梯度已消除平均loss的1/N因子，详细定义见配置和协议。

不装torch也可复核全部谱与精确背包：

```powershell
python scripts/verify_output_diagonal.py results/outputdiag_calibration_20261122_01
```

仅需NumPy。会核对导出SHA-256，并从NPZ独立重算9216个rank前缀收益及144个完整分配，不依赖原始runs目录或模型权重。

原始证据在`artifact-manifest.json`；报告、测试、NumPy验证器等补充文件在`supplement-manifest.json`。通过这项标量重参数化检查，不等于已经验证方法的预测能力或任务收益。
