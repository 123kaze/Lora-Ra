# 第0步：离线扫描及2026反向诊断

6个既有基础，72个FP32校准批次，没有新微调。2160个分配、20160条双侧谱前缀。原指定预算存在严格基线不可行与成本族不可达；用户随后选择整数预算均衡与末层七类主范围，1216/2464为相应候选。`resolved_specification.json`记录用户选择和正式协议散列；`selection.json`保存选择前的历史状态，不应将当时的待决状态误作当前执行状态。

查看`shapes.csv`、`budget_feasibility.csv`、`nondegeneracy_summary.csv`和`allocations.json`。V只为预算分布几何参考，不是1216/2464下可实现的VBalanced方法。`previous_reversal_diagnostic/`为旧结果事后诊断，不证明机制或预测规律。`legacy_inputs/`包含复核该诊断所需的旧NPZ和CSV，无需torch。

完整复核：`python scripts/verify_decisive_preflight.py results/decisive_preflight_20261122_01`。复跑旧案例：`python scripts/analyze_2026_reversal.py --input results/decisive_preflight_20261122_01/legacy_inputs --output <fresh-output>`。
