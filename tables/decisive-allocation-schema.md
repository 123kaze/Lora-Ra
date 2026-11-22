# 正式第0步分配摘要

`results/decisive_main_20261122_01/allocation_summary.csv`：20个基础×2预算×9方法=360行。actual_parameters从真实分组cost×rank逐项相加；total_rank不等于参数成本。nonzero_groups/candidate_groups和nonzero_modules/candidate_modules报告覆盖及零rank。WidthRankProportional为明确的GroupedBudgetBalanced别名。

`allocation_distances.csv`：20基础×2预算×2均衡对照=80行。module_parameter_tv = sum_module abs(parameters_true−parameters_baseline)/(2B)，不是rank向量原始L1。正式新基础全部保留，不根据距离或loss剔除；所有实测值>0.30只是描述本轮实际非退化性。

生成`python scripts/report_decisive.py results/decisive_main_20261122_01`，独立重算`python scripts/verify_decisive_main.py results/decisive_main_20261122_01`。原始精确rank和行/列索引见shards/<width>/<width>_base_<seed>/assignments.json。
