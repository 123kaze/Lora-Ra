# 方法与证据对应 D1

| Contribution / hypothesis | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 残差旁路、重新激活标记正确 | model.structure / Block | 历史与旁路测试 | verification.json | 工程实现符合本pilot定义 | 待运行 |
| C/R适配需求不同 | grouped input-column LoRA | C-only vs R-only vs Equal | main.csv / comparison | 此配置任务中的差异 | 待运行 |
| 梯度谱辅助容量分配 | average G / SVD / exact DP | Gradient vs Equal/Uniform | main.csv | 单位曲率梯度pilot的实测效果 | 待运行 |
| 真实结构超出任意同形状分组 | shuffled dimensions | Gradient vs Shuffled | main.csv | 固定随机划分控制下的差异 | 待运行 |
| 代理收益与实际收益排序有关 | U(k) | 固定V组rank sweep | sweep.csv / prediction | 相关诊断，不是全局理论保证 | 待运行 |
| 低成本、可合并、冻结 | lora.merge | 计时/显存/等价 | verification.json / main.csv | 本机本实现的开销 | 待运行 |
| K-FAC、动态rank、跨组耦合、年龄建模 | 未实现 | 无 | 无 | 仅未来工作，不能主张有效 | 未验证 |
