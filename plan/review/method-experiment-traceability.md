# 方法与证据对应 D1

| Contribution / hypothesis | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 残差旁路、重新激活标记正确 | model.structure / Block | 历史与旁路测试 | verification.json | 工程实现符合本pilot定义 | 10测试与运行审计通过 |
| C/R适配需求不同 | grouped input-column LoRA | C-only vs R-only vs Equal | main.csv / comparison | 此配置任务中的差异 | C-only三个种子loss均低于R-only，不支持优先R |
| 梯度谱辅助容量分配 | average G / SVD / exact DP | Gradient vs Equal/Uniform | main.csv | 单位曲率梯度pilot的实测效果 | 未稳定胜出，均值更差，不支持优势主张 |
| 真实结构超出任意同形状分组 | shuffled dimensions | Gradient vs Shuffled | main.csv | 固定随机划分控制下的差异 | 三种子验证均略好，均值0.00281，未达0.01门 |
| 代理收益与实际收益排序有关 | U(k) | 固定V组rank sweep | sweep.csv / prediction | 相关诊断，不是全局理论保证 | 整体rho0.5429，C内-0.5/R内1.0，混合证据 |
| 低成本、可合并、冻结 | lora.merge | 计时/显存/等价 | verification.json / main.csv | 本机本实现的开销 | 冻结/合并/重载通过；计时仅桌面本机诊断 |
| K-FAC、动态rank、跨组耦合、年龄建模 | 未实现 | 无 | 无 | 仅未来工作，不能主张有效 | 未验证 |

实际证据目录：`runs/pilot_20261122_01/`。该表仅更新证据状态，未修改运行前锁定的协议和门槛。

补充的必要一致性检查：`outputdiag_calibration_20261122_01`使用实测双侧对角二阶矩，通过三个基础、c=1/8与8、两种分组/两种预算下的标量Q/K重参数化不变性。它不支持最终收益或真实C/R标签优势主张；完整K-FAC依然未实现。

跨域补充证据：`crossdomain_capacity_20261122_01`通过有限步容量响应门；`crossdomain_labels_20261122_01`在长短旁路、3基础和5个随机列分组下未通过标签门。真实双侧分配与VBalanced六次配置相同，不能据此评价不同rank分配算法的性能胜负；随机列分组仍对应不同更新族，支持本设置的结构标签零/混合证据。具体数据与边界见`reports/crossdomain_labels_20261122.md`。
