# 阶段门

- D0：`experiment-protocol.md`、`configs/pilot.json` 在结果前固定。
- D1：`review/method-experiment-traceability.md` 标注已验证与未验证。
- D2：`tables/table-schema.md`、`figures/data-manifest.md` 定义真实数据来源。
- D3a：正确性测试、GPU环境、smoke全流程通过。
- D3b：从头预训练、源验证改善达标。
- D3c：6种方法×3适配种子、同预算、rank诊断、原始日志齐备。
- D4：运行审计与报告一致，负结果保留；工程通过与假设通过分别报告。
- D5：只在预设机制门通过后规划扩展；本次无论文完成或投稿质量承诺。

实测状态：D0/D1/D2/D3a/D3b/D3c/D4均完成，证据run=`pilot_20261122_01`。D5继续研究门未通过；保留负/混合结果，停止增加方法复杂度。具体指标和适用边界见`reports/pilot_20261122.md`。
