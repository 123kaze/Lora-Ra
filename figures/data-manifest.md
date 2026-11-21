# D2 图数据契约

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| comparison | runs/<id>/main.csv | Real, only after execution | 参数匹配的适配运行 | figures/pilot/plot_results.py | runs/<id>/figures/comparison.png, .svg |
| sensitivity | runs/<id>/spectra.csv | Real, only after execution | 训练集平均梯度SVD | figures/pilot/plot_results.py | runs/<id>/figures/sensitivity.png, .svg |
| prediction | runs/<id>/sweep.csv | Real, only after execution | 独立验证loss与代理收益 | figures/pilot/plot_results.py | runs/<id>/figures/prediction.png, .svg |

PNG为450dpi，SVG为矢量；图中英文标注避免WSL缺失中文字体。文件尚未生成时不视为证据。

2026-11-22实际生成：`runs/pilot_20261122_01/`下的`main.csv`、`summary.csv`、`spectra.csv`、`sweep.csv`为真实数据，三图PNG/SVG位于其`figures/`子目录，均已目视检查。`smoke_20261122_01`下图表只供流程检查，不作为研究证据。

仓库审阅副本：4个CSV在`results/pilot_20261122_01/`，三图PNG/SVG在`figures/pilot_20261122_01/`。与原始输出逐字节相同；对应SHA-256见`results/pilot_20261122_01/artifact-manifest.json`。完整报告只调整图链接，没有修改实验数值。

第二轮约定（仅在运行完成后构成证据）：`diagnostic_20261122_02`的small_budget/capacity/concentration三图由`src/carryrank/diagnostic_report.py`读取实际`diagnostic.csv`生成，输出至`figures/diagnostic_20261122_02/`，PNG450dpi与SVG各一份。基础seed先聚合adapter seed，再呈现基础之间的波动；smoke图不计入研究结果。

第二轮正式图已实际生成并逐幅查看；391个导出文件哈希及原始数据关联通过独立核验。源数据是`results/diagnostic_20261122_02/diagnostic.csv`及`by_base_summary.csv`，输出为同run-id下small_budget、capacity、concentration的PNG/SVG。解释与适用边界见`reports/diagnostic_review_20261122.md`。
