# D2 图数据契约

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| comparison | runs/<id>/main.csv | Real, only after execution | 参数匹配的适配运行 | figures/pilot/plot_results.py | runs/<id>/figures/comparison.png, .svg |
| sensitivity | runs/<id>/spectra.csv | Real, only after execution | 训练集平均梯度SVD | figures/pilot/plot_results.py | runs/<id>/figures/sensitivity.png, .svg |
| prediction | runs/<id>/sweep.csv | Real, only after execution | 独立验证loss与代理收益 | figures/pilot/plot_results.py | runs/<id>/figures/prediction.png, .svg |

PNG为450dpi，SVG为矢量；图中英文标注避免WSL缺失中文字体。文件尚未生成时不视为证据。

2026-11-22实际生成：`runs/pilot_20261122_01/`下的`main.csv`、`summary.csv`、`spectra.csv`、`sweep.csv`为真实数据，三图PNG/SVG位于其`figures/`子目录，均已目视检查。`smoke_20261122_01`下图表只供流程检查，不作为研究证据。
