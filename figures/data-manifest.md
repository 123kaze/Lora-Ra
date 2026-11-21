# D2 图数据契约

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| comparison | runs/<id>/main.csv | Real, only after execution | 参数匹配的适配运行 | figures/pilot/plot_results.py | runs/<id>/figures/comparison.png, .svg |
| sensitivity | runs/<id>/spectra.csv | Real, only after execution | 训练集平均梯度SVD | figures/pilot/plot_results.py | runs/<id>/figures/sensitivity.png, .svg |
| prediction | runs/<id>/sweep.csv | Real, only after execution | 独立验证loss与代理收益 | figures/pilot/plot_results.py | runs/<id>/figures/prediction.png, .svg |

PNG为450dpi，SVG为矢量；图中英文标注避免WSL缺失中文字体。文件尚未生成时不视为证据。
