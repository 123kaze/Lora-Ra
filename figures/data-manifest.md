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

跨域容量真实数据：`results/crossdomain_capacity_20261122_01/main.csv`与`summary.csv`，24条轨迹、72条定步记录。脚本`figures/crossdomain_capacity/plot_capacity.py`生成`figures/crossdomain_capacity_20261122_01/capacity.png,svg`及`paired_gate.png,svg`。均值和样本SD仅跨3个adapter seed，基础seed2026，每宽度1个基础；图不证明C/R方法优势。

跨域标签真实数据：`results/crossdomain_labels_20261122_01/main.csv`、`summary.csv`、`base_comparisons.csv`。84条轨迹、168行，3基础×2adapter、5随机分组。脚本`figures/crossdomain_labels/plot_labels.py`生成`figures/crossdomain_labels_20261122_01/comparison.png,svg`和`paired_differences.png,svg`。均值图SD跨基础均值，配对图展示3个基础原始差；非独立确认性测试。

第0步离线真实校准：`results/decisive_preflight_20261122_01/nondegeneracy_summary.csv`和`previous_reversal_diagnostic/full_spectra.csv`。脚本`figures/decisive_preflight/plot_preflight.py`输出budget_screen与old_reversal的PNG450dpi/SVG；无微调性能新证据，旧反向点仅事后诊断。

第0步扩大范围容量实测：`results/decisive_capacity_20261122_01/summary.csv`，3个基础均值的样本SD；`figures/decisive_aux/plot_aux.py`生成同run-id的capacity PNG450dpi/SVG。36条轨迹，独立于10新基础主试验。

第0步主实验真实数据：`results/decisive_main_20261122_01/confidence_intervals.csv`和`summary.csv`。`figures/decisive_main/plot_decisive.py`生成joint_gate与comparison PNG450dpi/SVG。联合门图为预定8项配对边际95%区间；方法图SD跨10个基础均值。720逻辑/640实际轨迹，宽度别名不重复画；两图均已目视检查。

第0步机制真实数据：`results/decisive_mechanism_20261122_01/single_group_results.csv`和`spearman_by_base.csv`。`figures/decisive_mechanism/plot_mechanism.py`生成prediction与sensitivity PNG450dpi/SVG。84条独立优化单元但仅3基础/宽度，192/416参数成本分层，非主检验、非14个独立基础；两图均已目视检查。

GPT-2真实两任务补充已实际完成：`results/realmodel_gpt2_20261122_01/main.csv`、`summary.csv`、`confidence_intervals.csv`。脚本`figures/realmodel/plot_results.py`已生成`figures/realmodel_gpt2_20261122_01/`中的comparison、paired_nll、allocation的PNG450dpi/SVG，三幅PNG已逐一目视检查。40条正式训练，误差条跨5个adapter/data seed，只有1个预训练checkpoint。NLL与任务accuracy/F1均呈现；配对图是20万次共享seed索引的边际95% bootstrap CI。SST-2的Spectral/Floor同rank且独立重训，不重复计为机制证据；所有静态分配均覆盖12层，分配图不能单独证明集中程度的因果作用。24条调参与smoke不进入正式图。279个清单文件在无torch、无runs的干净副本复核通过，含谱、精确DP及全部绘图统计数据；篡改汇总并刷新哈希仍被拒绝。
