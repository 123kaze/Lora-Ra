# D2 表格契约

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| main | 等参数主要对照 | 方法×adapter seed | val/test loss、bits/byte、accuracy、params、time、memory、merge/init/frozen | runs/<id>/main.csv | 实验脚本，真实运行 |
| summary | 汇总配对种子 | 每方法 | 均值±样本std，Gradient相对Equal/Shuffled配对差 | main.csv | report阶段 |
| spectra | 校准敏感度 | 分组×奇异方向 | sigma、每参数预测收益、组形状、age | calibration.pt→spectra.csv | calibrate阶段 |
| sweep | 预测诊断 | 组×rank | predicted utility、actual val improvement、params | sweep.csv | sweep阶段 |

无mock数据；空表不得回填估计数。std使用ddof=1。单种子诊断不计算显著性。
