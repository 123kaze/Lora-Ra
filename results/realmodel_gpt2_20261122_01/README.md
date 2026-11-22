# GPT-2两任务实际实验数据

只包含完成的非smoke运行。main.csv为40条正式训练，tuning.csv为24条独立调参；只有一个公开预训练checkpoint，每任务5个adapter/data seed。

查看[report.md](report.md)、[summary.csv](summary.csv)、[confidence_intervals.csv](confidence_intervals.csv)、[interpretation.json](interpretation.json)。逐样本标签logprob、预测与训练日志在units/；训练校准矩阵、谱和分配在calibration/。selection_lock.json证明正式seed前固定的LR选择；测试集未下载。

AdaLoRA匹配最终有效因子rank预算，包含额外E标量且峰值物理参数约2倍；不称全程同物理参数或同计算成本。CoverageFloor若与Spectral同分配会明确标注，即使独立重训也不增加机制证据。

全部40个正式adapter已独立GPU重载；.pt和原模型权重只在本地runs/data保留。本轻量包可无torch从预测重算指标、CI以及从FP32校准矩阵重算谱和rank；无权重的核验不等于独立重新训练或重载模型。

`python scripts/verify_realmodel_evidence.py results/realmodel_gpt2_20261122_01`
