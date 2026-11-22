# GPT-2两任务实测数据契约

运行目录使用新的realmodel_gpt2标识；调参、正式、smoke分开。

|文件|实际内容|统计单位与边界|
|---|---|---|
|main.csv|40条正式200步适配结果|2任务×4方法×5个adapter/data seed；只有1个预训练checkpoint|
|tuning.csv|24条独立seed0调参结果|2任务×4方法×3个固定LR，只用train派生tune集|
|summary.csv|8行方法/任务均值及样本SD|跨5个适配seed，不称5个独立基础；MRPC同时保留accuracy和正类F1|
|paired_differences.csv|30条种子内配对差|每任务3个预设对比×5seed；正负方向以first/second方法名为准|
|confidence_intervals.csv|6个配对NLL边际区间|Uniform−Spectral是主比较；两任务上界均<0才检出一致谱NLL劣势|
|bootstrap.json/bootstrap_input.csv|共享索引的重采样定义及输入|200000次，seed95531，5个适配seed为重采样单位|
|units/*/predictions.csv|逐样本标签、预测、标签logprob和两个标签logit|正式872/408例，调参256例。NLL来自完整词表，不是两个标签归一化后的NLL；单位nat/label-token|
|units/*/training.jsonl|200步CE、总loss、正则、LR、rank、数据SHA|AdaLoRA正则与原始NLL分列；有效rank不等于物理存储压缩|
|calibration/*/module_*.npz|每任务12个FP32 G/Hin/Hout|G按输出×输入存储为2304×768；矩/谱仅来自训练64例|
|calibration/*/spectra.csv|每矩阵8个奇异值与前缀收益|FP64 CPU NumPy SVD；可独立重算，非方法最终收益|
|calibration/*/allocations.json|固定方法的精确rank分配|总rank48，每rank3072因子参数；CoverageFloor同配置时不提供新机制证据|
|checkpoint_audit.json|40个正式adapter独立GPU重载的审计|轻量包的纯NumPy核验不声称重新执行GPU推理|

固定LoRA三方法物理/有效参数均147456；AdaLoRA物理295008，最终有效因子147456、加48个E标量后147504。匹配最终rank容量，不匹配全程物理参数或计算量。全部大权重及.pt只保留本地；公开CSV/NPZ/日志/报告须真实生成后才构成证据。
