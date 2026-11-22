# GPT-2两任务实际补充实验

没有同时在两个任务上检出双侧谱分配的标签NLL劣势，不能确认这个推广判断。

本轮检验标准等宽预训练模型上的层间rank分配；不重新检验C/R、不追改第0步的B路线结论。无论方法排名如何，单个checkpoint、两个分类任务都不能支持“集中分配总是有害”。

## 实际运行

[预训练模型固定版本](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)、[GLUE数据固定版本](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c)、[官方PEFT AdaLoRA接口](https://huggingface.co/docs/peft/package_reference/adalora)。实际软件版本、源代码、文件SHA另随运行保存。

模型参数数目：124439808，GPT-2固定revision。SST-2/MRPC各256条独立tune，正式验证分别872/408条，测试集未下载。SST-2训练池4096条、MRPC训练池3412条；200步按固定seed有放回抽样，共3200个样本位置，不是完整epoch。prompt末尾最多128token；只预测一个标签token。

4方法×2任务×3个LR×独立seed0=24条调参训练；选择仅看tune最终NLL。随后锁定选择，4方法×2任务×5个新adapter/data seed=40条正式训练。全部200步，不挑最好checkpoint。5个seed只反映同一预训练checkpoint下的适配波动。

## 正式验证结果

均值 ± 样本SD跨5个adapter/data seed。NLL是完整词表中目标标签token的负对数概率，分类预测在预设两个标签之间取argmax。MRPC的F1为正类F1，不以NLL排名替代分类排名。

|任务|方法|标签NLL ↓|Accuracy ↑|正类F1 ↑|
|---|---|---:|---:|---:|
|sst2|UniformLoRA|0.3336 ± 0.0102|86.9954 ± 0.1739%|87.0729 ± 0.4463%|
|sst2|AdaLoRA|0.6251 ± 0.0113|69.9312 ± 0.9608%|67.2839 ± 2.8558%|
|sst2|TwoSidedSpectral|0.3404 ± 0.0129|86.7202 ± 0.1884%|86.7987 ± 0.4699%|
|sst2|CoverageFloor|0.3404 ± 0.0129|86.7202 ± 0.1884%|86.7987 ± 0.4699%|
|mrpc|UniformLoRA|0.5523 ± 0.0146|72.1078 ± 1.5247%|80.9365 ± 1.7577%|
|mrpc|AdaLoRA|0.6140 ± 0.0194|71.0294 ± 1.7961%|81.0195 ± 0.9486%|
|mrpc|TwoSidedSpectral|0.5520 ± 0.0069|73.1373 ± 0.4101%|81.9832 ± 0.6332%|
|mrpc|CoverageFloor|0.5638 ± 0.0112|72.5000 ± 1.3016%|80.9130 ± 1.6174%|

冻结模型和不读文本的多数类预测只作参考，不参与LR选择：

|任务|冻结模型NLL|冻结模型Accuracy|冻结模型F1|多数类Accuracy|多数类F1|
|---|---:|---:|---:|---:|---:|
|sst2|6.2241|65.14%|72.06%|50.92%|67.48%|
|mrpc|9.3062|32.35%|2.13%|68.38%|81.22%|

## 配对差与预设判断

差值方向按方法名顺序：Uniform−Spectral为主比较，负值表示Uniform的NLL更低；Spectral−CoverageFloor为次要比较，正值有利覆盖下限；Uniform−AdaLoRA正值有利AdaLoRA。按5个配对适配seed共享索引重采样200000次，95%边际percentile CI；不是同时置信带或后验概率。预先规定只有两个主CI上界均<0，才写“两任务均检出谱NLL劣势”。

|任务|差值|均值|95% CI|主比较|
|---|---|---:|---|---|
|sst2|uniform_minus_spectral|-0.006771|[-0.011957, -0.001516]|True|
|sst2|spectral_minus_floor|+0.000000|[+0.000000, +0.000000]|False|
|sst2|uniform_minus_adalora|-0.291451|[-0.298087, -0.284494]|False|
|mrpc|uniform_minus_spectral|+0.000334|[-0.010846, +0.015693]|True|
|mrpc|spectral_minus_floor|-0.011866|[-0.021931, -0.002708]|False|
|mrpc|uniform_minus_adalora|-0.061725|[-0.080489, -0.038168]|False|

sst2：Uniform的标签NLL更低，区间不含0。
mrpc：该差值区间包含0，未建立稳定NLL胜负。

部分比较的分类指标均值方向与NLL相反；完整记录在interpretation.json。这种分歧不通过事后更换主指标解决，也不能只选择较有利的指标。

## 分配和机制区分力

|任务|方法|rank向量（层0→11）|HHI|非零层数|TV对Uniform|
|---|---|---|---:|---:|---:|
|sst2|UniformLoRA|[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]|0.083333|12|0.000000|
|sst2|TwoSidedSpectral|[5, 7, 6, 4, 4, 4, 4, 2, 3, 2, 3, 4]|0.093750|12|0.125000|
|sst2|CoverageFloor|[5, 7, 6, 4, 4, 4, 4, 2, 3, 2, 3, 4]|0.093750|12|0.125000|
|mrpc|UniformLoRA|[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]|0.083333|12|0.000000|
|mrpc|TwoSidedSpectral|[7, 8, 7, 6, 5, 4, 2, 2, 2, 1, 2, 2]|0.112847|12|0.270833|
|mrpc|CoverageFloor|[7, 8, 7, 6, 4, 4, 2, 2, 2, 2, 2, 2]|0.110243|12|0.250000|

同配置情况：sst2的CoverageFloor与TwoSidedSpectral相同。共5个正式单位虽实际重训，仍不是新的分配干预；不能据其打平或微小随机差证明覆盖率机制。

HHI越大表示rank越集中；Uniform为1/12。TV是同形状模块参数分布的L1/(2B)。这里只含12个同形状融合QKV矩阵；所有层均非零时不能描述为“只集中到少数矩阵”。CoverageFloor同时改变了具体层的容量，因此即使更好也不隔离所有位置与优化因素。

## 预算与学习率

|方法族|物理可训练参数|最终有效因子参数|最终有效参数（含E）|
|---|---:|---:|---:|
|固定LoRA三方法|147456|147456|147456|
|AdaLoRA|295008|147456|147504|

匹配的是最终有效rank容量。AdaLoRA训练期物理参数约2倍，最终还多48个E标量（约0.0326%）；不称全程同参数、同显存或同计算。其标准SVD初始化、正交正则及rank缩放与静态rsLoRA不同，各方法在同一LR网格独立选LR，以减少优化混杂，但不是穷尽超参数搜索。

|任务|方法|选定LR|tune NLL|
|---|---|---:|---:|
|sst2|UniformLoRA|1e-03|0.294742|
|sst2|AdaLoRA|1e-03|0.559244|
|sst2|TwoSidedSpectral|1e-03|0.287104|
|sst2|CoverageFloor|1e-03|0.287104|
|mrpc|UniformLoRA|1e-03|0.565555|
|mrpc|AdaLoRA|1e-03|0.631842|
|mrpc|TwoSidedSpectral|1e-03|0.559702|
|mrpc|CoverageFloor|1e-03|0.561725|

选定LR落在预定网格边界的方法：sst2/UniformLoRA、sst2/AdaLoRA、sst2/TwoSidedSpectral、sst2/CoverageFloor、mrpc/UniformLoRA、mrpc/AdaLoRA、mrpc/TwoSidedSpectral、mrpc/CoverageFloor。按协议不追加LR救结果；这些比较不是各方法经无限调优后的性能上限。

效率实测在summary.csv中，包含训练时间、PyTorch峰值allocated及动态有效参数峰值/均值。桌面、驱动占用不在allocated中，单机计时不是专用环境下的通用性能基准。

另在全部训练完成后、无本实验训练并发时复测校准开销；无optimizer更新，不参与调参。原始数据及模型资产重新核对SHA，冻结模型验证指标也独立重载。

|任务|校准资源复测秒数|PyTorch峰值allocated MiB|原分配复现|
|---|---:|---:|---|
|sst2|6.58|1026.77|True|
|mrpc|6.99|1026.52|True|

复核资产/数据文件16个，冻结基线指标重载最大误差0.0。

## 正确性与边界

全部40个正式adapter在新建适配器后独立重载，NLL/accuracy/F1误差0；初始化前向误差0，冻结基础SHA一致，真实参数和最终rank预算逐单位核对。临时FP32核对完整/末位前向和模块合并，随后恢复原精度并再核对基础SHA。原始失败smoke与成功smoke单独保留，不当成正式数据。

校准只用训练64例，FP32前反向和FP64 CPU SVD；所有校准矩阵、前缀及精确DP可纯NumPy重算。官方验证只用于固定评价，未进入LR选择；源资产与数据切分均固定revision和SHA。MRPC的多数类准确率/F1可用逐样本标签计算，不能把高F1自动解读为强句对理解能力。

该实验没有多个预训练种子、其他真实模型或生成任务，也没有纯浓度而位置不变的因果干预。它不能替代变宽C/R大模型验证，也不修改第0步阴性结论。GEAR未纳入下游性能比较。

## 直接审阅

- [正式表](../../results/realmodel_gpt2_20261122_01/main.csv)、[调参表](../../results/realmodel_gpt2_20261122_01/tuning.csv)、[全部CI](../../results/realmodel_gpt2_20261122_01/confidence_intervals.csv)、[机器判断](../../results/realmodel_gpt2_20261122_01/interpretation.json)。
- [主方法比较](../../figures/realmodel_gpt2_20261122_01/comparison.png)、[配对区间](../../figures/realmodel_gpt2_20261122_01/paired_nll.png)、[分配](../../figures/realmodel_gpt2_20261122_01/allocation.png)。
- `python scripts/verify_realmodel_evidence.py results/realmodel_gpt2_20261122_01`：无需torch、GPU或原始runs。纯数值核验不声称无权重重载了模型；大权重和.pt只在本地保留。
