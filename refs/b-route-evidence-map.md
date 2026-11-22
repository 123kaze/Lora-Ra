# B路线证据—论点映射（2026-11-22）

范围：第0步判为B后，筛查“标量Q/K重参数化不变性”能否成为独立方向。不是完整系统综述，也没有复跑外部算法。以下仅使用已读的一手摘要、论文正文或官方实现；代码以具体commit固定。新颖性尚未证明。

|ID|Citation / source|Source type|可核验事实|支持的具体论点|引用位置|风险与边界|
|---|---|---|---|---|---|---|
|AD|[AdaLoRA](https://arxiv.org/abs/2303.10512)；[官方adalora.py](https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/loralib/loralib/adalora.py#L212)|论文摘要+官方代码|update_ipt用abs(p*p.grad)，再以敏感度/不确定度EMA组合分量分数；另有正交正则|在参数与梯度作相应反向缩放、历史状态也一致映射时，该评分的乘积不变|B报告P2|这是评分代数性质；Adam、正则、初始化及整个训练过程不自动等变|
|EV|[EVA论文](https://arxiv.org/abs/2410.07170v5)；[作者仓库](https://github.com/ml-jku/EVA/tree/5cf4a6c31669aafdc33568a7618161bc329a0800)；[官方PEFT eva.py](https://github.com/huggingface/peft/blob/af1ccdf262b9fe1ad8f0861ae10113a851ee5cbd/src/peft/tuners/lora/eva.py#L302)|论文摘要+官方代码|按输入激活SVD的解释方差比例做rank重分配|仅将Q输出乘c、K输出除c而保持函数时，Q/K输入激活未变，因此该分配信号未变|B报告P2|特定Q/K标量变换，不代表任意基变换或完整训练不变|
|GA|[LoRA-GA](https://arxiv.org/abs/2407.05000v2)；[官方layer.py](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py#L171)|论文摘要+官方代码|给定rank，用梯度SVD方向初始化；unit/stable/gd分支不按梯度奇异值大小做跨模块预算排序|原始梯度被缩放并不自动意味着LoRA-GA会改变rank分配；初始化与rank分配必须区分|B报告P2|固定rank；weightS/clip等选项及训练等变性另需分析|
|GE|[GEAR正文§3.3公式5](https://arxiv.org/html/2611.07349v1#S3.SS3)|论文全文|normal-gradient候选方向分数为sigma²/(n+m)，跨模块全局选取，之后还有训练微批次接受门|若适配状态也作相应标量映射，Q/K normal子空间投影不变而梯度分别乘1/c与c，该候选分数分别乘1/c²与c²|B报告P3|我们的代数推论，不是作者宣称；只证明候选评分可能换序，未证明最终接受rank或任务性能变差；本次未定位到作者实现|
|KF|[Martens & Grosse 2015](https://proceedings.mlr.press/v37/martens15.pdf)|论文全文|§4讨论K-FAC曲率近似及对应更新的不变性，涉及输入/输出仿射重参数化及无阻尼等条件|利用二阶度量消除参数化依赖已有理论背景，不应把白化本身称为新发现|B报告P4|本项目经验对角量不是完整K-FAC或精确Fisher|
|LR|[LoRSum / diagonal fractional K-FAC](https://arxiv.org/abs/2602.16456v1)|一手摘要|LoRA低秩优化结合结构度量及对角K-FAC变体已被研究|低成本对角曲率与LoRA的结合已有直接近邻工作|B报告P4|只读摘要，不能据此声称其rank评分或Q/K不变性与本项目相同|
|UL|[ULoRA](https://arxiv.org/abs/2607.26247v1)|一手摘要|研究梯度到自然梯度之间的预条件初始化族|白化梯度初始化方向也已有直接近邻工作|B报告P4|只读摘要；初始化不同于结构分组及rank预算分配|
|NR|[When Gradient Importance Lies](https://arxiv.org/abs/2605.07366v2)|一手摘要|报告GRPO下梯度重要性分配的单seed/单任务负结果|“梯度重要性不等于容量需求”的方法论问题已有相邻负结果，可比较研究边界|B报告P5|任务、优化器、规模均不同，不能拿其数字佐证本项目C/R结论|

数学推导与合成评分检查独立保存在`reports/b_route_score_probe_20261122.json`；其数据不是语言模型训练或外部方法复现实验。第0步真实下游数据只由本项目正式CSV支持。
