# B路线：不变性方向的文献与代码初筛

第0步按训练前协议判为B。本报告在该判定之后进行，不同时启动A路线。当前选择优先整理结构感知PEFT的负结果与实验方法材料；独立“不变性新算法”方向尚未达到投入真实LLM训练的证据门。

检索日期2026-11-22，使用论文原文/摘要及官方仓库。覆盖AdaLoRA、EVA、LoRA-GA、GEAR、K-FAC及两个2026年的曲率近邻工作；这是定向初筛，不是穷尽性新颖性鉴定。[证据映射](../refs/b-route-evidence-map.md)、[段落蓝图](../plan/chapter-blueprints/b-route-screening-blueprint.md)和[覆盖审计](../plan/review/b-route-evidence-coverage.md)先于本报告建立。

## 哪些方法确实受到这次变换影响

设正标量c，W_Q'=cW_Q、W_K'=W_K/c；在本模型的点积注意力及RoPE下保持初始函数，G_Q'=G_Q/c、G_K'=cG_K。需要区分：评分不变、最终rank不变、初始化对应、整段优化路径等变。这四项不是同一件事。

|方法|实际读到的对象|在此特定变换下的判断|完整训练是否已验证|
|---|---|---|---|
|AdaLoRA|作者实现update_ipt：abs(p*p.grad)，然后EMA及分量组合|相应参数/梯度及历史状态映射时评分乘积不变。不能笼统当作raw-gradient反例|否；正交正则、优化器和初始化另论|
|EVA|官方PEFT的输入SVD解释方差比例及全局排序|Q/K输入不变，其分配输入信号不变|否；不外推到一般坐标变换|
|LoRA-GA|作者实现给定rank的梯度SVD初始化|正标量不改奇异子空间；常见unit/stable/gd选项没有按sigma大小做跨矩阵rank分配|否；固定rank初始化不能冒充分配算法|
|GEAR|论文公式5的normal-gradient方向单位成本分数|候选评分可改变，见下面推导；有候选接受门，最终rank尚不能下结论|否；未复跑作者代码或真实LLM|

代码依据：[AdaLoRA](https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/loralib/loralib/adalora.py)、[EVA官方PEFT](https://github.com/huggingface/peft/blob/af1ccdf262b9fe1ad8f0861ae10113a851ee5cbd/src/peft/tuners/lora/eva.py#L302)、[LoRA-GA](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py#L171)。LoRA-GA的正确论文编号是[2407.05000](https://arxiv.org/abs/2407.05000)，不是2406.08474；后者是另一篇论文，已排除。

## GEAR确有一个具体的评分层面问题

这里指[From What Is Used to What Is Missing: Rethinking Dynamic Rank in LoRA](https://arxiv.org/html/2611.07349v1#S3.SS3)，不是同名KV-cache压缩工作。论文公式5跨模块按normal-gradient奇异值平方除新增参数数排序；公式6之后还会用共享训练微批次验证候选扩张。这次阅读没有定位到作者实现链接，不把“未找到”写成“没有代码”。

下面是我们的推论。若已有适配状态也作X_Q'=cX_Q、X_K'=X_K/c的对应映射，正标量不会改变其左右奇异子空间，所以normal投影算子相同。于是G_N,Q'=G_N,Q/c，G_N,K'=cG_N,K，公式5的分数分别乘1/c²与c²，跨模块候选排序可能改变。

合成代数检查用3×3矩阵、相同单rank成本6，c=1时Q/K候选分数1/6与4/6，首选K；c=1/8时变为64/6与(4/64)/6，首选Q。双侧相对阻尼代理在这些相应缩放下误差0。[脚本](../scripts/probe_b_route_scores.py)和[输出](b_route_score_probe_20261122.json)明确标为synthetic_algebra_only。这是公式性质的数值校验，不是GEAR端到端复现，不证明其接受门后的rank改变或微调效果变差。

## 为什么仍不能宣称新颖性已经成立

K-FAC在[Martens与Grosse 2015的§4](https://proceedings.mlr.press/v37/martens15.pdf)已经讨论通过输入/输出度量获得重参数化不变性，以及等价初始化、无阻尼或可忽略阻尼等限制。LoRA的[对角fractional K-FAC/LoRSum](https://arxiv.org/abs/2602.16456v1)和[梯度—自然梯度初始化族ULoRA](https://arxiv.org/abs/2607.26247v1)也已出现。后两项本次只核查一手摘要，足以提示近邻风险，不足以判定算法等价。

本项目实测仅证明模块标量Q/K变换下，双侧经验对角量加相对阻尼可保持当前谱/分配。对角近似不等于完整K-FAC；含均值相对阻尼的对角矩阵不保证任意逐维缩放、旋转或一般基变换下不变。评分恢复一致性也没有自动带来C/R标签的下游优势。因此“首次曲率白化LoRA”或“普遍坐标不变的rank分配”都超出证据。

## 选路与后续门

已有[GRPO梯度分配负结果](https://arxiv.org/abs/2605.07366v2)也提醒梯度重要性与有限步容量需求未必一致；该工作只报告单seed/单任务，和本项目的变宽结构、10基础、随机标签控制不同，不能互相代替证据。

当前保留的研究材料是：可达整数预算造成的对照陷阱、非退化且容量响应明显时的真实C/R零优势、必要的一致性测试，以及代理预测力与标签价值的分离。可据此形成负结果/方法论稿件的证据骨架，尚未撰写论文或承诺投稿级别。

不变性方向若要单独重启，应依次满足：找到可复现的现有真实rank决策变化；排除接受门、初始化和优化器的混杂；在真实LLM及独立任务中证明该变化损害效果且修复有用；完成与上述近邻工作的算法及理论区别核查。当前仅通过候选评分层面的代数筛查，尚不启动大模型下载、租卡或性能训练。
