# B路线真实小模型补充协议v1

在下载权重、查看新训练指标或分配前固定。代码经过独立smoke后由正式run记录源码/配置/协议SHA；smoke不用于调优结论。旧方案、旧CI与B路线判定不改。

## 范围与问题

GPT-2标准预训练checkpoint（约124M、12层×768、融合QKV）和两个GLUE任务SST-2/MRPC。只适配所有12个attn.c_attn；每个矩阵768→2304，单rank3072参数。该同形状目标集使rank预算等于因子参数预算，避免AdaLoRA全局rank预算在不同形状矩阵上变成不同实际参数。验证的是标准模型上的层间容量分配，不是C/R，不代表所有Q/K/V/O/MLP目标集。

新假设H1：一次性双侧谱分配相对均衡LoRA有更高的验证标签NLL。新机制假设H2：在相同代理和预算下给所有层保留最低rank，可能改善相对集中的分配。代理与分散分配的比较不能单独证明“集中程度是唯一原因”；AdaLoRA动态优化也不能简化为一次性谱分配。

## 固定资产与切分

模型openai-community/gpt2 revision 607a30d783dfa663caf39e06633721c8d4cfcd7e；数据nyu-mll/glue revision bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c。只下载SST2和MRPC官方train/validation，不读取test。所有文件SHA和记录数保存。

split_seed=20261122，从官方train固定排列取256条作tune，其余SST2取4096条训练、MRPC保留全部余项。官方validation是最终评价集，不参与LR、rank、步数或模型选择。每个任务只使用固定训练前64条作校准。各训练seed控制同一批次序列和适配器随机性；只有一个公开预训练checkpoint，不称5个独立基础。

固定generative verbalizer：SST2 `Review: {sentence}\nSentiment:` → ` negative`/` positive`；MRPC `Sentence 1: {sentence1}\nSentence 2: {sentence2}\nParaphrase:` → ` no`/` yes`。各标签必须是GPT-2单token，否则规格错误并停止；不按性能换标签。保留prompt末尾最多128token、左填EOS、显式position_ids。仅预测下一个标签token，不训练语言模型的所有prompt token，也不训练embedding/lm_head。主指标为完整词表的标签NLL；分类仅在两个预定标签logit间取最大。

## 方法与预算

平均rank4，总rank48，固定LoRA因子参数147456，cap8。

1. UniformLoRA：12层全rank4。static use_rslora=True、alpha=2，其scale=1，等价于普通固定rank4/alpha4的LoRA。
2. TwoSidedSpectral：校准的M=(diag(Hout)+0.01mean(Hout))^(-1/2) G (diag(Hin)+0.01mean(Hin))^(-1/2)，FP32校准前反向，二阶矩以双精度累加后保存FP32；M构造和CPU NumPy SVD使用FP64取前8项，收益前缀sum(sigma²)/2；精确总rank48的DP，无组切分。
3. CoverageFloor：同一谱、预算、cap及初始化族，限制每层rank≥2后精确DP分配剩余容量。这是对覆盖率的干预，不新增校准信号。
4. AdaLoRA：官方PEFT实现，init_r8、target_r4、tinit20/tfinal40/deltaT10，orth_reg_weight0.5，lora_alpha8；在optimizer.step之后、zero_grad之前按实际optimizer步调用update_and_allocate。正交项必须被加入总loss，不能因手工分类损失漏掉。

前三个固定方法使用相同rsLoRA缩放alpha/sqrt(r)、A随机/B零初始化和训练设置。AdaLoRA使用其标准SVD参数化、初始化、正交正则与固定初始化rank分母；不伪称它具有完全相同的缩放或整个训练过程同参数数。

AdaLoRA最终有效因子参数按非屏蔽方向计数应为147456，另有48个E标量，实际有效147504（多0.0326%）；训练期保留init_r8因子及E，物理可训练295008，约2倍。逐步记录有效rank、峰值/平均因子参数、物理可训练参数和时间。比较匹配的是最终有效rank容量，不是峰值参数或总计算量。禁止用dead parameter补齐、把静态mask后仍存储的因子称作物理压缩。

## 优化与选择

全部200个optimizer步，microbatch8×accum2，有效batch16；AdamW，betas(0.9,0.999)、WD0、clip1，20步warmup后cosine。禁用基础模型和adapter dropout；冻结基础BF16、adapter FP32、autocast BF16，优先不启用checkpointing以减少重复计算。只计算最后预测位置的词表投影；必须与标准完整前向的同一位置及loss作数值核对。没有隐藏状态缓存来跨训练步骤复用受训练影响的层。

每方法/任务先用独立tuning_seed0，在LR{1e-4,3e-4,1e-3}各训练一次，仅用tune集200步NLL选择最低者（完全同分取较小LR），然后锁定选择。所有正式seed11/22/33/44/55使用该方法自己的选定LR，5个seed不能回流调参。共24条调参+40条正式训练；最终步200为唯一主时间点，不挑最好step。smoke单列。

方法特定LR可降低“集中分配只是LR不合适”的混杂；仍不是穷尽调优，也不是所有seed的最佳性能上界。若两分配恰相同可明确别名复用，但不删除不利样本或换预算。若代理与Uniform TV<0.10，标为近退化比较，不据此证明集中假设；不事后改预算救结果。

## 指标与判定

每任务主对比delta = NLL_Uniform − NLL_Spectral，负值表示谱方法更差。5个配对adapter/data seed，共享重采样索引，200000次percentile bootstrap、95%边际CI；它度量本固定checkpoint上的训练随机性，不是基础预训练波动。只有两个任务均CI上界<0，才写“两任务均检出谱方法NLL劣势”；否则完整报告混合/不确定结果。无总体等效性或后验概率解释。

Spectral − CoverageFloor（正值有利于覆盖下限）和Uniform − AdaLoRA（正值有利于AdaLoRA）为次要比较，全部报告均值/配对差/CI；主任务分类指标SST2 accuracy、MRPC F1及accuracy同时报告，不能用NLL代替任务能力胜负。分类方向与NLL不一致时明确展示。记录实际参数分布TV、HHI、非零层数、校准耗时/显存、训练时间；相关性不作因果结论。

## 审计与交付

每单位保存实际配置、方法/seed/LR、rank及mask历史、批次SHA、训练曲线、逐样本预测、真实adapter .pt、冻结基础散列、源码/资产/协议身份。每个正式adapter独立重载复算NLL/accuracy/F1，检查budget、zero-init、冻结和FP32合并误差。数值核对临时转FP32，同位置完整/仅末位logit最大误差须<5e-4，CE差须≤2×logit最大误差+1e-6（交叉熵关于logit无穷范数的解析Lipschitz界），适配模块与FP32合并误差<5e-4；还原原精度后基础SHA须一致。初始smoke发现不同形状FP32 GEMM的CE差5.82e-5略大于原5e-5经验阈值；在正式训练前改用上述解析界，保留失败日志，不改主实验统计门。合并验证可以检查真实权重增量对输出的影响；不把BF16基础权重直接合并的舍入误差混入FP32一致性阈值。

results/导出真实CSV/NPZ/日志/索引/源快照，figures/输出实测图，报告区分调参/正式/合成检查。纯NumPy核验从逐样本logprob和预测重算指标、配对CI与预算；不安装torch也能审核。大权重和adapter .pt只保留data/raw与runs，不上传。若工程失败记录失败，不以计划或smoke当结果。GEAR待作者代码，不在本轮扩展。
