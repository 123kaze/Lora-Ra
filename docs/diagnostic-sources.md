# 第二轮诊断的参考与解释边界

- rsLoRA原文：[A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)。本轮使用`sqrt(3/r)`，固定alpha=sqrt(3)，使rank3与旧版scale=1相同。它是受控缩放对照，不保证Adam下所有rank的真实步长相同；实际delta范数另行记录。
- [Martens与Grosse，K-FAC原文](https://proceedings.mlr.press/v37/martens15.html)讨论的是Fisher近似。当前只用`diag(E[x²])`与`H_out=I`，不把它称为完整K-FAC、精确Hessian或因果解释。
- C/R预测比例更接近1不是正确性的判据；应看归一化后的预测是否与独立实际收益更一致。
- Uniform优于Gradient不能单独证明Q有益；同时变化了V的rank、分组与初始化路径。本轮QFloor以同一代理/同预算施加最少Q容量约束，测量Q与V容量交换的效果。
- 预训练随机种子是独立重复层级；同一基础上的多个adapter seed先聚合，再报告基础之间的配对差，不能把它们当作更多独立基础样本。
- 首轮门槛未通过仅约束该次设定的主张。第二轮是依据首轮结果提出的前瞻配置、事后研究方向，不能将验证诊断冒充新的独立确认性测试。
