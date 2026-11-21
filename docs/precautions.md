# 实验注意事项

1. `研究方案.md` 保留原文。代码与文档实现的是CarryRank梯度初版，不把尚未实现的K-FAC称为Hessian实验。
2. 模型从头预训练，保留旁路历史；不是裁剪等宽预训练权重。当前规模远小于MIT论文200M–2B，结果只能用于本机初筛。
3. 输入Q/V的列有隐藏维度意义；本轮不把attention头输出维度错误地当成C/R。MLP和O不在本轮适配范围。
4. 初始化B=0、缩放恒1；分组LoRA每rank成本是输出维度+组输入维度。必须看实际params而非平均rank。
5. calibration来自目标训练集，val/test不得进入谱或rank分配。共享基础checkpoint与训练batch顺序。三种子是adapter种子，不是三个独立预训练模型。
6. 随机同形状分组只是一个控制；不能消除所有随机划分方差。未通过结构优势门时保留负结果，不扩大复杂度。
7. PyTorch峰值allocated不包含桌面/驱动显存。WSL可见同一GPU；不要安装Linux显卡驱动覆盖Windows驱动。
8. 当前直接复用Windows Conda `E:\envs\LangChain_agent\python.exe`，CUDA/BF16已实测，不更改其依赖；使用`scripts/run.ps1`。WSL `~/.venvs/carryrank`只作备用，安装因下载超时未完成；不在Windows Conda base安装torch。
9. 每次新实验用新run-id。保留配置、代码散列、数据散列、checkpoint与日志；报告引用run-id。中断后用同配置resume，不覆写不同结果。
10. 当桌面软件需要保存未保存内容时不能强制退出。当前已获用户授权退出非必要显存软件；编辑器和当前会话始终保留。
