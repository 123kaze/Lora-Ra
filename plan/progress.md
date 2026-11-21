# 进度

2026-11-22：读取研究方案；工作区只有方案及编辑器历史，无既有实现。

- 官方仓库检查：commit `69dde8143d9a7912de353b57093d36d8788070d4`；README无权重链接；Issue #1仍开放，最近回复未提供权重。
- 环境：Windows 11 / WSL Ubuntu 24.04 / Python 3.12，RTX4050 Laptop 6GB，内存约32GB。
- 用户授权通过WSL运行、退出非必要显存程序。已退出NVIDIA Overlay与PhoneExperienceHost，保留编辑器/聊天/桌面；显存约1460→696MiB。
- WSL缺少python3.12-venv，正在安装最小环境支持；不修改驱动或全局Python依赖。
- D0/D1/D2已建立。接下来：隔离依赖、正确性测试、smoke、预训练与正式pilot。

用户提醒已有Conda后，检查全部登记环境：`scTPF`为torch2.9.1+cpu；`LangChain_agent`为Python3.11.14/torch2.14.1+cu126，RTX4050 CUDA矩阵与BF16可用。切换直接复用后者，不修改现有依赖；unittest和NumPy实现验证/相关统计，免装scipy/pytest。WSL仅补venv组件，pip在解析CUDA依赖时下载超时退出，无残留安装进程。训练尚未开始。

## 实测进展

- `python -m unittest discover -s tests -v`：10项通过（11.197s）。覆盖旁路历史/梯度、因果性、输入列掩码、零初始化、冻结与合并、精确背包对照穷举、预算可达性、切分边界及秩统计。
- `runs/smoke_20261122_01`：完整流程退出码0，12项运行审计通过，最大FP32合并输出误差`7.152557373046875e-07`。此run名称明确标为smoke，预训练/机制研究门保持false，不纳入正式结论。
- 数据下载完成：1,115,394字节，SHA256=`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`；切分散列已写入manifest。
- `runs/pilot_20261122_01`正式运行中，1,439,296参数；源验证loss从5.55272下降到step400的1.74594；尚未得出C/R适配结论。

## 正式pilot完成

- 入口退出码0。预训练1200步、84.51s，源验证5.55272→1.57062；目标基础验证1.77083，质量门通过。
- 6方法×3适配种子全部完成，严格3840参数；另有最终V投影C/R各rank2/4/8共6个诊断。使用固定末步结果，没有按测试选模。
- 验证loss均值：Equal1.66550、Uniform1.66563、C-only1.66715、Gradient1.66903、Shuffled1.67183、R-only1.67273。梯度方法相对Equal平均差0.00352（更差），相对Shuffled改善0.00281。
- Spearman=0.542857（6点）；组内C=-0.5、R=1.0。保留混合证据，机制初筛未通过，不继续扩展K-FAC/动态rank/V2。
- 正式run12项工程检查全部通过，最大合并误差9.53674e-06。独立重载18个适配器，验证loss误差0；全部校准效用重新SVD误差0。
- 三幅PNG/SVG已生成并目视检查。摘要在`reports/pilot_20261122.md`，完整证据在`runs/pilot_20261122_01/`。
- 原始方案SHA256保持`ef6fc54d197af6f3d3f9d612a40c843a90c9f8482362ef2008f270847d3e3cb2`。
