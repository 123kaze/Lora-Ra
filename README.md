# CarryRank-LoRA 最小机制实验

研究方案见[研究方案.md](研究方案.md)，原文保留。第一轮从头预训练小型变宽Transformer，检验扩张层C/R适配及梯度谱分配；不实现K-FAC或V2，不预先假定R更重要。

首轮已完成：[结果与限制](reports/pilot_20261122.md)。10项测试、18个主对照及6个非零rank诊断完成；工程通过，梯度分配未稳定胜过均分/普通LoRA，暂不扩展曲率。

## 结构

```text
src/carryrank/    模型、可合并LoRA、精确背包、数据、实验与报告
configs/         pilot.json正式初筛 / smoke.json流程检查
scripts/         Windows现有Conda与WSL入口
tests/           影响实验可信度的正确性测试
plan/            协议、阶段门、进度与方法证据对应
docs/            来源、环境与注意事项
data/            原始语料/切分散列（不进Git）
runs/<run-id>/   检查点、原始日志、CSV、图表与审计（不进Git）
reports/         可读结果入口
```

## 当前已验证环境

Windows现有Conda：`E:\envs\LangChain_agent\python.exe`，Python3.11.14、torch2.14.1+cu126、RTX4050 Laptop6GB，CUDA矩阵计算和BF16已实测。直接设置本进程`PYTHONPATH=src`，不安装/升级该环境依赖。

```powershell
$env:PYTHONPATH = "$PWD\src"
& 'E:\envs\LangChain_agent\python.exe' -m unittest discover -s tests -v
./scripts/run.ps1 -Config configs/smoke.json -Output runs/smoke_20261122
./scripts/run.ps1 -Config configs/pilot.json -Output runs/pilot_20261122
```

恢复已完成的阶段/单个适配运行时加`-Resume`，会严格核对配置、代码、数据与环境。被中断的未完成单元从头确定性初始化重跑，旧attempt日志保留，不保证训练内逐步恢复或跨硬件位级复现。不要修改代码后复用旧run。

WSL为备用：Ubuntu24.04已补`python3.12-venv`；此前CUDA依赖下载超时，环境未装完。需要时再运行`scripts/setup_wsl.sh`。当前实验不依赖WSL。

## 实验边界

预设协议见[plan/experiment-protocol.md](plan/experiment-protocol.md)。6种方法×3个适配种子，每种3840个可训练参数、同基础checkpoint、相同训练batch顺序。先源训练集预训练，再在无交叠的目标训练段适配；校准不接触val/test。记录工程检查与机制初筛的独立判定。

查看`runs/<run-id>/report.md`与`verdict.json`，先确认`verification.json`。最终CSV和图直接来自真实运行。注意小模型、单语料与共享预训练种子的限制；不把流程跑通当作方法有效。
