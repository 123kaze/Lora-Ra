# CarryRank-LoRA 最小机制实验

第二轮诊断已完成：[复核结论与证据](reports/diagnostic_review_20261122.md)、[109条轨迹的完整CSV](results/diagnostic_20261122_02/diagnostic.csv)、[逐基础汇总](results/diagnostic_20261122_02/by_base_summary.csv)。输入尺度与rank更新尺度的混杂确实存在，C/R结构优势仍未证实；原始结果保持不变。

正式第二轮入口：`./scripts/run_diagnostic.ps1 -Output runs/<new-run-id>`。无需torch的结果核验：`python scripts/verify_phase2_evidence.py results/diagnostic_20261122_02`。

研究方案见[研究方案.md](研究方案.md)，原文保留。第一轮从头预训练小型变宽Transformer，检验扩张层C/R适配及梯度谱分配；不实现K-FAC或V2，不预先假定R更重要。

第二轮已完成：[109条诊断轨迹与证据](results/diagnostic_20261122_02/README.md)、[针对Q被舍弃的解读](reports/diagnostic_20261122.md)。新增两个独立预训练种子、小预算/输入归一化/Q保留/缩放与lr对照。Q/K保持输出不变的重标定可使Q预算从0变为3008，暴露当前代理的参数尺度依赖；不据此宣称新的LoRA方法已获验证。

首轮已完成：[结果与限制](reports/pilot_20261122.md)。10项测试、18个主对照及6个非零rank诊断完成；工程通过，梯度分配未稳定胜过均分/普通LoRA，暂不扩展曲率。

**实际结果入口：[证据包索引](results/pilot_20261122_01/README.md)**。
直接查看：[main.csv](results/pilot_20261122_01/main.csv)、[spectra.csv](results/pilot_20261122_01/spectra.csv)、[sweep.csv](results/pilot_20261122_01/sweep.csv)、[summary.csv](results/pilot_20261122_01/summary.csv)、[完整报告](results/pilot_20261122_01/report.md)。
生成图：[comparison](figures/pilot_20261122_01/comparison.png)、[sensitivity](figures/pilot_20261122_01/sensitivity.png)、[prediction](figures/pilot_20261122_01/prediction.png)。这些是实测文件，不是图表程序或占位数据。

## 结构

```text
src/carryrank/    模型、可合并LoRA、精确背包、数据、实验与报告
configs/         pilot.json正式初筛 / smoke.json流程检查
scripts/         Windows现有Conda与WSL入口
tests/           影响实验可信度的正确性测试
plan/            协议、阶段门、进度与方法证据对应
docs/            来源、环境与注意事项
data/            原始语料/切分散列（不进Git）
runs/<run-id>/   完整运行工作目录，保留检查点和原始输出
results/<id>/    供仓库审阅的CSV、日志、审计、源码快照和哈希清单
figures/<id>/    实测生成的PNG/SVG
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

## 结果交付与独立核验

实验完成后必须提供小型证据包，不能只引用本机运行目录。Windows入口的`all/report`阶段会自动导出；其他入口手动运行：

```powershell
python scripts/export_results.py --run runs/pilot_20261122_01
python scripts/export_results.py --verify results/pilot_20261122_01
```

导出不重训、不改原始run；复制实测CSV、日志、配置与源码快照，图保存在`figures/<id>/`。`artifact-manifest.json`逐文件列出字节数、SHA-256和原路径。核验只需Python标准库，重算均值/样本std、对照原始完成日志，并从谱重算rank预测效用；不依赖Conda、CUDA或`runs/`。完整检查点重载仍需原始tensor文件。
