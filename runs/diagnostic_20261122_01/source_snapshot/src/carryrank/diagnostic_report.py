"""Report phase-2 evidence without retroactively changing the first protocol."""
import csv
import json
import statistics
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(output, phase):
    from .experiment import write_csv, write_json
    output = Path(output)
    main, sweep, concentration = [rows(output / f"{family}.csv") for family in ("main", "sweep", "concentration")]
    expected = len(phase["base_seeds"]) * len(phase["adapter_seeds"]) * len(phase["scales"]) * len(phase["main_methods"]) * len(phase["eval_steps"])
    if len(main) != expected:
        raise AssertionError(f"Incomplete main table: {len(main)} != {expected}")
    base_summary = []
    for base in phase["base_seeds"]:
        for method in phase["main_methods"]:
            for scale in phase["scales"]:
                for step in phase["eval_steps"]:
                    selected = [r for r in main if int(r["base_seed"]) == base and r["method"] == method and r["scale"] == scale and int(r["step"]) == step]
                    base_summary.append({"base_seed": base, "method": method, "scale": scale, "step": step,
                                         "val_loss": statistics.mean(float(r["val_loss"]) for r in selected),
                                         "val_improvement": statistics.mean(float(r["val_improvement"]) for r in selected)})
    write_csv(output / "base_summary.csv", base_summary)
    summary = []
    for method in phase["main_methods"]:
        for scale in phase["scales"]:
            for step in phase["eval_steps"]:
                selected = [r for r in base_summary if r["method"] == method and r["scale"] == scale and r["step"] == step]
                losses = [r["val_loss"] for r in selected]
                summary.append({"method": method, "scale": scale, "step": step, "base_seeds": len(losses),
                                "val_loss_mean": statistics.mean(losses), "between_base_sd": statistics.stdev(losses),
                                "val_improvement_mean": statistics.mean(r["val_improvement"] for r in selected)})
    write_csv(output / "summary.csv", summary)
    pairs = []
    for scale in phase["scales"]:
        for base in phase["base_seeds"]:
            values = {r["method"]: r["val_loss"] for r in base_summary if r["scale"] == scale and r["base_seed"] == base and r["step"] == phase["steps"]}
            for reference in ("Balanced", "Gradient", "DiagInputShuffled", "VOnlyBalanced"):
                pairs.append({"scale": scale, "base_seed": base, "reference": reference,
                              "diag_advantage": values[reference] - values["DiagInput"]})
    write_csv(output / "paired_differences.csv", pairs)
    configs = json.loads((output / "unit_configs.json").read_text())
    checks = {"main_complete": len(main) == expected, "main_exact_budget": all(int(r["params"]) == phase["budgets"]["small"] for r in main),
              "merge_checks": all(r["merge_error"] <= 1e-4 for r in configs), "three_independent_bases": len({r['base_sha256'] for r in main}) == 3,
              "identical_batch_order": all(len({r["data_trace"] for r in main if int(r["adapter_seed"]) == seed and int(r["step"]) == step}) == 1
                                           for seed in phase["adapter_seeds"] for step in phase["eval_steps"])}
    write_json(output / "verification.json", checks)
    if not all(checks.values()):
        raise AssertionError(checks)
    lines = [f"# 第二轮诊断：{output.name}", "", "本轮是诊断性验证，不修改首轮门槛，不使用目标测试集，不产生方法优越性的确认性结论。", "",
             "3个独立预训练基础，每个基础2个适配种子；先在每个基础内平均，再对三个基础汇总。下表±为基础间样本std。预算准确1024参数，固定600步末结果。", "",
             "| 方法 | constant | rs_ref3 |", "|---|---:|---:|"]
    for method in phase["main_methods"]:
        values = []
        for scale in phase["scales"]:
            row = next(r for r in summary if r["method"] == method and r["scale"] == scale and r["step"] == phase["steps"])
            values.append(f"{row['val_loss_mean']:.5f} ± {row['between_base_sd']:.5f}")
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines += ["", "DiagInput相对对照的配对优势（正数更好，基础内已平均两个适配种子）：", "",
              "| scale | reference | 三个基础的配对差 | 均值 |", "|---|---|---|---:|"]
    for scale in phase["scales"]:
        for reference in ("Balanced", "Gradient", "DiagInputShuffled", "VOnlyBalanced"):
            values = [r["diag_advantage"] for r in pairs if r["scale"] == scale and r["reference"] == reference]
            lines.append(f"| {scale} | {reference} | {', '.join(f'{v:.5f}' for v in values)} | {statistics.mean(values):.5f} |")
    lines += ["", "原3840预算、已知故障seed11的事后优化诊断（不用于公平择优）：", "",
              "| 方法 | scale | lr | val@180 | val@600 |", "|---|---|---:|---:|---:|"]
    for method in phase["concentration_methods"]:
        for scale, lr in (("constant", .003), ("rs_ref3", .003), ("constant", .001)):
            selected = [r for r in concentration if r["method"] == method and r["scale"] == scale and abs(float(r["lr"])-lr)<1e-10]
            values = {int(r["step"]): float(r["val_loss"]) for r in selected}
            lines.append(f"| {method} | {scale} | {lr} | {values[180]:.5f} | {values[600]:.5f} |")
    lines += ["", "输入二阶矩与原始/归一化效用见`input_scales.csv`；16条rank轨迹见`sweep.csv`；真实更新范数见各unit的`result.json`与逐步日志。", "",
              "输入白化只用了对角H_in，H_out=I；输入归一化后C/R比例接近1不自动代表正确。rank sweep只有基础2026与适配seed11，不能用于独立泛化结论。", "",
              "不设事后优势门。先看预算/步数/缩放/基础种子的交互和配对效应，再决定下一轮是否需要新任务；当前不扩展完整K-FAC、动态rank、V2。", "",
              "![Capacity](../../figures/" + output.name + "/capacity.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plot(output, sweep)


def plot(output, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, scale in zip(axes, ("constant", "rs_ref3")):
        for label, color in (("C", "#0077BB"), ("R", "#009988")):
            for step, linestyle in ((180, "--"), (600, "-")):
                selected = [r for r in sweep if r["scale"] == scale and r["method"].startswith(label+"_") and int(r["step"]) == step]
                selected.sort(key=lambda r: int(r["method"].split("rank")[1]))
                ax.plot([int(r["method"].split("rank")[1]) for r in selected], [float(r["val_improvement"]) for r in selected],
                        color=color, linestyle=linestyle, marker="o", label=f"{label}, {step} steps")
        ax.set(title=scale, xlabel="Rank")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Validation loss improvement (nat/byte)")
    fig.tight_layout()
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "capacity.png", dpi=450, bbox_inches="tight")
    fig.savefig(destination / "capacity.svg", bbox_inches="tight")
    plt.close(fig)
