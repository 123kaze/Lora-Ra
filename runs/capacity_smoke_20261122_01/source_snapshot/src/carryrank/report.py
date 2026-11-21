"""Audited aggregation for measured outputs. No invented or placeholder metrics."""
from __future__ import annotations

import csv
import json
import math
import runpy
from pathlib import Path

import numpy as np


def ranks(values):
    values = np.asarray(values, dtype=float)
    result = np.empty(len(values), dtype=float)
    for value in np.unique(values):
        positions = np.flatnonzero(values == value)
        result[positions] = (np.sum(values < value) + 1 + np.sum(values <= value)) / 2
    return result


def spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def build_report(output):
    from .experiment import METHODS, ROOT, code_manifest, digest
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    pretrain = json.loads((output / "pretrain.json").read_text(encoding="utf-8"))
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    main = read_csv(output / "main.csv")
    sweep = read_csv(output / "sweep.csv")
    expected = {(method, seed) for method in METHODS for seed in config["adapter"]["seeds"]}
    actual = [(r["method"], int(r["seed"])) for r in main]
    if len(actual) != len(expected) or set(actual) != expected:
        raise AssertionError("Incomplete or duplicated main comparison; cannot issue final report")
    base_hash = pretrain["base_sha256"]
    checks = {
        "complete_main_comparison": True,
        "exact_parameter_budget": all(int(r["params"]) == config["adapter"]["budget"] for r in main),
        "zero_delta_initialization": all(float(r["initial_max_error"]) == 0 for r in main),
        "frozen_base": all(r["frozen_base_sha256"] == base_hash for r in main),
        "merge_equivalence": all(float(r["merge_max_error"]) <= 1e-4 for r in main),
        "identical_batches_per_seed": all(len({r["data_trace_sha256"] for r in main if int(r["seed"]) == s}) == 1
                                          for s in config["adapter"]["seeds"]),
        "calibration_did_not_update_weights": calibration["weights_unchanged"],
        "calibration_base_matches": calibration["base_sha256"] == base_hash,
        "code_matches_run_identity": identity["code"] == code_manifest(),
        "proposal_unchanged": digest((ROOT / "研究方案.md").read_bytes()) == json.loads((output / "environment.json").read_text(encoding="utf-8"))["original_proposal_sha256"],
        "fixed_adapter_training_steps": all(int(r["steps"]) == config["adapter"]["steps"] for r in main),
        "complete_sweep": len(sweep) == 1 + 2 * sum(k != 0 for k in config["sweep"]["ranks"]),
    }
    engineering = all(checks.values())
    write_json(output / "verification.json", {"checks": checks, "engineering_pass": engineering,
                                               "max_merge_error": max(float(r["merge_max_error"]) for r in main)})
    if not engineering:
        raise AssertionError(f"Run verification failed: {checks}")
    summary = []
    n = len(config["adapter"]["seeds"])
    for method in METHODS:
        rows = [r for r in main if r["method"] == method]
        entry = {"method": method, "seeds": n, "params": int(rows[0]["params"])}
        for metric in ("val_loss", "test_loss", "val_bits_per_byte", "test_bits_per_byte", "val_accuracy", "test_accuracy",
                       "training_seconds", "peak_allocated_mb", "unmerged_latency_ms", "merged_latency_ms"):
            values = [float(r[metric]) for r in rows]
            entry[metric + "_mean"] = float(np.mean(values))
            entry[metric + "_std"] = float(np.std(values, ddof=1)) if n > 1 else 0.0
        entry["val_improvement"] = pretrain["target_val"]["loss"] - entry["val_loss_mean"]
        summary.append(entry)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    by_method = {r["method"]: r for r in summary}
    comparisons = {}
    for reference in ("Equal", "Shuffled", "Uniform", "C-only", "R-only"):
        differences = []
        for seed in config["adapter"]["seeds"]:
            pair = {r["method"]: float(r["val_loss"]) for r in main if int(r["seed"]) == seed}
            differences.append(pair[reference] - pair["Gradient"])
        comparisons[reference] = {"paired_val_loss_advantage_of_gradient": differences,
                                  "mean_advantage": float(np.mean(differences)),
                                  "all_positive": all(d > 0 for d in differences)}
    nonzero = [r for r in sweep if int(r["rank"]) > 0]
    x = np.array([float(r["predicted_utility"]) for r in nonzero])
    y = np.array([float(r["actual_val_improvement"]) for r in nonzero])
    correlation = spearman(x, y)
    slope = max(0.0, float(np.dot(x, y) / np.dot(x, x))) if np.dot(x, x) > 0 else 0.0
    prediction = {"nonzero_points": len(nonzero), "spearman": correlation,
                  "raw_mae": float(np.mean(np.abs(x - y))), "nonnegative_scale_fit": slope,
                  "rescaled_mae_in_sample": float(np.mean(np.abs(slope * x - y))),
                  "scale_fit_is_descriptive_not_held_out": True,
                  "within_label_spearman": {label: spearman(
                      [float(r["predicted_utility"]) for r in nonzero if r["label"] == label],
                      [float(r["actual_val_improvement"]) for r in nonzero if r["label"] == label]) for label in ("C", "R")}}
    screening = bool(pretrain["pretraining_gate"] and n >= 3 and correlation is not None and correlation > 0.3 and
                     all(comparisons[ref]["all_positive"] and comparisons[ref]["mean_advantage"] >= 0.01
                         for ref in ("Equal", "Shuffled")))
    verdict = {"engineering_pass": engineering, "pretraining_gate": pretrain["pretraining_gate"],
               "mechanism_screening_pass": screening, "comparisons": comparisons, "prediction": prediction,
               "decision": "expand_validation_first" if screening else "stop_complexity_and_retain_negative_or_mixed_evidence",
               "not_publication_evidence": True}
    write_json(output / "verdict.json", verdict)
    lines = [f"# CarryRank-LoRA 最小实验：{output.name}", "",
             "本报告只使用该run的真实测量；是小模型机制初筛，不是MIT模型复现或论文有效性结论。", "",
             f"工程检查：**{'通过' if engineering else '未通过'}**；预训练质量门：**{'通过' if pretrain['pretraining_gate'] else '未通过'}**；预设机制初筛：**{'通过' if screening else '未通过'}**。", "",
             f"基础模型 {pretrain['parameters']:,} 参数，宽度 `{config['model']['widths']}`，从头预训练 {pretrain['total_steps']} 步，源验证最优步数 {pretrain['selected_step']}。",
             f"源验证loss：随机初始化 {pretrain['initial_source_val']['loss']:.5f} → {pretrain['source_val']['loss']:.5f} nat/byte；冻结基础模型目标验证loss为 {pretrain['target_val']['loss']:.5f}。",
             f"预训练耗时 {pretrain['time_s']:.1f}s；校准 {calibration['calibration_seconds']:.2f}s，{calibration['calibration_bytes']:,} 个训练字节。", "",
             f"主要对照都训练 {config['adapter']['steps']} 步、{config['adapter']['budget']:,} 个参数；{n}个适配种子，共用一个预训练模型。数值为均值±样本标准差。", "",
             "| 方法 | Val loss ↓ | Test loss ↓ | Val改善 ↑ | 训练秒 | 峰值MB |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['val_loss_mean']:.5f} ± {row['val_loss_std']:.5f} | {row['test_loss_mean']:.5f} ± {row['test_loss_std']:.5f} | {row['val_improvement']:.5f} | {row['training_seconds_mean']:.1f} | {row['peak_allocated_mb_mean']:.1f} |")
    lines += ["", "Gradient相对对照的配对验证loss优势（正数表示Gradient更好）：", "",
              "| 对照 | 三种子配对差 | 平均差 |", "|---|---|---:|"]
    for method, comparison in comparisons.items():
        diffs = ", ".join(f"{d:.5f}" for d in comparison["paired_val_loss_advantage_of_gradient"])
        lines.append(f"| {method} | {diffs} | {comparison['mean_advantage']:.5f} |")
    rho = f"{correlation:.4f}" if correlation is not None else "未定义（点数/方差不足）"
    lines += ["", f"预测诊断：排除共享rank0，{len(nonzero)}个点的Spearman={rho}；原始尺度MAE={prediction['raw_mae']:.6g}。",
              f"最佳非负缩放系数={slope:.6g}，缩放后MAE={prediction['rescaled_mae_in_sample']:.6g}（同批点拟合，只作描述）。",
              "相关系数混合rank与组别，不能单独证明跨组排序；单位曲率局部预测和180步Adam收益的尺度不一致属预期风险。", "",
              "继续判断：" + ("达到预设初筛门，下一轮先增加独立基础种子/分组种子/任务/模型规模，再决定是否加入曲率。" if screening else
                              "未达到预设结构优势门。本轮保留全部负/混合结果，暂不扩展K-FAC、动态rank或V2。"), "",
              "限制：仅一种小模型宽度、同语料后段适配、一个预训练种子、固定校准与随机分组种子、少量rank点；没有AdaLoRA/EVA/GEAR比较。测试集只用于最终固定检查点评估，不参与分配或选模。", "",
              "审计文件：`identity.json`、`data_manifest.json`、`environment.json`、`environment-freeze.txt`、`allocations.json`、`verification.json`、`verdict.json`；原始逐步日志在`logs/`。", "",
              "![Matched-budget comparison](figures/comparison.png)", "", "![Sensitivity](figures/sensitivity.png)", "", "![Prediction diagnostic](figures/prediction.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    namespace = runpy.run_path(str(ROOT / "figures/pilot/plot_results.py"))
    namespace["plot_run"](output)
    print(json.dumps(verdict, ensure_ascii=False), flush=True)

