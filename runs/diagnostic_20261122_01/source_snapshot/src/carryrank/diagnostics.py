"""Prospective phase-2 diagnostics; preserves phase-1 sources and outputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .allocation import allocate, equal_parameters
from .data import digest, random_batch
from .experiment import (ROOT, Experiment, base_digest, code_manifest, log, new_attempt,
                         read_json, seed_all, synchronize, write_csv, write_json)
from .lora import GroupedLinear, attach, merge_all


def scaling_for(rank, mode):
    if mode == "constant":
        return 1.0
    if mode == "rs_ref3":
        return math.sqrt(3.0 / rank)
    raise ValueError(mode)


def input_utility(gradient, moment, max_rank, damping=0.01):
    diagonal = moment + damping * moment.mean()
    if (diagonal <= 0).any():
        raise ValueError("Nonpositive input second moment")
    matrix = gradient / diagonal.sqrt()[None, :]
    sigma = torch.linalg.svdvals(matrix)[:max_rank]
    return [0.0] + (0.5 * sigma.square()).cumsum(0).tolist(), sigma.tolist()


def method_allocation(calibration, normalized, method, budget, cap):
    scheme = "shuffled" if "Shuffled" in method else "true"
    groups = calibration["groups"][scheme]
    if method == "VOnlyBalanced":
        groups = [g for g in groups if g["module"].endswith("v_proj")]
    if method in ("Balanced", "VOnlyBalanced"):
        ranks, _, _ = equal_parameters(groups, budget, cap)
    else:
        scores = normalized[scheme] if method.startswith("DiagInput") else calibration["utilities"][scheme]
        ranks, _, _ = allocate(groups, scores, budget, cap)
    if sum(g["cost"] * ranks.get(g["id"], 0) for g in groups) != budget:
        raise AssertionError("Budget is not exact")
    return groups, ranks


@torch.no_grad()
def update_measurement(model, probe, original_logits):
    total, reference = 0.0, 0.0
    per_module = {}
    for name, module, _ in model.targets():
        base = module.base if isinstance(module, GroupedLinear) else module
        reference += base.weight.float().square().sum().item()
        if isinstance(module, GroupedLinear):
            delta = torch.zeros_like(base.weight)
            for factor in module.factors:
                delta.index_add_(1, factor.indices, (factor.B @ factor.A) * factor.scaling)
            norm = delta.float().norm().item()
            total += norm * norm
            per_module[name] = norm
    model.eval()
    output = model(probe)
    return {"delta_frobenius": math.sqrt(total), "relative_delta": math.sqrt(total / reference),
            "probe_logit_rms_change": (output - original_logits).float().square().mean().sqrt().item(),
            "module_delta_norms": per_module}


class DiagnosticExperiment(Experiment):
    def input_calibration(self, damping):
        original = self.calibrate()
        destination = self.output / "input_calibration.pt"
        if destination.exists():
            return original, torch.load(destination, map_location="cpu", weights_only=True)
        model, checkpoint = self.load_base()
        model.eval()
        moments, counts, handles = {}, {}, []
        for name, linear, _ in model.targets():
            moments[name] = torch.zeros(linear.in_features, dtype=torch.float64)
            counts[name] = 0
            def capture(module, args, module_name=name):
                x = args[0].detach().reshape(-1, args[0].shape[-1])
                moments[module_name].add_(x.double().square().sum(0).cpu())
                counts[module_name] += x.shape[0]
            handles.append(linear.register_forward_pre_hook(capture))
        cfg = self.config["calibration"]
        generator = torch.Generator().manual_seed(cfg["seed"])
        with torch.no_grad():
            for _ in range(cfg["batches"]):
                x, _ = random_batch(self.splits["target_train"], cfg["batch_size"], self.model_config.seq_len, generator, self.device)
                model(x)
        for handle in handles:
            handle.remove()
        if base_digest(model) != checkpoint["sha256"]:
            raise AssertionError("Activation collection changed base")
        moments = {name: value / counts[name] for name, value in moments.items()}
        normalized, records, spectra = {}, [], []
        cap = self.config["adapter"]["max_rank"]
        for scheme, groups in original["groups"].items():
            normalized[scheme] = {}
            for group in groups:
                gid, name, indices = group["id"], group["module"], group["indices"]
                # One module-wide damping, identical for true and shuffled partitions.
                moment = moments[name][indices]
                diagonal = moment + damping * moments[name].mean()
                matrix = original["gradients"][name][:, indices]
                singular = torch.linalg.svdvals(matrix / diagonal.sqrt()[None, :])[:cap]
                normalized[scheme][gid] = [0.0] + (0.5 * singular.square()).cumsum(0).tolist()
                raw = original["utilities"][scheme][gid]
                records.append({"scheme": scheme, "group": gid, "label": group["label"], "layer": group["layer"],
                                "m": group["m"], "n": group["n"], "input_second_moment_mean": moment.mean().item(),
                                "input_second_moment_min": moment.min().item(), "input_second_moment_max": moment.max().item(),
                                "raw_u2": raw[min(2, len(raw)-1)], "diag_u2": normalized[scheme][gid][min(2, len(raw)-1)]})
                for rank, value in enumerate(singular.tolist(), 1):
                    spectra.append({"scheme": scheme, "group": gid, "rank": rank, "sigma": value,
                                    "utility": value * value / 2, "cost": group["cost"]})
        result = {"utilities": normalized, "input_moments": moments, "damping": damping,
                  "base_sha256": checkpoint["sha256"], "samples_per_module": counts}
        torch.save(result, destination)
        write_csv(self.output / "input_scales.csv", records)
        write_csv(self.output / "input_spectra.csv", spectra)
        return original, result

    def diagnostic_adapt(self, label, family, method, seed, groups, ranks, scale_mode, lr, phase):
        unit = self.output / "units" / label
        if (unit / "result.json").exists():
            return read_json(unit / "result.json")
        unit.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        model, checkpoint = self.load_base()
        baseline = checkpoint["summary"]["target_val"]["loss"]
        probe = self.splits["target_train"][:self.model_config.seq_len].unsqueeze(0).to(self.device)
        model.eval()
        with torch.no_grad():
            initial_logits = model(probe)
        seed_all(seed)
        params = attach(model, groups, ranks, scaling=1.0)
        for module in model.modules():
            if isinstance(module, GroupedLinear):
                for factor in module.factors:
                    factor.scaling = scaling_for(factor.A.shape[0], scale_mode)
        with torch.no_grad():
            initial_error = (model(probe) - initial_logits).abs().max().item()
        if initial_error != 0:
            raise AssertionError("Initialization is not zero delta")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        generator = torch.Generator().manual_seed(seed + 100000)
        trace, curve, update_records = hashlib.sha256(), [], []
        rolling = []
        synchronize(self.device)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        with new_attempt(self.output, label) as handle:
            for step in range(1, phase["steps"] + 1):
                model.train()
                x, y = random_batch(self.splits["target_train"], self.config["adapter"]["batch_size"], self.model_config.seq_len, generator, self.device)
                trace.update(x.detach().cpu().contiguous().numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {label} at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                rolling.append(loss.item())
                if step in (1, 60, 180, phase["steps"]):
                    measurement = update_measurement(model, probe, initial_logits)
                    update_records.append({"step": step, **measurement})
                    log(handle, {"stage": "update_measurement", "unit": label, "step": step,
                                 "interval_train_loss": float(np.mean(rolling)), **measurement})
                    rolling = []
                if step in phase["eval_steps"]:
                    value = self.evaluate(model, "target_val")
                    row = {"base_seed": self.config["pretrain"]["seed"], "adapter_seed": seed, "family": family,
                           "method": method, "scale": scale_mode, "lr": lr, "step": step, "params": params,
                           "val_loss": value["loss"], "val_improvement": baseline - value["loss"],
                           "val_accuracy": value["accuracy"], "data_trace": trace.hexdigest(),
                           "base_sha256": checkpoint["sha256"], "elapsed_s": time.perf_counter() - started}
                    curve.append(row)
                    log(handle, {"stage": "validation", "unit": label, **row})
                    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".factors." in k}
                    torch.save({"adapter": state, "groups": groups, "ranks": ranks, "scale_mode": scale_mode,
                                "base_sha256": checkpoint["sha256"]}, unit / f"adapter_step{step}.pt")
            if base_digest(model) != checkpoint["sha256"]:
                raise AssertionError("Frozen base changed")
            model.eval()
            with torch.no_grad():
                expected = model(probe)
            merge_all(model)
            with torch.no_grad():
                merge_error = (model(probe) - expected).abs().max().item()
            if merge_error > 1e-4:
                raise AssertionError(f"Merge error: {merge_error}")
            result = {"curve": curve, "updates": update_records, "groups": groups, "ranks": ranks,
                      "initial_error": initial_error, "merge_error": merge_error, "base_unchanged": True,
                      "peak_allocated_mb": self.peak_memory(), "completed": True}
            write_json(unit / "result.json", result)
            log(handle, {"stage": "unit_complete", "unit": label, "merge_error": merge_error})
        del model, optimizer
        torch.cuda.empty_cache()
        return result


def export_diagnostic(output):
    destination = ROOT / "results" / output.name
    files = []
    for source in sorted(output.rglob("*")):
        if source.is_file() and source.suffix in (".csv", ".json", ".jsonl", ".md", ".txt", ".py"):
            target = destination / source.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Export already differs: {target}")
            shutil.copy2(source, target)
            files.append({"path": target.relative_to(ROOT).as_posix(), "sha256": digest(target.read_bytes()), "bytes": target.stat().st_size})
    for source in sorted((output / "figures").glob("*")):
        target = ROOT / "figures" / output.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append({"path": target.relative_to(ROOT).as_posix(), "sha256": digest(target.read_bytes()), "bytes": target.stat().st_size})
    write_json(destination / "artifact-manifest.json", {"run": output.name, "files": files})


def run(phase, output, resume=False, probe_only=False):
    parent = ROOT / phase["parent_run"]
    original = read_json(parent / "identity.json")
    for path, expected in original["code"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Parent source changed: {path}")
    identity = {"phase": phase, "source": code_manifest(), "parent_base_sha256": digest((parent / "base.pt").read_bytes()),
                "parent_main_sha256": digest((parent / "main.csv").read_bytes()),
                "protocol_sha256": digest((ROOT / "plan/diagnostic-protocol.md").read_bytes())}
    if output.exists():
        if not resume or read_json(output / "identity.json") != identity:
            raise ValueError("Existing run/resume identity mismatch")
    else:
        output.mkdir(parents=True)
        write_json(output / "identity.json", identity)
        write_json(output / "config.json", phase)
        shutil.copy2(ROOT / "plan/diagnostic-protocol.md", output / "protocol.md")
        for relative, expected in identity["source"].items():
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    results, scales, configs = [], [], []
    for base_seed in phase["base_seeds"]:
        config = copy.deepcopy(original["config"])
        config["name"] = phase["name"]
        config["pretrain"]["seed"] = base_seed
        child = output / f"base{base_seed}"
        experiment = DiagnosticExperiment(config, child, resume=child.exists())
        if base_seed == original["config"]["pretrain"]["seed"] and not (child / "base.pt").exists():
            shutil.copy2(parent / "base.pt", child / "base.pt")
            shutil.copy2(parent / "pretrain.json", child / "pretrain.json")
            write_json(child / "parent_origin.json", {"parent": phase["parent_run"], "base_sha256": identity["parent_base_sha256"]})
        experiment.pretrain()
        if not read_json(child / "pretrain.json")["pretraining_gate"]:
            raise RuntimeError("Pretraining quality gate failed")
        calibration, normalized = experiment.input_calibration(phase["input_damping"])
        import csv
        with (child / "input_scales.csv").open(newline="") as handle:
            scales.extend({"base_seed": base_seed, **row} for row in csv.DictReader(handle))
        write_csv(output / "input_scales.csv", scales)
        if probe_only:
            break
        jobs = []
        for seed in phase["adapter_seeds"]:
            for scale in phase["scales"]:
                for method in phase["main_methods"]:
                    groups, ranks = method_allocation(calibration, normalized["utilities"], method, phase["budgets"]["small"], config["adapter"]["max_rank"])
                    label = f"main_{method}_{scale}_seed{seed}"
                    jobs.append((label, "main", method, seed, groups, ranks, scale, 0.003))
        if base_seed == 2026:
            for group in calibration["groups"]["true"]:
                if group["layer"] == 4 and group["module"].endswith("v_proj"):
                    for rank in phase["sweep_ranks"]:
                        for scale in phase["scales"]:
                            method = f"{group['label']}_rank{rank}"
                            jobs.append((f"sweep_{method}_{scale}", "sweep", method, 11, [group], {group["id"]: rank}, scale, 0.003))
            for method in phase["concentration_methods"]:
                groups, ranks = method_allocation(calibration, normalized["utilities"], method, phase["budgets"]["original"], config["adapter"]["max_rank"])
                for scale, lr in (("constant", 0.003), ("rs_ref3", 0.003), ("constant", 0.001)):
                    jobs.append((f"concentration_{method}_{scale}_lr{lr}", "concentration", method, 11, groups, ranks, scale, lr))
        for label, family, method, seed, groups, ranks, scale, lr in jobs:
            value = experiment.diagnostic_adapt(label, family, method, seed, groups, ranks, scale, lr, phase)
            results.extend(value["curve"])
            configs.append({"base_seed": base_seed, "unit": label, "family": family, "ranks": ranks,
                            "cost": sum(g["cost"] * ranks.get(g["id"], 0) for g in groups), "merge_error": value["merge_error"]})
            for subset in ("main", "sweep", "concentration"):
                write_csv(output / f"{subset}.csv", [r for r in results if r["family"] == subset])
            write_json(output / "unit_configs.json", configs)
    if not probe_only:
        report_diagnostic(output, phase)
        if digest((parent / "base.pt").read_bytes()) != identity["parent_base_sha256"] or digest((parent / "main.csv").read_bytes()) != identity["parent_main_sha256"]:
            raise AssertionError("Parent run changed")
        export_diagnostic(output)


def report_diagnostic(output, phase):
    from .diagnostic_report import build
    build(output, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    run(read_json(args.config), args.output.resolve(), args.resume, args.probe_only)
