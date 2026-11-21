"""Reproducible pretraining, gradient calibration and matched-budget adapters.

Run using scripts/run.ps1 or scripts/run_wsl.sh. No third-party trainer required.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .allocation import allocate, equal_parameters
from .data import digest, evaluation_batches, prepare, random_batch
from .lora import attach, merge_all
from .model import ModelConfig, VariableWidthLM, structure

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("C-only", "R-only", "Equal", "Gradient", "Shuffled", "Uniform")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def base_digest(model):
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".factors." in name:
            continue
        name = name.replace(".base.", ".")
        hasher.update(name.encode())
        hasher.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def code_manifest():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "figures").rglob("*.py"))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path.read_bytes()) for path in paths}


def new_attempt(output, name):
    parent = output / "logs" / name
    parent.mkdir(parents=True, exist_ok=True)
    attempt = parent / f"attempt_{len(list(parent.glob('attempt_*'))):03d}.jsonl"
    return attempt.open("x", encoding="utf-8")


def log(handle, event):
    handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **event}) + "\n")
    handle.flush()
    print(json.dumps(event), flush=True)


class Experiment:
    def __init__(self, config, output, resume):
        self.config, self.output = config, output
        self.device = torch.device(config["runtime"]["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
        torch.set_num_threads(config["runtime"]["threads"])
        torch.backends.cudnn.benchmark = False
        # SDPA may use nondeterministic kernels; record this scope rather than promise bitwise reruns.
        torch.backends.cuda.matmul.allow_tf32 = False
        self.dtype = torch.bfloat16 if config["runtime"]["precision"] == "bfloat16" else torch.float32
        if self.dtype == torch.bfloat16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but GPU does not support it")
        self.model_config = ModelConfig(**config["model"])
        self.splits, data_manifest = prepare(ROOT, config["data"])
        identity = {"config": config, "code": code_manifest(), "data": data_manifest,
                    "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "device": str(self.device), "platform": platform.platform()}
        if output.exists():
            if not resume:
                raise FileExistsError("Run already exists; choose a new --output, or explicitly --resume")
            if read_json(output / "identity.json") != identity:
                raise ValueError("Resume rejected: config/code/data/environment identity changed")
        else:
            if resume:
                raise FileNotFoundError("Cannot resume a nonexistent run")
            output.mkdir(parents=True)
            write_json(output / "identity.json", identity)
            write_json(output / "config.json", config)
            write_json(output / "data_manifest.json", data_manifest)
            write_json(output / "structure.json", structure(self.model_config.widths))
            write_json(output / "environment.json", {
                **{k: identity[k] for k in ("python", "torch", "numpy", "device", "platform")},
                "executable": sys.executable, "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "bf16": self.dtype == torch.bfloat16,
                "precision": config["runtime"]["precision"], "tf32": False,
                "bitwise_reproducibility_guaranteed": False,
                "original_proposal_sha256": digest((ROOT / "研究方案.md").read_bytes())})
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
            (output / "environment-freeze.txt").write_text(freeze.stdout, encoding="utf-8")

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype) if self.dtype != torch.float32 else contextlib.nullcontext()

    def model(self):
        return VariableWidthLM(self.model_config).to(self.device)

    def load_base(self):
        checkpoint = torch.load(self.output / "base.pt", map_location="cpu", weights_only=True)
        model = self.model()
        model.load_state_dict(checkpoint["model"])
        if base_digest(model) != checkpoint["sha256"]:
            raise ValueError("Base checkpoint hash mismatch")
        return model, checkpoint

    @torch.no_grad()
    def evaluate(self, model, split):
        model.eval()
        evaluation = self.config["evaluation"]
        total_loss, correct, count = 0.0, 0, 0
        for x, y in evaluation_batches(self.splits[split], evaluation["batch_size"],
                                       self.model_config.seq_len, evaluation["max_blocks"], self.device):
            with self.autocast():
                logits = model(x)
            # Accumulate loss in full precision, weighted by predicted bytes.
            total_loss += F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum").item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
        loss = total_loss / count
        return {"loss": loss, "bits_per_byte": loss / math.log(2), "accuracy": correct / count, "bytes": count}

    def pretrain(self):
        if (self.output / "base.pt").exists():
            self.load_base()
            print("Verified existing completed pretraining checkpoint", flush=True)
            return
        cfg = self.config["pretrain"]
        seed_all(cfg["seed"])
        model = self.model()
        generator = torch.Generator().manual_seed(cfg["seed"] + 1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
        initial_source = self.evaluate(model, "source_val")
        initial_target = self.evaluate(model, "target_val")
        best_loss, best_state, best_step = initial_source["loss"], None, 0
        synchronize(self.device)
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with new_attempt(self.output, "pretrain") as handle:
            log(handle, {"stage": "pretrain_initial", "parameters": sum(p.numel() for p in model.parameters()),
                         "source_val": initial_source, "target_val": initial_target})
            for step in range(1, cfg["steps"] + 1):
                model.train()
                x, y = random_batch(self.splits["source_train"], cfg["batch_size"], self.model_config.seq_len, generator, self.device)
                optimizer.zero_grad(set_to_none=True)
                warmup = max(1, int(cfg["steps"] * 0.08))
                fraction = min(1.0, step / warmup)
                decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, cfg["steps"] - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = cfg["lr"] * fraction * decay
                with self.autocast():
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite pretraining loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % self.config["runtime"]["log_every"] == 0 or step == cfg["steps"]:
                    log(handle, {"stage": "pretrain", "step": step, "train_loss": loss.item(),
                                 "elapsed_s": time.perf_counter() - started})
                if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                    value = self.evaluate(model, "source_val")
                    log(handle, {"stage": "source_validation", "step": step, **value})
                    if value["loss"] < best_loss:
                        best_loss, best_step = value["loss"], step
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            synchronize(self.device)
            elapsed = time.perf_counter() - started
            if best_state is None:
                raise RuntimeError("Pretraining failed to improve source validation loss")
            model.load_state_dict(best_state)
            base_hash = base_digest(model)
            source_result = self.evaluate(model, "source_val")
            target_result = self.evaluate(model, "target_val")
            improvement = initial_source["loss"] - source_result["loss"]
            # Smoke can complete but is always marked ineligible as research evidence.
            gate = improvement >= 0.3 and "smoke" not in self.config["name"]
            summary = {"initial_source_val": initial_source, "initial_target_val": initial_target,
                       "source_val": source_result, "target_val": target_result,
                       "source_improvement": improvement, "pretraining_gate": gate,
                       "selected_step": best_step, "total_steps": cfg["steps"], "time_s": elapsed,
                       "parameters": sum(p.numel() for p in model.parameters()), "base_sha256": base_hash,
                       "peak_allocated_mb": self.peak_memory()}
            torch.save({"model": best_state, "sha256": base_hash, "summary": summary}, self.output / "base.pt")
            write_json(self.output / "pretrain.json", summary)
            log(handle, {"stage": "pretrain_complete", **summary})

    def peak_memory(self):
        return torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else 0.0

    def calibrate(self):
        destination = self.output / "calibration.pt"
        if destination.exists():
            return torch.load(destination, map_location="cpu", weights_only=True)
        cfg = self.config["calibration"]
        seed_all(cfg["seed"])
        model, base = self.load_base()
        model.requires_grad_(False)
        targets = list(model.targets())
        for _, linear, _ in targets:
            linear.weight.requires_grad_(True)
        before = base_digest(model)
        gradients = {name: torch.zeros_like(linear.weight, device="cpu", dtype=torch.float64)
                     for name, linear, _ in targets}
        generator = torch.Generator().manual_seed(cfg["seed"])
        started = time.perf_counter()
        # Calibration deliberately uses FP32, no parameter or optimizer update.
        model.eval()
        for _ in range(cfg["batches"]):
            x, y = random_batch(self.splits["target_train"], cfg["batch_size"], self.model_config.seq_len, generator, self.device)
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            loss.backward()
            for name, linear, _ in targets:
                gradients[name] += linear.weight.grad.detach().cpu().double() / cfg["batches"]
        after = base_digest(model)
        if before != after or after != base["sha256"]:
            raise AssertionError("Calibration changed base weights")
        groups, utilities, spectra = {}, {}, []
        permutation_rng = torch.Generator().manual_seed(cfg["shuffle_seed"])
        for scheme in ("true", "shuffled", "uniform"):
            groups[scheme], utilities[scheme] = [], {}
            for name, linear, rec in targets:
                if scheme == "uniform":
                    partition = [("all", list(range(linear.in_features)))]
                elif scheme == "true":
                    partition = [(label, rec[label]) for label in ("C", "R")]
                else:
                    permutation = torch.randperm(linear.in_features, generator=permutation_rng).tolist()
                    partition = [("C", sorted(permutation[:len(rec["C"])])),
                                 ("R", sorted(permutation[len(rec["C"]):]))]
                for label, indices in partition:
                    gid = f"{name}/{label}"
                    group = {"id": gid, "module": name, "layer": rec["layer"], "label": label,
                             "indices": indices, "m": linear.out_features, "n": len(indices),
                             "cost": linear.out_features + len(indices),
                             "age_mean": float(np.mean(rec["R_ages"])) if label == "R" and scheme == "true" else 0.0}
                    matrix = gradients[name][:, indices]
                    singular = torch.linalg.svdvals(matrix)
                    cap = min(self.config["adapter"]["max_rank"], len(singular))
                    values = (0.5 * singular[:cap].square()).tolist()
                    utilities[scheme][gid] = [0.0] + np.cumsum(values).tolist()
                    groups[scheme].append(group)
                    for j, sigma in enumerate(singular[:cap].tolist(), 1):
                        spectra.append({"scheme": scheme, "group": gid, "layer": rec["layer"], "label": label,
                                        "m": group["m"], "n": group["n"], "cost": group["cost"],
                                        "age_mean": group["age_mean"], "rank": j, "sigma": sigma,
                                        "marginal_utility": sigma * sigma / 2,
                                        "utility_per_parameter": sigma * sigma / (2 * group["cost"]),
                                        "gradient_frobenius": matrix.norm().item()})
        result = {"groups": groups, "utilities": utilities, "gradients": gradients,
                  "base_sha256": before, "calibration_seconds": time.perf_counter() - started,
                  "calibration_bytes": cfg["batches"] * cfg["batch_size"] * self.model_config.seq_len,
                  "weights_unchanged": before == after}
        torch.save(result, destination)
        write_csv(self.output / "spectra.csv", spectra)
        write_json(self.output / "calibration.json", {k: v for k, v in result.items() if k not in ("gradients", "utilities")})
        print(f"Calibration complete: {result['calibration_seconds']:.2f}s; base unchanged", flush=True)
        return result

    def allocations(self, calibration):
        cfg = self.config["adapter"]
        budget, cap = cfg["budget"], cfg["max_rank"]
        true, shuffled, uniform = [calibration["groups"][s] for s in ("true", "shuffled", "uniform")]
        result = {}
        for label in ("C", "R"):
            restricted = [g for g in true if g["label"] == label]
            ranks, _, _ = equal_parameters(restricted, budget, cap)
            result[f"{label}-only"] = (true, ranks)
        equal = {}
        if budget % 2:
            raise ValueError("Equal comparison needs an even parameter budget")
        for label in ("C", "R"):
            ranks, _, _ = equal_parameters([g for g in true if g["label"] == label], budget // 2, cap)
            equal.update(ranks)
        result["Equal"] = (true, equal)
        for method, scheme, groups in (("Gradient", "true", true), ("Shuffled", "shuffled", shuffled)):
            ranks, _, _ = allocate(groups, calibration["utilities"][scheme], budget, cap)
            result[method] = (groups, ranks)
        uniform_ranks = {g["id"]: cfg["uniform_rank"] for g in uniform}
        result["Uniform"] = (uniform, uniform_ranks)
        for method, (groups, ranks) in result.items():
            cost = sum(g["cost"] * ranks.get(g["id"], 0) for g in groups)
            if cost != budget:
                raise AssertionError(f"Budget mismatch for {method}: {cost} != {budget}")
        write_json(self.output / "allocations.json", {name: {"ranks": ranks, "groups": groups} for name, (groups, ranks) in result.items()})
        return result

    @torch.no_grad()
    def latency(self, model, x, repeats=20):
        model.eval()
        for _ in range(5):
            model(x)
        synchronize(self.device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(x)
        synchronize(self.device)
        return (time.perf_counter() - started) * 1000 / repeats

    def adapt(self, method, seed, groups, ranks, steps, folder="adapters", test=True):
        unit = self.output / folder / f"{method}_seed{seed}"
        if (unit / "result.json").exists():
            return read_json(unit / "result.json")
        unit.mkdir(parents=True, exist_ok=True)
        cfg = self.config["adapter"]
        seed_all(seed)
        model, base = self.load_base()
        probe = self.splits["target_train"][:self.model_config.seq_len].unsqueeze(0).to(self.device)
        model.eval()
        with torch.no_grad():
            expected = model(probe)
        # Reset adapter RNG after loading base (whose fresh initialization consumes RNG).
        seed_all(seed)
        params = attach(model, groups, ranks, cfg["scaling"])
        with torch.no_grad():
            initial_error = (model(probe) - expected).abs().max().item()
        if initial_error != 0.0:
            raise AssertionError(f"Nonzero initial adapter delta: {initial_error}")
        parameters = [p for p in model.parameters() if p.requires_grad]
        if params != sum(g["cost"] * ranks.get(g["id"], 0) for g in groups):
            raise AssertionError("Actual parameter count differs from rank cost")
        generator = torch.Generator().manual_seed(seed + 100000)
        data_trace = hashlib.sha256()
        optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0) if parameters else None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        with new_attempt(self.output, f"{folder}_{method}_seed{seed}") as handle:
            for step in range(1, steps + 1) if optimizer else ():
                model.train()
                x, y = random_batch(self.splits["target_train"], cfg["batch_size"], self.model_config.seq_len, generator, self.device)
                # Record actual batch order, not just the RNG seed.
                data_trace.update(x.detach().cpu().contiguous().numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite adapter loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                if step % self.config["runtime"]["log_every"] == 0 or step == steps:
                    log(handle, {"stage": folder, "method": method, "seed": seed, "step": step,
                                 "train_loss": loss.item(), "elapsed_s": time.perf_counter() - started})
            synchronize(self.device)
            training_seconds, peak = time.perf_counter() - started, self.peak_memory()
            frozen_hash = base_digest(model)
            if frozen_hash != base["sha256"]:
                raise AssertionError("Frozen base changed during adaptation")
            validation = self.evaluate(model, "target_val")
            testing = self.evaluate(model, "target_test") if test else None
            model.eval()
            with torch.no_grad():
                unmerged_output = model(probe)
            unmerged_latency = self.latency(model, probe)
            torch.save({"adapter": {k: v.detach().cpu() for k, v in model.state_dict().items() if ".factors." in k},
                        "groups": groups, "ranks": ranks, "scaling": cfg["scaling"], "base_sha256": frozen_hash}, unit / "adapter.pt")
            merge_all(model)
            with torch.no_grad():
                merged_error = (model(probe) - unmerged_output).abs().max().item()
            if merged_error > 0.0001:
                raise AssertionError(f"Merge mismatch: {merged_error}")
            merged_latency = self.latency(model, probe)
            result = {"method": method, "seed": seed, "steps": steps, "params": params,
                      "val_loss": validation["loss"], "val_bits_per_byte": validation["bits_per_byte"],
                      "val_accuracy": validation["accuracy"], "val_bytes": validation["bytes"],
                      "test_loss": testing["loss"] if testing else None,
                      "test_bits_per_byte": testing["bits_per_byte"] if testing else None,
                      "test_accuracy": testing["accuracy"] if testing else None,
                      "test_bytes": testing["bytes"] if testing else None,
                      "training_seconds": training_seconds, "peak_allocated_mb": peak,
                      "unmerged_latency_ms": unmerged_latency, "merged_latency_ms": merged_latency,
                      "initial_max_error": initial_error, "merge_max_error": merged_error,
                      "frozen_base_sha256": frozen_hash, "data_trace_sha256": data_trace.hexdigest()}
            write_json(unit / "result.json", result)
            write_json(unit / "ranks.json", ranks)
            log(handle, {"stage": "adapt_complete", **result})
        del model, optimizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def main_comparison(self):
        calibration = self.calibrate()
        allocations = self.allocations(calibration)
        results = []
        for seed in self.config["adapter"]["seeds"]:
            # Rotate execution order to reduce systematic thermal/order bias.
            offset = self.config["adapter"]["seeds"].index(seed) % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                groups, ranks = allocations[method]
                result = self.adapt(method, seed, groups, ranks, self.config["adapter"]["steps"])
                if result["params"] != self.config["adapter"]["budget"]:
                    raise AssertionError("Main comparison is not budget matched")
                results.append(result)
                write_csv(self.output / "main.csv", results)
        for seed in self.config["adapter"]["seeds"]:
            if len({r["data_trace_sha256"] for r in results if r["seed"] == seed}) != 1:
                raise AssertionError("Methods did not see identical training batch sequences")

    def sweep(self):
        calibration = self.calibrate()
        cfg = self.config["sweep"]
        records = []
        base, checkpoint = self.load_base()
        baseline = self.evaluate(base, "target_val")["loss"]
        del base
        records.append({"group": "shared_baseline", "label": "none", "rank": 0, "params": 0,
                        "predicted_utility": 0.0, "actual_val_improvement": 0.0, "val_loss": baseline})
        groups = [g for g in calibration["groups"]["true"]
                  if g["layer"] == cfg["layer"] and g["module"].endswith(cfg["projection"])]
        if len(groups) != 2:
            raise ValueError("Sweep must select one expansion projection with C and R")
        for group in groups:
            for rank in cfg["ranks"]:
                if rank == 0:
                    continue
                if rank >= len(calibration["utilities"]["true"][group["id"]]):
                    raise ValueError("Sweep rank exceeds available calibrated spectrum")
                result = self.adapt(f"{group['label']}_rank{rank}", cfg["seed"], [group],
                                    {group["id"]: rank}, cfg["steps"], folder="sweep", test=False)
                records.append({"group": group["id"], "label": group["label"], "rank": rank, "params": result["params"],
                                "predicted_utility": calibration["utilities"]["true"][group["id"]][rank],
                                "actual_val_improvement": baseline - result["val_loss"], "val_loss": result["val_loss"]})
                write_csv(self.output / "sweep.csv", records)

    def report(self):
        from .report import build_report
        build_report(self.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "pretrain", "main", "sweep", "report"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        parser.error("--output must be a child of the project runs directory")
    config = read_json(args.config)
    experiment = Experiment(config, output, args.resume)
    if args.stage in ("all", "pretrain"):
        experiment.pretrain()
    if args.stage in ("all", "main", "sweep"):
        pretrain = read_json(output / "pretrain.json")
        if not pretrain["pretraining_gate"] and "smoke" not in config["name"]:
            raise RuntimeError("Pretraining quality gate failed; stopping mechanism experiment")
    if args.stage in ("all", "main"):
        experiment.main_comparison()
    if args.stage in ("all", "sweep"):
        experiment.sweep()
    if args.stage in ("all", "report"):
        experiment.report()


if __name__ == "__main__":
    main()
