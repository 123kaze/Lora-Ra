"""Reconstruct every final phase-2 adapter checkpoint and reproduce validation.

Requires the original local tensors and the trained code version.
"""
import argparse
import json
from pathlib import Path

import torch

from carryrank.diagnostic import rank_scale
from carryrank.experiment import Experiment, base_digest, read_json
from carryrank.lora import GroupedLinear, attach


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    cfg = read_json(output / "diagnostic_config.json")
    files = sorted(output.glob("base_*/diagnostic_adapters/*/adapter.pt"))
    expected = (len(cfg["base_seeds"]) * len(cfg["adapter_seeds"]) * len(cfg["scales"]) * len(cfg["main_methods"])
                + 2 * len(cfg["scales"]) * len(cfg["sweep_ranks"]) + 3 * len(cfg["concentration_methods"]))
    if len(files) != expected:
        raise AssertionError(f"Incomplete run: {len(files)} != {expected}")
    experiments, records = {}, []
    for index, path in enumerate(files, 1):
        base_dir = path.parents[2]
        if base_dir not in experiments:
            experiments[base_dir] = Experiment(read_json(base_dir / "config.json"), base_dir, resume=True)
        experiment = experiments[base_dir]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        measured = read_json(path.parent / "result.json")
        final = next(row for row in measured["evaluations"] if row["step"] == cfg["steps"])
        model, checkpoint = experiment.load_base()
        actual_params = attach(model, saved["groups"], saved["ranks"], scaling=1.0)
        for module in model.modules():
            if isinstance(module, GroupedLinear):
                for factor in module.factors:
                    factor.scaling = rank_scale(saved["mode"], factor.A.shape[0])
        state = model.state_dict()
        expected_keys = {key for key in state if ".factors." in key}
        if expected_keys != set(saved["adapter"]):
            raise AssertionError(f"Saved adapter keys do not match architecture: {path}")
        state.update(saved["adapter"])
        model.load_state_dict(state, strict=True)
        if base_digest(model) != saved["base_sha256"] or actual_params != final["budget"]:
            raise AssertionError(f"Base/budget mismatch: {path}")
        validation = experiment.evaluate(model, "target_val")
        difference = abs(validation["loss"] - final["val_loss"])
        if difference > 1e-5:
            raise AssertionError(f"Checkpoint failed to reproduce final metric: {path}: {difference}")
        records.append({"unit": path.relative_to(output).as_posix(), "params": actual_params,
                        "val_loss": validation["loss"], "absolute_error": difference})
        del model, state
        if index % 10 == 0 or index == len(files):
            print(f"Audited checkpoints: {index}/{len(files)}", flush=True)
    result = {"run": output.name, "checkpoints_reconstructed": len(records),
              "max_validation_loss_error": max(row["absolute_error"] for row in records),
              "base_hashes_and_actual_budgets_match": True, "pass": True, "records": records}
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
