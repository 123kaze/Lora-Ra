"""Read-only independent checkpoint reconstruction and artifact audit.

Usage: python scripts/audit_run.py runs/<id> (with PYTHONPATH=src).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from carryrank.data import digest
from carryrank.experiment import Experiment, ROOT, base_digest
from carryrank.lora import attach


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run.resolve()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    experiment = Experiment(config, output, resume=True)
    with (output / "main.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    calibration = torch.load(output / "calibration.pt", map_location="cpu", weights_only=True)
    allocation = json.loads((output / "allocations.json").read_text(encoding="utf-8"))
    spectral_errors = []
    costs = {}
    for method, entry in allocation.items():
        costs[method] = sum(g["cost"] * entry["ranks"].get(g["id"], 0) for g in entry["groups"])
    for scheme, groups in calibration["groups"].items():
        for group in groups:
            matrix = calibration["gradients"][group["module"]][:, group["indices"]]
            sigma = torch.linalg.svdvals(matrix)
            actual = torch.tensor(calibration["utilities"][scheme][group["id"]], dtype=torch.float64)
            expected = torch.cat((torch.zeros(1, dtype=torch.float64), (0.5 * sigma[:len(actual)-1].square()).cumsum(0)))
            spectral_errors.append((actual - expected).abs().max().item())
    # Reconstruct every saved adapter, including frozen base, from disk.
    reconstruction_errors = []
    for record in records:
        unit = output / "adapters" / f"{record['method']}_seed{record['seed']}"
        saved = torch.load(unit / "adapter.pt", map_location="cpu", weights_only=True)
        model, base = experiment.load_base()
        params = attach(model, saved["groups"], saved["ranks"], saved["scaling"])
        state = model.state_dict()
        if set(saved["adapter"]) != {key for key in state if ".factors." in key}:
            raise AssertionError("Saved adapter tensor keys do not match reconstructed architecture")
        state.update(saved["adapter"])
        model.load_state_dict(state, strict=True)
        if base_digest(model) != saved["base_sha256"] or params != int(record["params"]):
            raise AssertionError("Reconstructed base/parameter mismatch")
        metric = experiment.evaluate(model, "target_val")
        error = abs(metric["loss"] - float(record["val_loss"]))
        reconstruction_errors.append(error)
        if error > 1e-5:
            raise AssertionError(f"Saved adapter failed to reproduce validation: {record['method']} {error}")
        del model
    print(json.dumps({"run": output.name, "adapter_checkpoints_reconstructed": len(records),
                      "parameter_costs": costs, "max_spectral_utility_error": max(spectral_errors),
                      "max_reconstructed_val_loss_error": max(reconstruction_errors),
                      "original_proposal_sha256": digest((ROOT / "研究方案.md").read_bytes()),
                      "pass": all(c == config["adapter"]["budget"] for c in costs.values()) and max(spectral_errors) < 1e-12}, indent=2))


if __name__ == "__main__":
    main()
