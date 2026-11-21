"""Verify published phase-2 CSV/JSON/log evidence using the standard library.

python scripts/verify_phase2_evidence.py results/diagnostic_20261122_02
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def optimal_proxy_value(groups, utilities, budget, cap, q_floor=False):
    """Independent cost/mask DP; verify objective value, not a tie-broken rank vector."""
    q_modules = sorted({g['module'] for g in groups if g['module'].endswith('q_proj')})
    bits = {name: 1 << index for index, name in enumerate(q_modules)}
    states = {(0, 0): 0.0}
    for group in groups:
        updated = {}
        for (spent, mask), value in states.items():
            for rank in range(min(cap, group['m'], group['n']) + 1):
                cost = spent + rank * group['cost']
                if cost > budget:
                    break
                next_mask = mask | (bits.get(group['module'], 0) if rank else 0)
                key = cost, next_mask
                score = value + utilities[group['id']][rank]
                updated[key] = max(updated.get(key, float('-inf')), score)
        states = updated
    target_mask = (1 << len(q_modules)) - 1
    candidates = [value for (spent, mask), value in states.items()
                  if spent == budget and (not q_floor or mask == target_mask)]
    require(bool(candidates), 'Proxy budget is not feasible')
    return max(candidates)


def verify(bundle):
    bundle = bundle.resolve()
    manifest = read(bundle / "artifact-manifest.json")
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), f"Missing artifact: {item['path']}")
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == item["sha256"], f"Hash mismatch: {item['path']}")
        require(len(content) == item["bytes"], f"Size mismatch: {item['path']}")
    cfg = read(bundle / "diagnostic_config.json")
    rows = table(bundle / "diagnostic.csv")
    n_main = len(cfg["base_seeds"]) * len(cfg["adapter_seeds"]) * len(cfg["scales"]) * len(cfg["main_methods"])
    n_capacity = 2 * len(cfg["scales"]) * len(cfg["sweep_ranks"])
    n_concentration = len(cfg["concentration_methods"]) * 3
    expected = n_main + n_capacity + n_concentration
    require(len(rows) == expected * len(cfg["eval_steps"]), "Wrong number of fixed-step evaluation rows")
    keys = [(r["base_seed"], r["adapter_seed"], r["regime"], r["method"], r["mode"], r["lr"], r["step"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicated evaluation rows")
    records = list(bundle.glob("base_*/diagnostic_adapters/*/result.json"))
    require(len(records) == expected, "Wrong number of completed trajectories")
    checks = read(bundle / "diagnostic_verification.json")
    require(checks["engineering_pass"] and all(checks["checks"].values()), "Run engineering checks failed")
    allocation_checks = {}
    for path in records:
        result = read(path)
        require(result["zero_initial_error"] == 0 and result["base_frozen"], "Zero-init/frozen-base check failed")
        require(result["merge_max_error"] < 1e-4, "Merge-equivalence check failed")
        cost = sum(g["cost"] * result["ranks"].get(g["id"], 0) for g in result["groups"])
        require(all(cost == e["budget"] for e in result["evaluations"]), "Actual rank cost disagrees with budget")
        method = result['evaluations'][0]['method']
        if method.startswith(('Gradient', 'DiagInput')):
            scheme = 'shuffled' if 'Shuffled' in method else 'true'
            score_field = 'diag_utility' if method.startswith('DiagInput') else 'raw_utility'
            spectra = table(path.parents[2] / 'normalized_spectra.csv')
            utilities = {}
            for point in spectra:
                if point['scheme'] == scheme:
                    utilities.setdefault(point['group'], {0: 0.0})[int(point['rank'])] = float(point[score_field])
            achieved = sum(utilities[g['id']][result['ranks'].get(g['id'], 0)] for g in result['groups'])
            cache_key = path.parents[2].name, method, cost
            if cache_key not in allocation_checks:
                cap = read(path.parents[2] / 'config.json')['adapter']['max_rank']
                allocation_checks[cache_key] = optimal_proxy_value(result['groups'], utilities, cost, cap,
                                                                 q_floor=method == 'GradientQFloor')
            require(abs(achieved - allocation_checks[cache_key]) < 1e-10, 'Saved allocation is not proxy-optimal')
        if result["evaluations"][0]["method"] == "GradientQFloor":
            modules = {g["module"] for g in result["groups"] if g["module"].endswith("q_proj")}
            require(len(modules) == 2, "Q-floor control must include both expansion projections")
            for module in modules:
                require(sum(result["ranks"].get(g["id"], 0) for g in result["groups"] if g["module"] == module) >= 1,
                        "Q-floor constraint not satisfied")
        logs = path.parents[2] / "logs" / path.parent.name
        events = [json.loads(line) for logfile in logs.glob("*.jsonl")
                  for line in logfile.read_text(encoding="utf-8").splitlines()]
        require(any(e.get("stage") == "diagnostic_complete" for e in events), "Missing original completion log")
        for evaluation in result["evaluations"]:
            matching = [r for r in rows if all(str(evaluation[k]) == r[k]
                        for k in ("base_seed", "adapter_seed", "regime", "method", "mode", "step"))
                        and float(r["lr"]) == evaluation["lr"]]
            require(len(matching) == 1 and float(matching[0]["val_loss"]) == evaluation["val_loss"],
                    "Raw CSV disagrees with individual saved result")
            require(any(e.get("stage") == "diagnostic_validation" and e["step"] == evaluation["step"]
                        and e["val_loss"] == evaluation["val_loss"] for e in events), "Saved result disagrees with original log")
    base_summary = table(bundle / "by_base_summary.csv")
    lookup = {}
    for entry in base_summary:
        selected = [r for r in rows if r["regime"] == "small" and all(r[k] == entry[k] for k in ("base_seed", "method", "mode", "step"))]
        require(len(selected) == len(cfg["adapter_seeds"]), "Wrong number of within-base adapter repeats")
        value = statistics.mean(float(r["val_loss"]) for r in selected)
        require(abs(value - float(entry["val_loss_mean"])) < 1e-12, "Within-base mean mismatch")
        lookup[(entry["base_seed"], entry["method"], entry["mode"], entry["step"])] = value
    for pair in table(bundle / "paired_effects.csv"):
        prefix = (pair["base_seed"], pair["mode"], pair["step"])
        difference = lookup[(prefix[0], "Gradient", prefix[1], prefix[2])] - lookup[(prefix[0], pair["method"], prefix[1], prefix[2])]
        require(abs(difference - float(pair["val_advantage_over_gradient"])) < 1e-12, "Paired-effect mismatch")
    traces = {}
    for row in rows:
        traces.setdefault((row["adapter_seed"], row["step"]), set()).add(row["data_trace_sha256"])
    require(all(len(values) == 1 for values in traces.values()), "Batch order differs across paired trajectories")
    base_hashes = {row["base_sha256"] for row in rows}
    require(len(base_hashes) == len(cfg["base_seeds"]), "Independent base checkpoint hashes are not distinct")
    result = {"run": bundle.name, "artifacts_verified": len(manifest["files"]), "trajectories": len(records),
              "evaluation_rows": len(rows), "independent_bases": len(base_hashes),
              "raw_csv_matches_logs": True, "budget_and_q_floor_constraints": True,
              "within_base_means_and_paired_effects_recomputed": True,
              "proxy_optimality_checks": len(allocation_checks),
              "all_paired_batch_sequences_match": True, "requires_torch": False, "pass": True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
