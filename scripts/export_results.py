"""Export measured artifacts for repository review; never modify or rerun a run.

Export: python scripts/export_results.py --run runs/<run-id>
Verify with no training dependencies: python scripts/export_results.py --verify results/<run-id>
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(bundle):
    manifest = read_json(bundle / "artifact-manifest.json")
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise AssertionError(f"Missing artifact: {entry['path']}")
        data = path.read_bytes()
        if sha(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise AssertionError(f"Hash mismatch: {entry['path']}")
    main, summary, sweep, spectra = [rows(bundle / name) for name in ("main.csv", "summary.csv", "sweep.csv", "spectra.csv")]
    config = read_json(bundle / "config.json")
    expected_count = 6 * len(config["adapter"]["seeds"])
    assert len(main) == len({(r['method'], r['seed']) for r in main}) == expected_count
    assert all(int(r["params"]) == config["adapter"]["budget"] for r in main)
    for entry in summary:
        selected = [r for r in main if r["method"] == entry["method"]]
        for metric in ("val_loss", "test_loss"):
            values = [float(r[metric]) for r in selected]
            assert abs(statistics.mean(values) - float(entry[metric + "_mean"])) < 1e-12
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            assert abs(std - float(entry[metric + "_std"])) < 1e-12
    for row in main:
        name = f"{row['method']}_seed{row['seed']}"
        saved = read_json(bundle / "adapters" / name / "result.json")
        completed = [json.loads(line) for path in (bundle / "logs" / f"adapters_{name}").glob("*.jsonl")
                     for line in path.read_text(encoding="utf-8").splitlines()
                     if json.loads(line).get("stage") == "adapt_complete"]
        assert completed and float(row["val_loss"]) == saved["val_loss"]
        assert any(event["val_loss"] == saved["val_loss"] for event in completed)
    for point in sweep:
        k = int(point["rank"])
        if k:
            prediction = sum(float(r["marginal_utility"]) for r in spectra
                             if r["scheme"] == "true" and r["group"] == point["group"] and int(r["rank"]) <= k)
            assert abs(prediction - float(point["predicted_utility"])) < 1e-12
    result = {"run_id": manifest["run_id"], "files_verified": len(manifest["files"]),
              "main_rows": len(main), "summary_rows": len(summary), "spectra_rows": len(spectra),
              "nonzero_rank_rows": sum(int(r["rank"]) > 0 for r in sweep),
              "hashes_match": True, "summary_recomputed": True, "csv_matches_original_logs": True,
              "rank_predictions_recomputed": True, "requires_original_run_directory": False}
    print(json.dumps(result, indent=2))
    return result


def export(run):
    run = run.resolve()
    if run.parent != (ROOT / "runs").resolve():
        raise ValueError("Expected runs/<run-id>")
    bundle, figure_dir = ROOT / "results" / run.name, ROOT / "figures" / run.name
    if (bundle / "artifact-manifest.json").exists():
        return verify(bundle)
    if not read_json(run / "verification.json")["engineering_pass"]:
        raise ValueError("Engineering verification failed")
    entries = []

    def save(destination, content, origin, source_sha=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError(f"Existing export differs: {destination}")
        if not destination.exists():
            destination.write_bytes(content)
        entries.append({"path": destination.relative_to(ROOT).as_posix(), "bytes": len(content),
                        "sha256": sha(content), "origin": origin, "source_sha256": source_sha or sha(content)})

    required = ("main.csv", "spectra.csv", "sweep.csv", "summary.csv", "report.md", "identity.json", "config.json",
                "data_manifest.json", "environment.json", "environment-freeze.txt", "pretrain.json", "calibration.json",
                "allocations.json", "structure.json", "verification.json", "verdict.json")
    for name in required + tuple(n for n in ("checkpoint_audit.json", "final_validation.json") if (run / n).exists()):
        source = run / name
        content = source.read_bytes()
        if name == "report.md":
            content = content.decode("utf-8").replace("(figures/", f"(../../figures/{run.name}/").encode("utf-8")
        save(bundle / name, content, source.relative_to(ROOT).as_posix(), sha(source.read_bytes()))
    for stem in ("comparison", "sensitivity", "prediction"):
        for suffix in ("png", "svg"):
            source = run / "figures" / f"{stem}.{suffix}"
            save(figure_dir / source.name, source.read_bytes(), source.relative_to(ROOT).as_posix())
    for folder, pattern in (("logs", "*.jsonl"), ("adapters", "*.json"), ("sweep", "*.json")):
        for source in sorted((run / folder).rglob(pattern)):
            save(bundle / source.relative_to(run), source.read_bytes(), source.relative_to(ROOT).as_posix())
    for relative, expected in read_json(run / "identity.json")["code"].items():
        content = (ROOT / relative).read_bytes()
        if sha(content) != expected:
            raise AssertionError(f"Cannot snapshot changed original source: {relative}")
        save(bundle / "source_snapshot" / relative, content, relative)
    private = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
               for p in sorted(run.rglob("*.pt"))]
    checkpoint_note = "[checkpoint audit](checkpoint_audit.json)" if (run / "checkpoint_audit.json").exists() else "checkpoint reconstruction not yet audited"
    index = f"""# Measured results: {run.name}

Original measurements exported without retraining. The original run remains unchanged.

- [Full report](report.md)
- [Raw main CSV](main.csv), [summary](summary.csv), [spectra](spectra.csv), [rank sweep](sweep.csv)
- [Comparison](../../figures/{run.name}/comparison.png), [sensitivity](../../figures/{run.name}/sensitivity.png), [prediction](../../figures/{run.name}/prediction.png)
- [Engineering audit](verification.json), {checkpoint_note}, [decision](verdict.json)
- Original step logs: `logs/`; individual result records: `adapters/`, `sweep/`.
- Original source snapshot: `source_snapshot/`, checked against `identity.json`.
- Sizes, SHA-256 and original paths: `artifact-manifest.json`.

Verify with Python's standard library, independently of the original run directory:

```powershell
python scripts/export_results.py --verify results/{run.name}
```

This checks all artifact hashes, recomputes summary means/sample SD, checks raw CSV against original completion logs, and recomputes sweep utility from spectra. Tensor files are excluded from this review bundle and inventoried by hash; reconstructing checkpoint predictions also requires the original tensors in `runs/{run.name}/`.
"""
    save(bundle / "README.md", index.encode("utf-8"), "generated index")
    manifest = {"run_id": run.name, "source_run": run.relative_to(ROOT).as_posix(), "files": entries,
                "excluded_tensor_files": private, "export_retrained_model": False,
                "report_transform": "Only figure links changed to ../../figures/<run-id>/"}
    (bundle / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return verify(bundle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", type=Path)
    choice.add_argument("--verify", type=Path)
    args = parser.parse_args()
    export(args.run) if args.run else verify(args.verify.resolve())
