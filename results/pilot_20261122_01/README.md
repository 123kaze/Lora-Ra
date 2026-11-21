# Measured results: pilot_20261122_01

Original measurements exported without retraining. The original run remains unchanged.

- [Full report](report.md)
- [Raw main CSV](main.csv), [summary](summary.csv), [spectra](spectra.csv), [rank sweep](sweep.csv)
- [Comparison](../../figures/pilot_20261122_01/comparison.png), [sensitivity](../../figures/pilot_20261122_01/sensitivity.png), [prediction](../../figures/pilot_20261122_01/prediction.png)
- [Engineering audit](verification.json), [checkpoint audit](checkpoint_audit.json), [decision](verdict.json)
- Original step logs: `logs/`; individual result records: `adapters/`, `sweep/`.
- Original source snapshot: `source_snapshot/`, checked against `identity.json`.
- Sizes, SHA-256 and original paths: `artifact-manifest.json`.

Verify with Python's standard library, independently of the original run directory:

```powershell
python scripts/export_results.py --verify results/pilot_20261122_01
```

This checks all artifact hashes, recomputes summary means/sample SD, checks raw CSV against original completion logs, and recomputes sweep utility from spectra. Tensor files are excluded from this review bundle and inventoried by hash; reconstructing checkpoint predictions also requires the original tensors in `runs/pilot_20261122_01/`.
