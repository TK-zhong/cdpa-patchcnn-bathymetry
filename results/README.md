# Historical result provenance

`experiment_manifest.csv` identifies saved runs using repository-relative paths and SHA256 hashes of their original metrics files. `historical_metrics/run_metrics.csv` preserves the numeric metrics without reference overrides. Regenerate locally with `python scripts/export_provenance.py` when the source run files are available.

The underlying large prediction grids/checkpoints are excluded from the source release. CSV exports cannot substitute for those files when reproducing predictions. Runs are not automatically assigned to a final manuscript table. See `docs/REPRODUCIBILITY.md` for known baseline discrepancies and the historical `arch_full` plot override.
