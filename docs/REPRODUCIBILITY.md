# Reproducibility and scope

## Experimental protocol

The current CNN pipeline uses a random observation-level split of approximately 70/15/15 percent. The split seed is 42. The training seed controls initialization and training randomness; 123 is the main single-run setting. Training and test observations can occupy the same gravity cell. These results measure regional interpolation, not spatially independent generalization.

Normalization uses training observations only. The current training settings are AdamW (learning rate 0.001, weight decay 0.0001), batch size 64, Huber loss with delta 1, gradient clipping at 1, up to 200 epochs, ReduceLROnPlateau (patience 10, factor 0.5), and early stopping after 30 epochs without a new best validation Huber loss. Test metrics are evaluated using the selected checkpoint.

The JSON files in `configs/` specify actual command-line arguments. Other settings remain defined in the source; these configurations are not a complete standalone specification of the implementation. Record a code revision and file hashes with any new run. GPU availability depends on the installed PyTorch build. CUDA execution is not configured for strict deterministic replay in historical runs.

## Known historical discrepancies

- `raw8` and `arch_raw8` have identical saved split indices and normalization statistics in the inspected 170-degree-region runs, but different training trajectories and selected checkpoints. Their historical test RMSE values are 258.447 and 312.577 m, respectively. Equal seeds do not establish identical execution.
- The revised architecture plot previously replaced `arch_full` values with an earlier CDPA12 reference (RMSE 248.497 m, MAE 80.550 m, R2 0.9718). The original architecture run has RMSE 255.051 m and MAE 69.016 m. These are separate runs, not interchangeable measurements.
- Explicit plot overrides are recorded by `plot_architecture_errors.py`; they do not modify source metrics. Do not describe a mixed-run plot as a single controlled experiment.
- Traditional parameters are fitted using training soundings and selected using validation RMSE. A caption stating that all parameters were selected from training soundings alone would be inaccurate for this implementation.

`results/historical_metrics/run_metrics.csv` contains unchanged metrics harvested from available saved runs. `results/experiment_manifest.csv` records relative source paths and hashes. These are provenance records, not an endorsement of a final paper table. The externally supplied older traditional-method table has not been linked to matching prediction files and is not included as a verified result.

## Reproduction commands

```bash
python scripts/synthetic_example.py
python -m unittest discover -s tests -v
python scripts/run_config.py configs/cdpa12_seed123.json --dry-run
python scripts/run_config.py configs/architecture_seed123.json
python scripts/run_config.py configs/traditional.json
```

The first two commands require no external observations. Full experiments require the local data tree described in `data/README.md`. Reproduction outputs use a separate directory to protect historical results. The existing plot scripts may require `--results-root` to locate those outputs; inspect `--help` before running them.

## Environment

The recorded experiment machine used Windows 11, Intel i7-14700HX, approximately 16 GB RAM, and an NVIDIA RTX 4070 Laptop GPU with approximately 8 GB VRAM. The recorded software versions were Python 3.12.13, PyTorch 2.14.1+cu126, CUDA runtime 12.6, cuDNN 9.10.2, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1 and Matplotlib 3.10.9. These describe the recorded machine, not a new compatibility certification. `requirements.txt` defines broad compatibility ranges, not a locked environment. The release has been checked in the existing DepthNN environment, not in a newly provisioned clean environment.

## Release status

Full CDPA decomposition generation, independent data download instructions, a clean-environment end-to-end rerun, final manuscript-to-run mapping, author citation metadata, and a project-owner-selected license remain to be finalized. Do not claim full paper reproducibility until these are supplied.
