# cdpa-patchcnn-bathymetry
Research code for CDPA-based fusion of SIO V32.1 and SWOT_04 gravity products with a multi-scale PatchCNN for bathymetric prediction. Includes feature/architecture ablations,and evaluation utilities. External data and complete reproduction assets are not included.

## Installation

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate DepthNN
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Alternatively, install `requirements.txt` in an existing Python 3.12 environment. Select a PyTorch build appropriate for your hardware before GPU training. These are compatibility ranges, not a reproducibility lock. The preparation was validated in the existing DepthNN environment; a fresh installation has not been tested.

## Data-free example and tests

```bash
python scripts/synthetic_example.py --epochs 2
python -m unittest discover -s tests -v
```

The example uses synthetic two-channel inputs, a smaller CNN and CPU execution. It demonstrates training, validation selection and held-out evaluation; its scores are not scientific results. Outputs go to `build/synthetic_example/`.

## Full experiments

Place external files according to [data preparation](data/README.md). Then inspect or execute versioned configurations:

```bash
python scripts/run_config.py configs/cdpa12_seed123.json --dry-run
python scripts/run_config.py configs/cdpa12_seed123.json
python scripts/run_config.py configs/architecture_seed123.json
python scripts/run_config.py configs/traditional.json
```

These commands use separate reproduction output directories. Training seed 123 and split seed 42 have different roles. Negative-longitude regions must use `--region=-165_-160_30_35`.

The configurations invoke current code; they do not guarantee recreation of historical checkpoints. Inputs, implementation revision, dependency versions and GPU execution also matter.

## Repository map

| Location | Purpose |
|---|---|
| `CDPA_split.py` | Current CNN training, evaluation and map prediction |
| `gravity_models.py` | Shared model definitions |
| `run_ablation.py`, `run_architecture_ablation.py` | Feature and architecture experiments |
| `configs/`, `scripts/run_config.py` | Executable experiment configurations |
| `traditional_gravity.py`, `run_traditional.py` | Regional traditional baselines |
| `export_dataset_splits.py` | Export existing observation partitions |
| `plot_*.py` | Metric, density, histogram and profile figures |
| `audit_cdpa_features.py` | Precomputed feature diagnostics |
| `tests/` | Automated checks |
| `results/` | Lightweight historical metrics and provenance |
| `docs/` | Method definitions, limitations and release checklist |

Entry points remain at the repository root to preserve imports. The curated release excludes historical scripts, Chinese working notes, internal publication checklists, Windows wrappers, optional conversion tools and environment collectors. These remain in the local working project. Single-source experiments use `CDPA_split.py --experiment sio` or `--experiment swot`.

## Scientific scope

The main CNN uses three independent 9/17/25-pixel branches with SE attention and residual blocks. CDPA12 averages the two sources' common components and retains both distinct components. Raw12 is raw8 plus four source-wise means, not CDPA12. See [methods](docs/METHODS.md).

Random observation-level splits can share grid cells across training and test sets. Accuracy therefore characterizes regional interpolation and is not evidence of spatially independent generalization. Traditional methods use training-sounding spatial backgrounds; SAS and Parker are explicitly simplified regional variants. Validation selects their hyperparameters. Nonlinear Parker background coupling still requires further verification.

## Results and figures

Published-result candidates are exported without modifying saved metrics:

```bash
python scripts/export_provenance.py
```

This requires local historical runs. Consult `results/experiment_manifest.csv` and `results/historical_metrics/run_metrics.csv`. These files record available runs, not a finalized manuscript table. Plot reference overrides must not be confused with actual reruns. Default architecture plotting now excludes raw8 only from the displayed figure; the run is retained in provenance.

## Preparing a release

```bash
python scripts/build_release.py
```

The explicit `release_manifest.json` controls the release contents, independently of other local files. The script creates `dist/CDPA-journal-source-clean.zip` and includes per-file hashes. Git is not required for packaging. Author metadata, licensing, data access and final manuscript-to-run mapping remain to be completed before publication.

## License and citation

The project owner has not selected a code license or supplied final author/manuscript metadata. No open-source permission, citation DOI or data redistribution permission is implied by this preparation. Add an appropriate `LICENSE` and a completed `CITATION.cff` before public release; do not use placeholder scholarly identities.
