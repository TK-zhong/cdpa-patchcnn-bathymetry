"""Plot measured versus predicted depth for identical held-out samples."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
REGIONS = ('-165_-160_30_35', '131_136_1_6', '170_175_31_37')
RAW_REGIONS = dict(zip(REGIONS, ('-170_-160_30_40', '130_140_0_10', '170_180_30_40')))


def load_points(run_dir, seed, region, subset):
    manifest = json.loads((run_dir / 'experiment.json').read_text(encoding='utf-8'))
    if manifest['seed'] != seed or manifest['region'] != region:
        raise ValueError(f'Seed/region mismatch: {run_dir}')
    ship_path = Path(manifest['ship_path'])
    if not ship_path.exists():
        ship_path = ROOT / 'data' / 'ship' / 'single' / f'{RAW_REGIONS[region]}.csv'
    ship = pd.read_csv(ship_path, encoding='utf-8-sig')
    ship.columns = ship.columns.str.strip().str.lower()
    with np.load(run_dir / 'CNN_cdpa_multiscale_split.npz') as split:
        indices = split[f'{subset}_idx']
        rows = split['ship_row_indices'][indices].astype(np.int64)
        lat_indices = split['lat_idx'][indices].astype(np.int64)
        lon_indices = split['lon_idx'][indices].astype(np.int64)
    if not len(rows) or rows.min() < 0 or rows.max() >= len(ship):
        raise ValueError(f'Invalid ship row IDs: {run_dir}')
    grid = pd.read_csv(run_dir / 'CNN_cdpa_multiscale_predicted.csv', index_col=0)
    if lat_indices.max() >= len(grid) or lon_indices.max() >= grid.shape[1]:
        raise ValueError(f'Prediction indices outside grid: {run_dir}')
    true = pd.to_numeric(ship.iloc[rows]['bathymetry'], errors='coerce').to_numpy(dtype=float)
    predicted = grid.to_numpy(dtype=float)[lat_indices, lon_indices]
    if not (np.isfinite(true).all() and np.isfinite(predicted).all()):
        raise ValueError(f'Non-finite comparison samples; refusing to silently drop points: {run_dir}')
    # Match the original pipeline's nearest-cell mapping, including float32 coordinates.
    for field, coords, index in (
        ('latitude', grid.index.to_numpy(dtype=np.float32), lat_indices),
        ('longitude', grid.columns.to_numpy(dtype=np.float32), lon_indices),
    ):
        points = ship.iloc[rows][field].to_numpy(dtype=np.float32)
        order = np.argsort(coords)
        sorted_coords = coords[order]
        right = np.clip(np.searchsorted(sorted_coords, points), 0, len(coords)-1)
        left = np.maximum(right-1, 0)
        nearest = order[np.where(abs(sorted_coords[left]-points) <= abs(sorted_coords[right]-points), left, right)]
        if not np.array_equal(nearest, index):
            raise ValueError(f'Ship/grid mapping changed ({field}): {run_dir}')
    error = predicted - true
    ss_tot = np.sum((true-true.mean())**2)
    metrics = dict(N=len(true), RMSE=float(np.sqrt(np.mean(error**2))),
                   MAE=float(np.mean(abs(error))),
                   R2=float(1-np.sum(error**2)/ss_tot) if ss_tot > 0 else None)
    saved = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    expected = saved['metrics'][subset.capitalize()]
    if expected['N'] != len(true) or not np.isclose(metrics['RMSE'], expected['RMSE'], rtol=1e-5, atol=0.01):
        raise ValueError(f'Plot values disagree with saved metrics: {run_dir}')
    return rows, true, predicted, metrics


def density_values(true, predicted):
    if len(true) < 3:
        raise ValueError('At least three samples are needed for a density scatter plot')
    try:
        values = np.vstack((true, predicted))
        return gaussian_kde(values)(values), 'Gaussian KDE (Scott bandwidth)'
    except np.linalg.LinAlgError:
        # Singular clouds (e.g. perfect predictions) cannot have a full-rank 2D KDE.
        counts, xedges, yedges = np.histogram2d(true, predicted, bins=80)
        area = np.diff(xedges)[:, None] * np.diff(yedges)[None, :]
        density = counts / (len(true) * area)
        xi = np.clip(np.searchsorted(xedges, true, side='right')-1, 0, len(xedges)-2)
        yi = np.clip(np.searchsorted(yedges, predicted, side='right')-1, 0, len(yedges)-2)
        return density[xi, yi], '2D histogram density (singular KDE fallback)'


def plot_region(args, region):
    experiments = ('sio', 'swot', 'raw8', args.cdpa)
    titles = ('SIO (4)', 'SWOT (4)', 'SIO-SWOT (8)',
              f'CDPA ({args.cdpa.removeprefix("cdpa").replace("_split", " + split")})')
    base = args.results_root / region / 'ablations' / f'seed_{args.seed}'
    loaded = [load_points(base / experiment, args.seed, region, args.subset)
              for experiment in experiments]
    reference_rows, reference_true = loaded[0][:2]
    for experiment, (rows, true, _, _) in zip(experiments, loaded):
        if not np.array_equal(rows, reference_rows) or not np.array_equal(true, reference_true):
            raise ValueError(f'Test samples/order differ for {experiment}; cannot compare directly')
    densities = [density_values(true, predicted) for _, true, predicted, _ in loaded]
    all_depths = np.concatenate([np.concatenate((true, predicted)) for _, true, predicted, _ in loaded])
    low, high = np.floor(all_depths.min()/500)*500, np.ceil(all_depths.max()/500)*500
    if low == high:
        low, high = low-500, high+500
    margin = (high-low)*0.025
    limits = (low-margin, high+margin)
    common_norm = Normalize(0, max(d.max() for d, _ in densities)) if args.shared_density else None
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')
    metadata, metric_rows, point_tables = [], [], []
    for ax, experiment, title, data, density in zip(axes.flat, experiments, titles, loaded, densities):
        rows, true, predicted, metrics = data
        colors, method = density
        order = np.argsort(colors)
        points = ax.scatter(true[order], predicted[order], c=colors[order], cmap='plasma',
                            norm=common_norm, s=args.point_size, edgecolors='none', rasterized=True)
        ax.plot([low, high], [low, high], 'k--', lw=1.3, label='y = x')
        ax.set(xlim=limits, ylim=limits, xlabel='Ship Depth (m)', ylabel='Predicted Depth (m)')
        ax.set_title(title, fontweight='bold', fontsize=15)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(linestyle=':', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', framealpha=0.9)
        if not args.no_metrics:
            text = f"N = {metrics['N']:,}\nRMSE = {metrics['RMSE']:.1f} m\nMAE = {metrics['MAE']:.1f} m"
            if metrics['R2'] is not None:
                text += f"\nR² = {metrics['R2']:.4f}"
            ax.text(.035, .965, text, transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        fig.colorbar(points, ax=ax, label='Point density (m$^{-2}$)', format=formatter, shrink=.88)
        metric_rows.append(dict(experiment=experiment, **metrics))
        point_tables.append(pd.DataFrame(dict(experiment=experiment, ship_row_id=rows,
                                             ship_depth=true, predicted_depth=predicted, density=colors)))
        metadata.append(dict(experiment=experiment, run_dir=str(base/experiment), density_method=method))
    # fig.suptitle(f'{region} | training seed {args.seed} | {args.subset.capitalize()} set', fontsize=15)
    destination = args.output_dir or args.results_root / 'visualizations'
    destination.mkdir(parents=True, exist_ok=True)
    stem = f'depth_density_{region}_seed{args.seed}_{args.subset}_{args.cdpa}'
    for suffix in ('png', 'pdf'):
        fig.savefig(destination / f'{stem}.{suffix}', dpi=args.dpi)
    plt.close(fig)
    pd.DataFrame(metric_rows).to_csv(destination / f'{stem}_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(point_tables, ignore_index=True).to_csv(destination / f'{stem}_points.csv', index=False)
    metadata = dict(region=region, training_seed=args.seed, subset=args.subset,
                    shared_density=args.shared_density, identical_samples=True, panels=metadata)
    (destination / f'{stem}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved: {destination / (stem + ".png")}')
    print(pd.DataFrame(metric_rows).to_string(index=False))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'data' / 'result' / '\u6d88\u878d')
    parser.add_argument('--region', choices=(*REGIONS, 'all'), default='all')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--subset', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--cdpa', choices=('cdpa12', 'cdpa16', 'cdpa16_split'), default='cdpa12')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--shared-density', action='store_true', help='Use a common density color scale')
    parser.add_argument('--no-metrics', action='store_true')
    parser.add_argument('--point-size', type=float, default=5)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    for region in REGIONS if args.region == 'all' else (args.region,):
        plot_region(args, region)


if __name__ == '__main__':
    main()
