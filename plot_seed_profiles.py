"""Compare seed-specific bathymetry profiles and absolute errors along one survey."""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_seed_results import ROOT, REGIONS, RAW_REGIONS, load_points


def load_profile_samples(base, experiment, seed, region, subset):
    subsets = ('train', 'val', 'test') if subset == 'all' else (subset,)
    loaded = [load_points(base / experiment, seed, region, part) for part in subsets]
    rows = np.concatenate([item[0] for item in loaded])
    true = np.concatenate([item[1] for item in loaded])
    predicted = np.concatenate([item[2] for item in loaded])
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f'Duplicate CSV row IDs across subsets: {experiment}')
    order = np.argsort(rows)
    return rows[order], true[order], predicted[order]


def make_profile(args, region):
    experiments = ('sio', 'swot', 'raw8', args.cdpa)
    base = args.results_root / region / 'ablations' / f'seed_{args.seed}'
    loaded = [load_profile_samples(base, exp, args.seed, region, args.subset) for exp in experiments]
    rows, true, _ = loaded[0]
    for exp, (other_rows, other_true, _) in zip(experiments, loaded):
        if not np.array_equal(rows, other_rows) or not np.array_equal(true, other_true):
            raise ValueError(f'Profile sample mismatch: {exp}')
    manifest = json.loads((base / 'sio' / 'experiment.json').read_text(encoding='utf-8'))
    ship_path = Path(manifest['ship_path'])
    if not ship_path.exists():
        ship_path = ROOT / 'data' / 'ship' / 'single' / f'{RAW_REGIONS[region]}.csv'
    ship = pd.read_csv(ship_path, encoding='utf-8-sig')
    ship.columns = ship.columns.str.strip().str.lower()
    if 'survey_name' not in ship.columns:
        raise ValueError(f'Missing survey_name: {ship_path}')
    selected = ship.iloc[rows].copy()
    selected['ship_row_id'] = rows
    selected['ship_depth'] = true
    for exp, (_, _, prediction) in zip(experiments, loaded):
        selected[f'{exp}_predicted'] = prediction
        selected[f'{exp}_abs_error'] = abs(prediction - true)
    counts = selected.groupby('survey_name').size().sort_values(ascending=False)
    if args.list_surveys:
        print(f'{region} | seed {args.seed} | {args.subset}\n{counts.to_string()}')
        return
    candidates = counts[counts >= args.min_points]
    if args.survey is not None:
        # Explicit string comparison also supports numeric survey names.
        match = selected['survey_name'].astype(str) == args.survey
        name = args.survey
    else:
        if candidates.empty:
            raise ValueError(f'No survey with at least {args.min_points} samples: {region}')
        name = candidates.index[0]
        match = selected['survey_name'] == name
    profile = selected.loc[match].copy()
    if len(profile) < 2:
        raise ValueError(f'Survey {name} has fewer than 2 available samples. Use --list-surveys.')

    # Build index from all finite regional ship measurements, never from model error.
    west, east, south, north = map(float, region.split('_'))
    numeric = {c: pd.to_numeric(ship[c], errors='coerce') for c in ('longitude', 'latitude', 'bathymetry')}
    mask = (numeric['longitude'].between(west, east) & numeric['latitude'].between(south, north)
            & np.isfinite(numeric['bathymetry']))
    regional = ship.loc[mask].copy()
    regional['track_index'] = regional.groupby('survey_name').cumcount()
    profile['track_index'] = regional.loc[profile['ship_row_id'], 'track_index'].to_numpy()
    profile['plot_index'] = (np.arange(len(profile)) if args.x_axis == 'sample'
                             else profile['track_index'].to_numpy())
    x = profile['plot_index'].to_numpy()

    def plot_values(column):
        values = profile[column].to_numpy(dtype=float)
        if args.x_axis == 'track':
            # Missing observations remain gaps rather than invented/interpolated curves.
            full_x = np.arange(x[0], x[-1] + 1)
            full_y = np.full(len(full_x), np.nan)
            full_y[x - x[0]] = values
            return full_x, full_y
        return x, values

    labels = ('SIO (4)', 'SWOT (4)', 'SIO-SWOT (8)',
              f'CDPA ({args.cdpa.removeprefix("cdpa").replace("_split", " + split")})')
    colors = ('#87c6ff', '#c97ce0', '#92d59c', '#be2026')
    fig, (ax, error_ax) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                      layout='constrained', gridspec_kw={'height_ratios': [1.15, 1]})
    for exp, label, color in zip(experiments, labels, colors):
        width = 1.7 if exp == args.cdpa else 1.05
        line_x, line_y = plot_values(f'{exp}_predicted')
        marker = '.' if args.x_axis == 'track' else None
        ax.plot(line_x, line_y, label=label, color=color, lw=width, marker=marker, ms=2)
        line_x, line_y = plot_values(f'{exp}_abs_error')
        error_ax.plot(line_x, line_y, label=label, color=color, lw=width, marker=marker, ms=2)
    line_x, line_y = plot_values('ship_depth')
    ax.plot(line_x, line_y, color='#37474f', lw=1.8, label='Ship Measurement', zorder=5,
            marker='.' if args.x_axis == 'track' else None, ms=2)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1], *handles[:-1]], [legend_labels[-1], *legend_labels[:-1]], loc='best')
    error_ax.legend(loc='best')
    ax.set_ylabel('Bathymetric elevation (m)')
    if args.invert_y:
        ax.invert_yaxis()
    error_ax.set_ylabel('Absolute Error (m)')
    error_ax.set_ylim(bottom=0)
    error_ax.set_xlabel('Along-Track Selected Point Index' if args.x_axis == 'sample'
                        else 'Along-Track Regional Point Index (gaps retained)')
    for axes in (ax, error_ax):
        axes.grid(linestyle=':', alpha=.4)
        axes.set_xlim(x[0], x[-1])
    subset_label = 'All samples (includes training/validation)' if args.subset == 'all' else f'{args.subset.capitalize()} samples only'
    ax.set_title(f'Bathymetry Profile Comparison [{name}] ({region})\n'
                 f'Training seed {args.seed} | {subset_label} | N = {len(profile):,}', fontweight='bold')
    destination = args.output_dir or args.results_root / 'profile_visualizations'
    destination.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w.-]+', '_', str(name))
    stem = f'profile_{region}_seed{args.seed}_{args.subset}_{safe_name}_{args.cdpa}_{args.x_axis}'
    for suffix in ('png', 'pdf'):
        fig.savefig(destination / f'{stem}.{suffix}', dpi=args.dpi)
    plt.close(fig)
    profile.to_csv(destination / f'{stem}_points.csv', index=False, encoding='utf-8-sig')
    metrics = []
    for exp in experiments:
        err = profile[f'{exp}_predicted'].to_numpy() - profile['ship_depth'].to_numpy()
        metrics.append(dict(experiment=exp, N=len(err), RMSE=float(np.sqrt(np.mean(err**2))),
                            MAE=float(np.mean(abs(err)))))
    pd.DataFrame(metrics).to_csv(destination / f'{stem}_metrics.csv', index=False, encoding='utf-8-sig')
    metadata = dict(region=region, seed=args.seed, subset=args.subset, survey=str(name),
                    survey_selection='explicit' if args.survey else 'largest sample count',
                    x_axis=args.x_axis, invert_y=args.invert_y, experiments=experiments,
                    source=str(base), ship_path=str(ship_path), identical_samples=True,
                    note='Sample-index lines connect consecutive selected observations; no repeated or synthetic points.')
    (destination / f'{stem}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved: {destination / (stem + ".png")}')
    print(pd.DataFrame(metrics).to_string(index=False))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'data' / 'result' / '\u6d88\u878d')
    parser.add_argument('--region', choices=(*REGIONS, 'all'), default='all')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--survey', help='Exact survey_name; default: largest available survey')
    parser.add_argument('--subset', choices=('test', 'val', 'train', 'all'), default='test')
    parser.add_argument('--cdpa', choices=('cdpa12', 'cdpa16', 'cdpa16_split'), default='cdpa12')
    parser.add_argument('--x-axis', choices=('sample', 'track'), default='sample')
    parser.add_argument('--min-points', type=int, default=20)
    parser.add_argument('--list-surveys', action='store_true')
    parser.add_argument('--invert-y', action='store_true', help='Match the reference image vertical direction')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.survey and args.region == 'all':
        parser.error('Select a single --region when specifying --survey')
    for region in REGIONS if args.region == 'all' else (args.region,):
        make_profile(args, region)


if __name__ == '__main__':
    main()
