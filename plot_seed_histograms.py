"""2x2 held-out error histograms for SIO, SWOT, raw8 and CDPA."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_seed_results import ROOT, REGIONS, load_points


def plot_region(args, region):
    experiments = ('sio', 'swot', 'raw8', args.cdpa)
    titles = ('SIO (4)', 'SWOT (4)', 'SIO-SWOT (8)',
              f'CDPA ({args.cdpa.removeprefix("cdpa").replace("_split", " + split")})')
    base = args.results_root / region / 'ablations' / f'seed_{args.seed}'
    data = [load_points(base / name, args.seed, region, 'test') for name in experiments]
    for rows, true, _, _ in data[1:]:
        if not np.array_equal(rows, data[0][0]) or not np.array_equal(true, data[0][1]):
            raise ValueError(f'Test samples differ: {region}')
    errors = [predicted - true for _, true, predicted, _ in data]
    if args.absolute:
        errors = [np.abs(error) for error in errors]
    extent = max(float(abs(error).max()) for error in errors)
    extent = max(args.bin_width, np.ceil(extent / args.bin_width) * args.bin_width)
    edges = np.arange(0 if args.absolute else -extent, extent + args.bin_width * .5, args.bin_width)
    # Keep full-data bins and statistics; zoom all four panels to a common range.
    if args.x_limit is not None:
        display_extent = args.x_limit
    else:
        central_extent = max(float(np.percentile(np.abs(error), args.coverage)) for error in errors)
        tick_step = max(args.bin_width, 100.)
        display_extent = max(tick_step, np.ceil(central_extent / tick_step) * tick_step)
    display_limits = (0 if args.absolute else -display_extent, display_extent)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True, layout='constrained')
    stats, bins, points = [], [], []
    for ax, name, title, error, (rows, true, predicted, metrics) in zip(axes.flat, experiments, titles, errors, data):
        counts, _ = np.histogram(error, bins=edges)
        if int(counts.sum()) != len(error):
            raise ValueError('Histogram bounds excluded samples')
        heights = counts / (len(error) * args.bin_width) if args.density else counts
        ax.bar(edges[:-1], heights, width=args.bin_width, align='edge', color='#4c8fc0',
               edgecolor='white', linewidth=.3, alpha=.9)
        if not args.absolute:
            ax.axvline(0, color='#222222', linestyle='--', lw=1.1, label='Zero error')
        mean = float(error.mean())
        ax.axvline(mean, color='#bf3030', lw=1.2, label='Mean')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Absolute Error (m)' if args.absolute else 'Prediction Error (Predicted − Ship) (m)')
        ax.set_ylabel('Probability Density (m$^{-1}$)' if args.density else 'Frequency')
        ax.set_xlim(*display_limits)
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(axis='y', linestyle=':', alpha=.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=9)
        outside = int(((error < display_limits[0]) | (error > display_limits[1])).sum())
        text = (f"N = {len(error):,}\n" + ('Mean absolute error' if args.absolute else 'Bias') + f' = {mean:.1f} m\n'
                f"RMSE = {metrics['RMSE']:.1f} m\nMAE = {metrics['MAE']:.1f} m\n"
                f"Outside view: {outside} ({outside/len(error):.2%})")
        ax.text(.97, .95, text, transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        stats.append(dict(experiment=name, **metrics, bias=float(np.mean(predicted-true)),
                          outside_view=outside,
                          error_std=float(np.std(predicted-true)), median_error=float(np.median(predicted-true))))
        bins.append(pd.DataFrame(dict(experiment=name, left=edges[:-1], right=edges[1:], count=counts, height=heights)))
        points.append(pd.DataFrame(dict(experiment=name, ship_row_id=rows, ship_depth=true,
                                        predicted_depth=predicted, error=predicted-true)))
    fig.suptitle(f'{region} | Training seed {args.seed} | Test set', fontsize=14)
    output = args.output_dir or args.results_root / 'histogram_visualizations'
    output.mkdir(parents=True, exist_ok=True)
    stem = f'error_histogram_{region}_seed{args.seed}_test_{args.cdpa}' + ('_absolute' if args.absolute else '')
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{suffix}', dpi=args.dpi)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(output / f'{stem}_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(bins, ignore_index=True).to_csv(output / f'{stem}_bins.csv', index=False)
    pd.concat(points, ignore_index=True).to_csv(output / f'{stem}_points.csv', index=False)
    (output / f'{stem}.json').write_text(json.dumps(dict(region=region, seed=args.seed, subset='test',
             experiments=experiments, bin_width_m=args.bin_width, density=args.density,
             absolute=args.absolute, full_error_range=bool(display_extent >= extent), identical_samples=True,
             x_limits_m=display_limits, coverage_percent=args.coverage,
             all_samples_in_statistics_and_exported_bins=True,
             source=str(base.resolve())), indent=2), encoding='utf-8')
    print(f'Saved: {output / (stem + ".png")}')
    print(f'Common x limits: {display_limits} m; full-data metrics retained')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'data' / 'result' / '\u6d88\u878d')
    parser.add_argument('--region', choices=(*REGIONS, 'all'), default='all')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cdpa', choices=('cdpa12', 'cdpa16', 'cdpa16_split'), default='cdpa12')
    parser.add_argument('--bin-width', type=float, default=50, help='Common histogram bin width, metres')
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--coverage', type=float, default=99,
                        help='Auto range covers at least this percentile of absolute errors in each panel')
    parser.add_argument('--x-limit', type=float, help='Explicit maximum absolute displayed error, metres')
    parser.add_argument('--density', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if not np.isfinite(args.bin_width) or args.bin_width <= 0:
        parser.error('--bin-width must be finite and positive')
    if not np.isfinite(args.coverage) or not 0 < args.coverage <= 100:
        parser.error('--coverage must be in (0, 100]')
    if args.x_limit is not None and (not np.isfinite(args.x_limit) or args.x_limit <= 0):
        parser.error('--x-limit must be finite and positive')
    for region in REGIONS if args.region == 'all' else (args.region,):
        plot_region(args, region)


if __name__ == '__main__':
    main()
