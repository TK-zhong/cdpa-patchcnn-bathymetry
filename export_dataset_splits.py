"""Export existing ship-observation partitions without resampling or deduplication."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from plot_seed_results import ROOT, REGIONS, RAW_REGIONS


def export_region(args, region):
    source = args.results_root / region / 'ablations' / f'seed_{args.seed}' / args.experiment
    manifest = json.loads((source / 'experiment.json').read_text(encoding='utf-8'))
    if manifest['region'] != region or manifest['seed'] != args.seed:
        raise ValueError(f'Manifest mismatch: {source}')
    split_path = source / 'CNN_cdpa_multiscale_split.npz'
    with np.load(split_path) as saved:
        arrays = {key: saved[key].copy() for key in
                  ('keep_idx', 'ship_row_indices', 'lat_idx', 'lon_idx', 'train_idx', 'val_idx', 'test_idx')}
        split_seed = int(saved['split_seed']) if 'split_seed' in saved else manifest['split_seed']
    n = len(arrays['ship_row_indices'])
    if any(len(arrays[key]) != n for key in ('keep_idx', 'lat_idx', 'lon_idx')):
        raise ValueError('Sample metadata lengths differ')
    combined = np.concatenate([arrays[f'{part}_idx'] for part in ('train', 'val', 'test')])
    if not np.array_equal(np.sort(combined), np.arange(n)):
        raise ValueError('Saved partitions do not form a complete, disjoint sample partition')
    if len(np.unique(arrays['ship_row_indices'])) != n:
        raise ValueError('Original CSV row IDs repeat in sample mapping')

    ship_path = Path(manifest['ship_path'])
    if not ship_path.exists():
        ship_path = ROOT / 'data' / 'ship' / 'single' / f'{RAW_REGIONS[region]}.csv'
    ship = pd.read_csv(ship_path, encoding='utf-8-sig')
    ship.columns = ship.columns.str.strip().str.lower()
    rows = arrays['ship_row_indices'].astype(np.int64)
    if rows.min() < 0 or rows.max() >= len(ship):
        raise ValueError('Original row IDs are outside ship file')
    coords = pd.read_csv(source / 'CNN_cdpa_multiscale_predicted.csv', index_col=0)
    lat, lon = coords.index.to_numpy(float), coords.columns.to_numpy(float)
    grid_rows, grid_cols = arrays['lat_idx'].astype(int), arrays['lon_idx'].astype(int)
    if min(grid_rows.min(), grid_cols.min()) < 0 or grid_rows.max() >= len(lat) or grid_cols.max() >= len(lon):
        raise ValueError('Grid indices are outside coordinate file')
    sample = ship.iloc[rows].reset_index(drop=True).copy()
    metadata_columns = dict(sample_index=np.arange(n), ship_row_id=rows,
                            region=np.full(n, region), grid_row=grid_rows, grid_col=grid_cols,
                            grid_latitude=lat[grid_rows], grid_longitude=lon[grid_cols])
    if set(metadata_columns) & set(sample.columns):
        raise ValueError('Metadata column name conflicts with source ship columns')
    for key, values in reversed(list(metadata_columns.items())):
        sample.insert(0, key, values)
    labels = np.empty(n, dtype=object)
    within = np.zeros(n, dtype=np.int64)
    for part in ('train', 'val', 'test'):
        indices = arrays[f'{part}_idx'].astype(int)
        labels[indices] = part
        within[indices] = np.arange(len(indices))
        expected = manifest['samples'][part.capitalize()]
        if expected != len(indices):
            raise ValueError(f'Sample count differs from manifest: {part}')
    sample.insert(2, 'split', labels)
    sample.insert(3, 'split_order', within)
    if not np.isfinite(pd.to_numeric(sample['bathymetry'], errors='coerce')).all():
        raise ValueError('Export contains invalid bathymetry')
    west, east, south, north = map(float, region.split('_'))
    if not (sample.longitude.between(west, east) & sample.latitude.between(south, north)).all():
        raise ValueError('Ship source changed: points outside region')
    for field, grid, mapped in (('latitude', lat, grid_rows), ('longitude', lon, grid_cols)):
        grid = grid.astype(np.float32)
        values = sample[field].to_numpy(dtype=np.float32)
        order = np.argsort(grid)
        right = np.clip(np.searchsorted(grid[order], values), 0, len(grid)-1)
        left = np.maximum(right-1, 0)
        found = order[np.where(abs(grid[order[left]]-values) <= abs(grid[order[right]]-values), left, right)]
        if not np.array_equal(found, mapped):
            raise ValueError(f'Ship/grid mapping mismatch: {field}')

    verified_peers = []
    for peer in (args.results_root / region / 'ablations').glob('seed_*/*/CNN_cdpa_multiscale_split.npz'):
        with np.load(peer) as other:
            if any(not np.array_equal(arrays[key], other[key]) for key in arrays):
                raise ValueError(f'Existing experiment partition differs: {peer}')
        verified_peers.append(str(peer))

    destination = args.output_dir / f'split_seed_{split_seed}' / region
    files = [destination / name for name in ('train.csv', 'val.csv', 'test.csv', 'all_samples.csv',
                                            'split_indices.npz', 'dataset_summary.json')]
    if not args.overwrite and any(path.exists() for path in files):
        raise FileExistsError(f'Export exists: {destination}; use --overwrite to replace it')
    destination.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for part in ('train', 'val', 'test'):
        partitions[part] = sample.iloc[arrays[f'{part}_idx']].copy()
        partitions[part].to_csv(destination / f'{part}.csv', index=False, encoding='utf-8-sig')
    sample.to_csv(destination / 'all_samples.csv', index=False, encoding='utf-8-sig')
    np.savez(destination / 'split_indices.npz', **arrays, split_seed=split_seed)
    grid_groups = sample.groupby(['grid_row', 'grid_col']).size()
    train_cells = set(map(tuple, partitions['train'][['grid_row', 'grid_col']].to_numpy()))
    test_shared = sum(tuple(cell) in train_cells for cell in partitions['test'][['grid_row', 'grid_col']].to_numpy())
    summary = dict(region=region, split_seed=split_seed, source_training_seed=args.seed,
                   source_experiment=args.experiment, source_split=str(split_path.resolve()),
                   split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                   ship_source=str(ship_path.resolve()), ship_sha256=hashlib.sha256(ship_path.read_bytes()).hexdigest(),
                   counts={key: len(value) for key, value in partitions.items()}, total_samples=n,
                   unique_grid_cells=len(grid_groups), repeated_grid_cells=int((grid_groups>1).sum()),
                   test_records_sharing_train_grid=int(test_shared), verified_existing_splits=len(verified_peers),
                   partition_indices_disjoint=True, deduplicated=False, resplit=False,
                   note='ship_row_id is the zero-based CSV data-row index (header excluded). '
                        'sample_index is the original y_raw index; split_order preserves saved split order. '
                        'These CSVs are observation manifests, not flattened CNN patches.')
    (destination / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{region}: {summary["counts"]}; verified {len(verified_peers)} experiment splits; {destination}')
    return dict(region=region, split_seed=split_seed, **summary['counts'], total_samples=n,
                unique_grid_cells=len(grid_groups), test_records_sharing_train_grid=int(test_shared))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'data' / 'result' / '\u6d88\u878d')
    parser.add_argument('--region', choices=(*REGIONS, 'all'), default='all')
    parser.add_argument('--seed', type=int, default=123, help='Existing training run whose split to export')
    parser.add_argument('--experiment', default='cdpa12')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'dataset_splits')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    summaries = [export_region(args, region) for region in REGIONS if args.region == 'all' or region == args.region]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / 'dataset_counts.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
