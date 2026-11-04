"""Calibrate and evaluate traditional gravity baselines on immutable dataset_splits."""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from traditional_gravity import IDW, Spectrum, ggm, sas, parker, metrics

ROOT = Path(__file__).resolve().parent
REGIONS = {'-165_-160_30_35': '-170_-160_30_40', '131_136_1_6': '130_140_0_10',
           '170_175_31_37': '170_180_30_40'}


def read_grid(source, region):
    path = ROOT/'data'/source/'faa'/f'z_{REGIONS[region]}.csv'
    if not path.exists() and source == 'sio':
        path = ROOT/'data'/source/'grav'/f'z_{REGIONS[region]}.csv'
    frame = pd.read_csv(path, index_col=0)
    west, east, south, north = map(float, region.split('_'))
    lat, lon = frame.index.to_numpy(float), frame.columns.to_numpy(float)
    frame = frame.loc[(lat >= south) & (lat <= north), (lon >= west) & (lon <= east)]
    return frame.to_numpy(float), frame.index.to_numpy(float), frame.columns.to_numpy(float), path


def run(args, region):
    split_dir = args.splits_root/region
    data = {part: pd.read_csv(split_dir/f'{part}.csv') for part in ('train', 'val', 'test')}
    ids = [set(f.ship_row_id) for f in data.values()]
    if any(a & b for a, b in itertools.combinations(ids, 2)):
        raise ValueError('Split observation IDs overlap')
    train = data['train']
    grouped = train.groupby(['grid_row', 'grid_col'], sort=True).bathymetry.median()
    cells, z = np.asarray(grouped.index.tolist(), dtype=int), grouped.to_numpy(float)
    summary = json.loads((split_dir/'dataset_summary.json').read_text(encoding='utf-8'))
    region_rows = []
    for source in args.sources:
        gravity, lat, lon, gravity_path = read_grid(source, region)
        if not np.isfinite(gravity).all():
            raise ValueError(f'FAA contains missing data: {gravity_path}')
        for part, frame in data.items():
            r, c = frame.grid_row.to_numpy(int), frame.grid_col.to_numpy(int)
            if not ((r >= 0).all() and (c >= 0).all() and (r < len(lat)).all() and (c < len(lon)).all()):
                raise ValueError(f'Grid indices outside FAA: {part}')
            if not (np.allclose(lat[r], frame.grid_latitude, atol=1e-5, rtol=0) and
                    np.allclose(lon[c], frame.grid_longitude, atol=2e-5, rtol=0)):
                raise ValueError('FAA and dataset coordinate mapping mismatch')
        spacing = np.array([abs(np.diff(lat).mean())*111195.,
                            abs(np.diff(lon).mean())*111195.*np.cos(np.deg2rad(lat.mean()))])
        interpolator = IDW(cells, gravity.shape, spacing)
        reference = interpolator(z)
        spectrum = Spectrum(gravity.shape, spacing)
        for method in args.methods:
            directory = args.output_dir/region/source/method
            if (directory/'metrics.json').exists() and not args.overwrite:
                print(f'Skip existing {region}/{source}/{method}', flush=True)
                old = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
                region_rows.extend(dict(region=region, source=source, method=method, set=part, **values)
                                   for part, values in old['metrics'].items())
                continue
            start = time.perf_counter()
            best, candidates = None, []
            if method == 'GGM':
                configurations = [dict(density_kg_m3=d) for d in (1000, 1400, 1670, 2000, 2400)]
            elif method == 'SAS':
                configurations = [dict(short_km=s, long_km=l) for s, l in itertools.product((10, 15, 25), (80, 160, 240))]
            else:
                configurations = [dict(density_kg_m3=d, short_km=s, long_km=l)
                                  for d, s, l in itertools.product((1400, 1670, 2000), (15, 25, 40), (80, 160))]
            for config in configurations:
                record = dict(config)
                try:
                    if method == 'GGM':
                        prediction = ggm(gravity, z, cells, interpolator, config['density_kg_m3'])
                        fit = dict(datum_m=float(z.min()))
                    elif method == 'SAS':
                        prediction, fit = sas(gravity, reference, spectrum, train.grid_row.to_numpy(int),
                                              train.grid_col.to_numpy(int), train.bathymetry.to_numpy(float),
                                              config['short_km']*1000, config['long_km']*1000)
                    else:
                        prediction, fit = parker(gravity, reference, spectrum, float(train.bathymetry.mean()),
                                                  config['density_kg_m3'], config['short_km']*1000, config['long_km']*1000)
                    val = data['val']
                    score = metrics(prediction[val.grid_row, val.grid_col], val.bathymetry)['RMSE']
                    record.update(val_RMSE=score, status='ok', **fit)
                    if best is None or score < best[0]:
                        best = score, prediction.copy(), dict(record)
                except ValueError as error:
                    record.update(status='rejected', reason=str(error))
                candidates.append(record)
            if best is None:
                raise RuntimeError(f'No valid candidates for {region}/{source}/{method}: {candidates}')
            _, prediction, selected = best
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(candidates).to_csv(directory/'validation_search.csv', index=False, encoding='utf-8-sig')
            np.save(directory/'predicted.npy', prediction)
            pd.DataFrame(prediction, index=lat, columns=lon).to_csv(directory/'predicted.csv')
            evaluation = {}
            for part, frame in data.items():
                pred = prediction[frame.grid_row.to_numpy(int), frame.grid_col.to_numpy(int)]
                evaluation[part] = metrics(pred, frame.bathymetry)
                points = frame[['sample_index', 'ship_row_id', 'grid_row', 'grid_col', 'bathymetry']].copy()
                points['predicted_depth'] = pred
                points['error_m'] = pred-frame.bathymetry.to_numpy(float)
                points.to_csv(directory/f'{part}_predictions.csv', index=False, encoding='utf-8-sig')
                region_rows.append(dict(region=region, source=source, method=method, set=part, **evaluation[part]))
            result = dict(region=region, source=source, method=method, selected=selected, metrics=evaluation,
                          seconds=time.perf_counter()-start, split_seed=summary['split_seed'],
                          input_sha256={p: hashlib.sha256((split_dir/f'{p}.csv').read_bytes()).hexdigest() for p in data},
                          gravity_file=str(gravity_path), gravity_sha256=hashlib.sha256(gravity_path.read_bytes()).hexdigest(),
                          training_unique_cells=len(cells), training_aggregation='median within train only',
                          gravity_units_assumed='mGal', bathymetry_units='m, positive upward',
                          selection='minimum validation RMSE; no refit on val; test evaluated after selection',
                          implementation='regional variants documented in TRADITIONAL_METHODS.md',
                          executable=sys.executable, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
            (directory/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(f'{region} {source} {method}: Test RMSE={evaluation["test"]["RMSE"]:.3f}, '
                  f'MAE={evaluation["test"]["MAE"]:.3f}; {result["seconds"]:.1f}s', flush=True)
    return region_rows


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-root', type=Path, default=ROOT/'data/dataset_splits/split_seed_42')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/result/traditional_methods')
    parser.add_argument('--region', choices=(*REGIONS, 'all'), default='all')
    parser.add_argument('--sources', nargs='+', choices=('sio', 'swot'), default=['sio', 'swot'])
    parser.add_argument('--methods', nargs='+', choices=('GGM', 'SAS', 'Parker'), default=['GGM', 'SAS', 'Parker'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    rows = []
    for region in REGIONS if args.region == 'all' else (args.region,):
        rows.extend(run(args, region))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_dir/'comparison_metrics.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(rows).query("set == 'test'").to_string(index=False))


if __name__ == '__main__':
    main()
