"""Random-split five-group CDPA ablation with fixed sample splits."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from CDPA_split import COMPONENTS, RAW_REGIONS, resolve_raw_path, save_comparison_csv

GROUPS = ('raw8', 'cdpa12', 'cdpa16', 'cdpa16_split', 'raw8_matched')
ROOT = Path(__file__).resolve().parent


def reconstruction_report(region, directory):
    west, east, south, north = map(float, region.split('_'))
    def read(path):
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(float)
        frame.columns = frame.columns.astype(float)
        return frame.loc[(frame.index >= south) & (frame.index <= north),
                         (frame.columns >= west) & (frame.columns <= east)]
    results = []
    common_arrays = {}
    for source, prefix in (('sio', 'standard'), ('swot', 'swot')):
        for component in COMPONENTS:
            raw = read(resolve_raw_path(ROOT / 'data', source, component, region))
            common = read(ROOT / 'data' / region / f'{prefix}_{component}_common.csv')
            distinct = read(ROOT / 'data' / region / f'{prefix}_{component}_distinct.csv')
            if not (np.allclose(raw.index, common.index, rtol=0, atol=1e-6) and
                    np.allclose(raw.columns, common.columns, rtol=0, atol=1e-6) and
                    common.index.equals(distinct.index) and common.columns.equals(distinct.columns)):
                raise ValueError(f'Reconstruction grid mismatch: {source} {component}')
            x, c, d = (a.to_numpy(dtype=float).ravel() for a in (raw, common, distinct))
            mask = np.isfinite(x) & np.isfinite(c) & np.isfinite(d)
            x, c, d = x[mask], c[mask], d[mask]
            if not len(x):
                raise ValueError('No valid reconstruction samples')
            common_arrays[(source,component)] = common.to_numpy(dtype=float)
            reconstructed = c + d
            rmse = float(np.sqrt(np.mean((reconstructed-x)**2)))
            std = float(x.std())
            row = dict(source=source, component=component, n=len(x), raw_mean=float(x.mean()),
                       raw_std=std, common_std=float(c.std()), distinct_std=float(d.std()),
                       reconstruction_rmse=rmse, rmse_over_raw_std=rmse/max(std,1e-12),
                       correlation=float(np.corrcoef(x,reconstructed)[0,1]) if std>0 and reconstructed.std()>0 else None)
            results.append(row)
    differences = []
    for component in COMPONENTS:
        difference = common_arrays[('sio',component)] - common_arrays[('swot',component)]
        differences.append(dict(component=component,
                                common_difference_mean=float(np.nanmean(difference)),
                                common_difference_std=float(np.nanstd(difference)),
                                common_difference_rmse=float(np.sqrt(np.nanmean(difference**2)))))
    pd.DataFrame(differences).to_csv(directory / 'common_differences.csv', index=False)
    pd.DataFrame(results).to_csv(directory / 'reconstruction.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    return results


def summarize(directory, records):
    if not records:
        return
    table = pd.DataFrame(records)
    save_comparison_csv(records, directory)
    metrics = ['RMSE','MAE','R2']
    summary = table.groupby(['experiment','set'])[metrics].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(directory / 'ablation_mean_std.csv', index=False, encoding='utf-8-sig')
    import matplotlib.pyplot as plt
    test = table[table['set'] == 'Test'].groupby('experiment')['RMSE'].agg(['mean','std']).reindex(GROUPS).dropna(subset=['mean'])
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(test.index, test['mean'], yerr=test['std'].fillna(0), capsize=4)
    ax.set_ylabel('Test RMSE (m); mean +/- sample std across seeds')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(directory / 'ablation_test_rmse.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=tuple(RAW_REGIONS), default='170_175_31_37')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,123,2026])
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', choices=('cpu','cuda','auto'), default='cuda')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    directory = args.output_dir or ROOT / 'data' / args.region / 'ablations'
    directory.mkdir(parents=True, exist_ok=True)
    diagnostics = reconstruction_report(args.region, directory)
    large = [r for r in diagnostics if r['rmse_over_raw_std'] > 0.01]
    if large:
        print('NOTE: reconstruction differs from raw by over 1% of raw std; inspect reconstruction.csv before interpreting decomposition gains.', flush=True)
    code_hash = hashlib.sha256(b''.join((ROOT / name).read_bytes() for name in
                                       ('CDPA_split.py','gravity_models.py','run_ablation.py'))).hexdigest()
    config = dict(region=args.region,seeds=args.seeds,split_seed=args.split_seed,
                  epochs=args.epochs,device=args.device,groups=GROUPS,code_hash=code_hash,
                  check_data=args.check_data)
    config_path = directory / 'run_config.json'
    if args.resume and config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Resume configuration/code differs; use a new output directory')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    records, reference = [], None
    for seed in args.seeds:
        for group in GROUPS:
            base = directory / f'seed_{seed}'
            output = base / group
            done = output / 'completed.json'
            command = [sys.executable,'-u',str(ROOT/'CDPA_split.py'),f'--region={args.region}',
                       '--experiment',group,'--comparison-cohort','--seed',str(seed),
                       '--split-seed',str(args.split_seed),'--epochs',str(args.epochs),
                       '--device',args.device,'--output-dir',str(base)]
            if args.check_data:
                command.append('--check-data')
            start = time.monotonic()
            if not (args.resume and done.exists()):
                output.mkdir(parents=True,exist_ok=True)
                if done.exists():
                    done.unlink()
                print(f'START {group} seed={seed}',flush=True)
                with (output/'console.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                done.write_text(json.dumps(dict(seconds=time.monotonic()-start)),encoding='utf-8')
            with np.load(output/'CNN_cdpa_multiscale_split.npz') as split:
                current = {key:split[key].copy() for key in
                           ('keep_idx','train_idx','val_idx','test_idx','ship_row_indices')}
            if reference is None:
                reference = current
            elif any(not np.array_equal(current[key],reference[key]) for key in current):
                raise RuntimeError(f'Split mismatch: {group}, seed={seed}')
            if not args.check_data:
                elapsed=json.loads(done.read_text())['seconds']
                training=json.loads((output/'training_summary.json').read_text())
                result=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
                for subset, values in result['metrics'].items():
                    records.append(dict(experiment=group,seed=seed,set=subset,
                                        parameters=result['parameters'],seconds=elapsed,best_epoch=training['best_epoch'], **values))
                summarize(directory,records)
            print(f'DONE {group} seed={seed}',flush=True)
    print(f'All ablations completed: {directory}',flush=True)


if __name__ == '__main__':
    main()
