"""Six controlled architecture ablations, using training seed 123 and split seed 42."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from CDPA_split import RAW_REGIONS, save_comparison_csv
from gravity_models import (ARCHITECTURE_EXPERIMENTS, OPTIONAL_ARCHITECTURE_EXPERIMENTS,
                            architecture_config, make_model, parameter_count, conv_linear_macs)

ROOT = Path(__file__).resolve().parent
SPLIT_KEYS = ('keep_idx', 'train_idx', 'val_idx', 'test_idx', 'ship_row_indices', 'lat_idx', 'lon_idx')


def read_split(directory):
    with np.load(directory / 'CNN_cdpa_multiscale_split.npz') as data:
        return {key: data[key].copy() for key in SPLIT_KEYS}


def verify_split(actual, expected, label):
    if any(not np.array_equal(actual[key], expected[key]) for key in SPLIT_KEYS):
        raise RuntimeError(f'Sample/split mismatch: {label}')


def run_child(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'console.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((directory / 'console.log').read_text(encoding='utf-8', errors='replace')[-5000:], flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)


def summarize(root, groups, seed):
    rows = []
    specs_path = root / 'model_specifications.csv'
    specs = pd.read_csv(specs_path).set_index('experiment') if specs_path.exists() else None
    for group in groups:
        directory = root / f'seed_{seed}' / group
        if not (directory / 'completed.json').exists():
            continue
        result = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
        training = json.loads((directory / 'training_summary.json').read_text(encoding='utf-8'))
        completion = json.loads((directory / 'completed.json').read_text(encoding='utf-8'))
        config = architecture_config(group)
        for subset, metrics in result['metrics'].items():
            rows.append(dict(experiment=group, seed=seed, set=subset, channels=result['channels'],
                             parameters=result['parameters'], patches='/'.join(str(2*h+1) for h in config['halfs']),
                             conv_linear_MACs_per_sample=int(specs.loc[group, 'conv_linear_MACs_per_sample']) if specs is not None and 'conv_linear_MACs_per_sample' in specs.columns else None,
                             use_se=config['use_se'], use_skip=config['use_skip'],
                             best_epoch=training['best_epoch'], epochs_run=training['epochs_run'],
                             seconds=completion['seconds'], **metrics))
    if not rows:
        return
    save_comparison_csv(rows, root)
    test = pd.DataFrame(rows).query("set == 'Test'")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    for ax, metric in zip(axes, ('RMSE', 'MAE')):
        ax.bar(test['experiment'], test[metric])
        ax.set_ylabel(f'Test {metric} (m)')
        ax.tick_params(axis='x', rotation=30)
    fig.savefig(root / 'architecture_errors.png', dpi=200)
    plt.close(fig)
    print(test[['experiment', 'channels', 'parameters', 'RMSE', 'MAE', 'R2']].to_string(index=False), flush=True)


def main():
    # Parent process only: tiny CPU profiling passes do not benefit from many threads.
    # Training children retain their own default CPU/GPU configuration.
    torch.set_num_threads(1)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace', line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=tuple(RAW_REGIONS), default='170_175_31_37')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--groups', nargs='+', choices=(*ARCHITECTURE_EXPERIMENTS,
                        *OPTIONAL_ARCHITECTURE_EXPERIMENTS), default=list(ARCHITECTURE_EXPERIMENTS))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.groups)) != len(args.groups):
        parser.error('Epochs must be positive and groups must be unique')
    output = args.output_dir or ROOT / 'data' / 'result' / 'architecture_ablation' / args.region
    if args.summarize_only:
        summarize(output, args.groups, args.seed)
        return
    output.mkdir(parents=True, exist_ok=True)
    sources = ('CDPA_split.py', 'gravity_models.py', 'run_architecture_ablation.py')
    code_hash = hashlib.sha256(b''.join((ROOT / name).read_bytes() for name in sources)).hexdigest()
    config = dict(region=args.region, seed=args.seed, split_seed=args.split_seed, groups=args.groups,
                  epochs=args.epochs, device=args.device, check_data=args.check_data, code_hash=code_hash,
                  split_method='random', selection_metric='validation Huber loss')
    path = output / 'run_config.json'
    if path.exists():
        if not args.resume:
            raise FileExistsError(f'Run exists: {output}; use --resume or a new --output-dir')
        if json.loads(path.read_text(encoding='utf-8')) != config:
            raise ValueError('Configuration or code changed; use a new output directory')
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output / 'runner_pid.json').write_text(json.dumps(dict(pid=os.getpid())), encoding='utf-8')

    def command(group, base, check=False):
        result = [sys.executable, '-u', str(ROOT / 'CDPA_split.py'), f'--region={args.region}',
                  '--seed', str(args.seed), '--split-seed', str(args.split_seed),
                  '--experiment', group, '--comparison-cohort', '--epochs', str(args.epochs),
                  '--device', args.device, '--output-dir', str(base)]
        return result + ['--check-data'] if check else result

    # Validate all inputs and identical splits before the first expensive training run.
    reference = None
    specs = []
    for group in args.groups:
        base = output / '_validation'
        run_child(command(group, base, check=True), base / group)
        actual = read_split(base / group)
        if reference is None:
            reference = actual
        verify_split(actual, reference, group)
        manifest = json.loads((base / group / 'experiment.json').read_text(encoding='utf-8'))
        channels = len(manifest['input_channels'])
        model = make_model(group, channels)
        specs.append(dict(experiment=group, channels=channels, parameters=parameter_count(model),
                          conv_linear_MACs_per_sample=conv_linear_macs(model, channels, architecture_config(group)['halfs']),
                          **architecture_config(group)))
        del model
        print(f'VALIDATED {group}', flush=True)
    pd.DataFrame(specs).to_csv(output / 'model_specifications.csv', index=False)
    counts = {spec['experiment']: spec['parameters'] for spec in specs}
    for other in ('arch_same17', 'arch_no_skip', 'arch_raw12'):
        if 'arch_full' in counts and other in counts and counts[other] != counts['arch_full']:
            raise RuntimeError(f'Parameter control failed: {other}')
    if args.check_data:
        print(f'Data/parameter validation completed: {output}', flush=True)
        return
    for group in args.groups:
        base = output / f'seed_{args.seed}'
        directory = base / group
        done = directory / 'completed.json'
        if args.resume and done.exists():
            verify_split(read_split(directory), reference, group)
            print(f'SKIP completed {group}', flush=True)
            summarize(output, args.groups, args.seed)
            continue
        print(f'START {group} seed={args.seed} region={args.region}', flush=True)
        start = time.monotonic()
        run_child(command(group, base), directory)
        verify_split(read_split(directory), reference, group)
        done.write_text(json.dumps(dict(seconds=time.monotonic()-start)), encoding='utf-8')
        summarize(output, args.groups, args.seed)
        print(f'DONE {group}', flush=True)
    print(f'All architecture ablations completed: {output}', flush=True)


if __name__ == '__main__':
    main()
