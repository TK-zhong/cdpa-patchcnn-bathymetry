import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import sys
import json
import argparse
import subprocess
from gravity_models import (make_model, architecture_config, ARCHITECTURE_EXPERIMENTS,
                            OPTIONAL_ARCHITECTURE_EXPERIMENTS)
from datetime import datetime
import uuid
from pathlib import Path

# 设置编码
def configure_console_encoding():
    # Reconfigure existing streams; wrapping their buffers can close shared handles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace', line_buffering=True)


EXPERIMENTS = ('sio_faa', 'swot_faa', 'sio_swot_faa',
               'sio', 'swot', 'sio_swot', 'cdpa')
ABLATIONS = ('raw8', 'cdpa12', 'cdpa16', 'cdpa16_split', 'raw8_matched')
COMPONENTS = ('faa', 'vgg', 'east', 'north')
RAW_REGIONS = {'-165_-160_30_35': '-170_-160_30_40',
               '131_136_1_6': '130_140_0_10',
               '170_175_31_37': '170_180_30_40'}


def resolve_raw_path(data_dir, source, component, region):
    # SIO gravity grids can be stored under either the current faa or legacy grav folder.
    folders = ('faa', 'grav') if source == 'sio' and component == 'faa' else (component,)
    candidates = [data_dir / source / folder / f'z_{RAW_REGIONS[region]}.csv' for folder in folders]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('Missing raw gravity grid. Checked:\n' + '\n'.join(map(str, candidates)))


def build_experiment_features(cdpa, raw, experiment):
    if experiment == 'arch_raw12':
        names = [f'{source}_{c}' for source in ('sio', 'swot') for c in COMPONENTS]
        arrays = [raw[name] for name in names]
        names += [f'raw_avg_{c}' for c in COMPONENTS]
        arrays += [(raw[f'sio_{c}'] + raw[f'swot_{c}']) / 2 for c in COMPONENTS]
        return np.stack(arrays), names
    if experiment.startswith('arch_'):
        experiment = 'raw8' if experiment == 'arch_raw8' else 'cdpa12'
    if experiment in ('cdpa16', 'cdpa16_split'):
        names = [f'{source}_{c}_{part}' for part in ('common', 'distinct')
                 for source in ('standard', 'swot') for c in COMPONENTS]
        return np.stack([cdpa[name] for name in names]), names
    if experiment in ('raw8', 'raw8_matched'):
        experiment = 'sio_swot'
    if experiment in ('cdpa', 'cdpa12'):
        names = ([f'avg_{c}_common' for c in COMPONENTS] +
                 [f'standard_{c}_distinct' for c in COMPONENTS] +
                 [f'swot_{c}_distinct' for c in COMPONENTS])
        arrays = ([(cdpa[f'standard_{c}_common'] + cdpa[f'swot_{c}_common']) / 2
                   for c in COMPONENTS] +
                  [cdpa[f'standard_{c}_distinct'] for c in COMPONENTS] +
                  [cdpa[f'swot_{c}_distinct'] for c in COMPONENTS])
    else:
        sources = ('sio', 'swot') if experiment.startswith('sio_swot') else (experiment.split('_')[0],)
        components = ('faa',) if experiment.endswith('_faa') else COMPONENTS
        names = [f'{source}_{c}' for source in sources for c in components]
        arrays = [raw[name] for name in names]
    return np.stack(arrays, axis=0), names


def save_comparison_csv(summaries, directory):
    path = directory / 'comparison_metrics.csv'
    table = pd.DataFrame(summaries)
    try:
        table.to_csv(path, index=False, encoding='utf-8-sig')
    except PermissionError:
        # A spreadsheet may keep the original file locked on Windows.
        path = directory / f"comparison_metrics_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.csv"
        table.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'Original summary is locked or unwritable; saved to {path}')
    return path


def main():
    configure_console_encoding()
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    data_dir = Path(__file__).resolve().parent / 'data'
    parser = argparse.ArgumentParser(description='Gravity feature comparison experiments')
    parser.add_argument('--region', choices=tuple(RAW_REGIONS), default='170_175_31_37')
    parser.add_argument('--cdpa-dir', type=Path)
    parser.add_argument('--ship-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--experiment', choices=(*EXPERIMENTS, *ABLATIONS,
                        *ARCHITECTURE_EXPERIMENTS, *OPTIONAL_ARCHITECTURE_EXPERIMENTS, 'all'), default='cdpa')
    parser.add_argument('--comparison-cohort', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--summarize-only', action='store_true',
                        help='Rebuild summary from saved metrics without training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()
    if args.experiment.startswith('arch_'):
        args.comparison_cohort = True
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    print(f'Training device: {device}')
    cdpa_dir = args.cdpa_dir or data_dir / args.region
    SHIP_PATH = args.ship_path or data_dir / 'ship' / 'single' / f'{RAW_REGIONS[args.region]}.csv'
    lon_min, lon_max, lat_min, lat_max = map(float, args.region.split('_'))
    lon_range, lat_range = (lon_min, lon_max), (lat_min, lat_max)
    region_label = f'{lat_min:g}-{lat_max:g} latitude, {lon_min:g}-{lon_max:g} longitude'
    base_output_dir = args.output_dir or cdpa_dir / 'comparisons'
    output_dir = base_output_dir / args.experiment
    if args.summarize_only:
        summaries = []
        reference = None
        for experiment in EXPERIMENTS:
            directory = base_output_dir / experiment
            metrics_path = directory / 'metrics.json'
            if not metrics_path.is_file():
                print(f'Skipping missing result: {experiment}')
                continue
            with np.load(directory / 'CNN_cdpa_multiscale_split.npz') as split:
                current = {key: split[key].copy() for key in ('keep_idx', 'train_idx', 'val_idx', 'test_idx', 'ship_row_indices')}
            if reference is None:
                reference = current
            elif any(not np.array_equal(current[key], reference[key]) for key in current):
                raise RuntimeError(f'Comparison splits differ: {experiment}')
            result = json.loads(metrics_path.read_text(encoding='utf-8'))
            for subset, metrics in result['metrics'].items():
                summaries.append(dict(experiment=experiment, set=subset, channels=result['channels'],
                                      parameters=result['parameters'], **metrics))
        if not summaries:
            raise FileNotFoundError(f'No saved metrics under {base_output_dir}')
        path = save_comparison_csv(summaries, base_output_dir)
        print(f'Summary saved: {path}')
        print(pd.DataFrame(summaries).query("set == 'Test'").to_string(index=False))
        return
    if args.experiment == 'all':
        # Fail before any long training run if the comparison inputs are incomplete.
        for source in ('sio', 'swot'):
            for component in COMPONENTS:
                resolve_raw_path(data_dir, source, component, args.region)
        required = [SHIP_PATH] + [cdpa_dir / f'{source}_{c}_{part}.csv'
                   for source in ('standard', 'swot') for c in COMPONENTS
                   for part in ('common', 'distinct')]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing comparison inputs:\n' + '\n'.join(missing))
        base_output_dir.mkdir(parents=True, exist_ok=True)
        summaries, reference = [], None
        for experiment in EXPERIMENTS:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), f'--region={args.region}',
                       '--cdpa-dir', str(cdpa_dir), '--ship-path', str(SHIP_PATH),
                       '--output-dir', str(base_output_dir), '--epochs', str(args.epochs),
                       '--seed', str(args.seed), '--split-seed', str(args.split_seed),
                       '--device', args.device, '--experiment', experiment, '--comparison-cohort']
            if args.check_data:
                command.append('--check-data')
            subprocess.run(command, check=True)
            directory = base_output_dir / experiment
            with np.load(directory / 'CNN_cdpa_multiscale_split.npz') as split:
                current = {key: split[key].copy() for key in ('keep_idx', 'train_idx', 'val_idx', 'test_idx', 'ship_row_indices')}
            if reference is None:
                reference = current
            elif any(not np.array_equal(current[key], reference[key]) for key in current):
                raise RuntimeError('Comparison splits differ')
            if not args.check_data:
                result = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
                for subset, metrics in result['metrics'].items():
                    summaries.append(dict(experiment=experiment, set=subset, channels=result['channels'],
                                          parameters=result['parameters'], **metrics))
                save_comparison_csv(summaries, base_output_dir)
        if summaries:
            test = pd.DataFrame(summaries).query("set == 'Test'")
            print(test.to_string(index=False))
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, metric in zip(axes, ('RMSE', 'MAE')):
                ax.bar(test['experiment'], test[metric])
                ax.set_ylabel(f'{metric} (m)')
                ax.tick_params(axis='x', rotation=35)
            fig.tight_layout()
            fig.savefig(base_output_dir / 'comparison_errors.png', dpi=200)
            plt.close(fig)
        print(f'Comparison completed: {base_output_dir}')
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_cdpa_result(path):
        df = pd.read_csv(path, index_col=0, encoding='utf-8-sig')
        lat_g, lon_g = df.index.to_numpy(dtype=np.float32), df.columns.to_numpy(dtype=np.float32)
        for coordinates in (lat_g, lon_g):
            if not np.isfinite(coordinates).all() or not (np.all(np.diff(coordinates)>0) or np.all(np.diff(coordinates)<0)):
                raise ValueError(f'Invalid grid coordinates: {path}')
        rows = (lat_g >= lat_min) & (lat_g <= lat_max)
        cols = (lon_g >= lon_min) & (lon_g <= lon_max)
        if not rows.any() or not cols.any():
            raise ValueError(f'No grid coverage: {path}')
        values = df.to_numpy(dtype=np.float32)[rows][:,cols]
        print(f'{path}: {values.shape}')
        return values, lat_g[rows], lon_g[cols]

    def load_ship_data(path):
        df = pd.read_csv(path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.lower()
        for column in ('longitude', 'latitude', 'bathymetry'):
            df[column] = pd.to_numeric(df[column], errors='coerce')
        mask = (df['longitude'].between(lon_min,lon_max) & df['latitude'].between(lat_min,lat_max)
                & np.isfinite(df['bathymetry']))
        df = df.loc[mask]
        return (df['latitude'].to_numpy(dtype=np.float32), df['longitude'].to_numpy(dtype=np.float32),
                df['bathymetry'].to_numpy(dtype=np.float32), df.index.to_numpy())

    components = list(COMPONENTS)
    feature_data, raw_data = {}, {}
    lat_grid = lon_grid = None

    def register_grid(path):
        nonlocal lat_grid, lon_grid
        data, lat_g, lon_g = load_cdpa_result(path)
        if lat_grid is None:
            lat_grid, lon_grid = lat_g, lon_g
        elif not (np.array_equal(lat_grid, lat_g) and np.array_equal(lon_grid, lon_g)):
            raise ValueError(f'Feature coordinates differ: {path}')
        return data

    if args.experiment.startswith('cdpa') or args.comparison_cohort:
        for source in ('standard', 'swot'):
            for component in components:
                for part in ('common', 'distinct'):
                    name = f'{source}_{component}_{part}'
                    feature_data[name] = register_grid(cdpa_dir / f'{name}.csv')
    if not args.experiment.startswith('cdpa') or args.comparison_cohort:
        sources = ('sio', 'swot') if args.comparison_cohort or args.experiment in ('raw8', 'raw8_matched') or args.experiment.startswith('sio_swot') else (args.experiment.split('_')[0],)
        selected_components = components if args.comparison_cohort or not args.experiment.endswith('_faa') else ['faa']
        for source in sources:
            for component in selected_components:
                path = resolve_raw_path(data_dir, source, component, args.region)
                raw_data[f'{source}_{component}'] = register_grid(path)
    input_data, input_channels = build_experiment_features(feature_data, raw_data, args.experiment)
    if args.comparison_cohort:
        # Any invalid value in any candidate feature excludes the sample for every experiment.
        all_finite = np.logical_and.reduce([np.isfinite(a) for a in
                                           [*feature_data.values(), *raw_data.values()]])
        cohort_data = np.where(all_finite, 0.0, np.nan)[None].astype(np.float32)
    else:
        cohort_data = input_data
    del feature_data, raw_data
    N_CHANNELS = input_data.shape[0]
    H, W = input_data.shape[1], input_data.shape[2]
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    print(f"输入特征维度: {input_data.shape}")

    # ── 3. 船测点映射 ──────────────────────────────────────────────────────────────

    ship_lat, ship_lon, ship_depth, ship_rows = load_ship_data(
        SHIP_PATH
    )

    if len(ship_depth) == 0:
        raise ValueError(f"No finite ship measurements in {args.region}; supply matching --ship-path.")

    def nearest_indices(grid, points):
        order = np.argsort(grid)
        sorted_grid = grid[order]
        right = np.clip(np.searchsorted(sorted_grid, points), 0, len(grid) - 1)
        left = np.maximum(right - 1, 0)
        chosen = np.where(np.abs(sorted_grid[left] - points) <=
                          np.abs(sorted_grid[right] - points), left, right)
        return order[chosen]

    lat_idx = nearest_indices(lat_grid, ship_lat)
    lon_idx = nearest_indices(lon_grid, ship_lon)
    print(f"船测点映射完成，共 {len(ship_depth)} 个有效点")

    # ── 4. 多尺度 patch 提取（以最大 HALF 控制边界）─────────────────────────────────

    HALFS = architecture_config(args.experiment)['halfs']
    HALF_MAX = max(12, max(HALFS))  # Same valid cohort, including single/same-scale runs.

    def valid_patch_indices(data, rows, cols, half):
        bad = (~np.isfinite(data).all(axis=0)).astype(np.int64)
        integral = np.pad(bad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        inside = ((rows >= half) & (rows < data.shape[1] - half) &
                  (cols >= half) & (cols < data.shape[2] - half))
        keep = np.flatnonzero(inside)
        r, c = rows[keep], cols[keep]
        counts = (integral[r + half + 1, c + half + 1] - integral[r - half, c + half + 1]
                  - integral[r + half + 1, c - half] + integral[r - half, c - half])
        return keep[counts == 0].astype(np.int64)


    keep_idx = valid_patch_indices(cohort_data, lat_idx, lon_idx, HALF_MAX)
    y_raw    = ship_depth[keep_idx]
    lat_keep = lat_idx[keep_idx]
    lon_keep = lon_idx[keep_idx]
    print(f"有效图像块数量: {len(y_raw)}（去除边界后）")

    if len(y_raw) == 0:
        print("错误：无有效图像块，请检查数据覆盖情况")
        sys.exit(1)


    # ── 5. 数据集划分（随机样本划分，70/15/15）────────────────────────────────────

    idx_all = np.arange(len(y_raw))
    if len(idx_all) < 4:
        raise ValueError("At least four valid samples are required for train/val/test splitting.")
    train_idx, temp_idx = train_test_split(
        idx_all, test_size=0.30, random_state=args.split_seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=args.split_seed, shuffle=True
    )
    np.savez(f"{output_dir}/CNN_cdpa_multiscale_split.npz", train_idx=train_idx,
             val_idx=val_idx, test_idx=test_idx, keep_idx=keep_idx,
             lat_idx=lat_keep, lon_idx=lon_keep, seed=SEED, split_seed=args.split_seed,
             split_method='random', ship_row_indices=ship_rows[keep_idx],
             experiment=args.experiment, comparison_cohort=args.comparison_cohort)
    print(f"Random split (approximately 70/15/15): "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    manifest = {'experiment': args.experiment, 'input_channels': input_channels,
                'architecture': architecture_config(args.experiment),
                'region': args.region, 'ship_path': str(SHIP_PATH.resolve()),
                'comparison_cohort': args.comparison_cohort, 'seed': SEED, 'split_seed': args.split_seed,
                'samples': {name: len(idx) for name, idx in
                            [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]}}
    with (output_dir / 'experiment.json').open('w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2)
    if args.check_data:
        print(f'Data validation completed: {output_dir}')
        return

    # Normalization statistics are fitted only on the training samples.

    def raw_patch(index, half):
        r, c = lat_keep[index], lon_keep[index]
        return input_data[:, r-half:r+half+1, c-half:c+half+1]

    X_means, X_stds = [], []
    for half in HALFS:
        count = 0
        mean = np.zeros(N_CHANNELS, dtype=np.float64)
        m2 = np.zeros(N_CHANNELS, dtype=np.float64)
        for index in train_idx:
            values = raw_patch(index, half).reshape(N_CHANNELS, -1).astype(np.float64)
            n = values.shape[1]
            batch_mean = values.mean(axis=1)
            delta = batch_mean - mean
            total = count + n
            m2 += ((values - batch_mean[:, None]) ** 2).sum(axis=1) + delta**2 * count * n / total
            mean += delta * n / total
            count = total
        X_means.append(mean.astype(np.float32).reshape(1, -1, 1, 1))
        X_stds.append(np.maximum(np.sqrt(m2 / count), 1e-8).astype(np.float32).reshape(1, -1, 1, 1))

    y_mean = float(y_raw[train_idx].mean())
    y_std  = float(y_raw[train_idx].std()) + 1e-8

    def norm_y(y):   return (y - y_mean) / y_std
    def denorm_y(y): return y * y_std + y_mean

    y_scaled = norm_y(y_raw)


    # ── 7. Dataset & DataLoader ───────────────────────────────────────────────────

    class MultiScaleDataset(Dataset):
        def __init__(self, y, idx):
            self.idx = np.asarray(idx, dtype=np.int64)
            self.y = torch.from_numpy(np.asarray(y[idx], dtype=np.float32))
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            patches = [torch.from_numpy((raw_patch(self.idx[i], half) - X_means[s][0]) / X_stds[s][0])
                       for s, half in enumerate(HALFS)]
            return patches, self.y[i]


    def collate_fn(batch):
        patches_batch = [torch.stack([item[0][s] for item in batch]) for s in range(len(HALFS))]
        y_batch = torch.stack([item[1] for item in batch])
        return patches_batch, y_batch


    train_loader = DataLoader(
        MultiScaleDataset(y_scaled, train_idx),
        batch_size=64, shuffle=True, drop_last=False, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        MultiScaleDataset(y_scaled, val_idx),
        batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        MultiScaleDataset(y_scaled, test_idx),
        batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    # ── 8. 模型：SE通道注意力 + 多尺度ResNet分支 ──────────────────────────────────



    # ── 9. 训练 ───────────────────────────────────────────────────────────────────

    criterion = nn.HuberLoss(delta=1.0)

    def train_model(model, train_loader, val_loader, num_epochs=200):
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        best_val  = float('inf')
        patience, no_improve = 30, 0
        train_losses, val_losses = [], []
        history = []
        best_epoch = 0
        train_eval_loader = DataLoader(train_loader.dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        model_path = f"{output_dir}/CNN_cdpa_multiscale_best.pth"

        for epoch in range(num_epochs):
            model.train()
            t_loss = []
            for patches_batch, by in train_loader:
                patches_batch = [p.to(device) for p in patches_batch]
                by = by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(patches_batch), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss.append((loss.item(), len(by)))

            model.eval()
            v_loss = []
            with torch.no_grad():
                for patches_batch, by in val_loader:
                    patches_batch = [p.to(device) for p in patches_batch]
                    by = by.to(device)
                    v_loss.append((criterion(model(patches_batch), by).item(), len(by)))

            tl = sum(loss * n for loss, n in t_loss) / sum(n for _, n in t_loss)
            vl = sum(loss * n for loss, n in v_loss) / sum(n for _, n in v_loss)
            if not np.isfinite(tl) or not np.isfinite(vl):
                raise RuntimeError("Non-finite training loss; stopped before loading a stale checkpoint.")
            with torch.no_grad():
                eval_total, eval_square, eval_count, val_square = 0.0, 0.0, 0, 0.0
                for patches, targets in train_eval_loader:
                    prediction = model([p.to(device) for p in patches])
                    targets = targets.to(device)
                    eval_total += criterion(prediction, targets).item() * len(targets)
                    eval_square += ((prediction-targets)**2).sum().item()
                    eval_count += len(targets)
                for patches, targets in val_loader:
                    prediction = model([p.to(device) for p in patches])
                    val_square += ((prediction-targets.to(device))**2).sum().item()
            history.append(dict(epoch=epoch+1, train_loss=tl, train_eval_loss=eval_total/eval_count,
                                val_loss=vl, train_rmse=np.sqrt(eval_square/eval_count)*y_std,
                                val_rmse=np.sqrt(val_square/len(val_loader.dataset))*y_std,
                                lr=optimizer.param_groups[0]['lr']))
            pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
            train_losses.append(tl)
            val_losses.append(vl)
            scheduler.step(vl)

            if vl < best_val:
                best_val   = vl
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                no_improve += 1

            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d} | Train: {tl:.4f} | Val: {vl:.4f} | LR: {lr:.2e}", flush=True)

            if no_improve >= patience:
                print(f"早停触发 @ epoch {epoch+1}")
                break

        with (output_dir / 'training_summary.json').open('w', encoding='utf-8') as file:
            json.dump(dict(best_epoch=best_epoch, best_val_loss=best_val, epochs_run=len(history)), file, indent=2)
        plt.figure(figsize=(10, 6))
        plt.plot([h['train_eval_loss'] for h in history], label='Train Eval Loss')
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses,   label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Huber Loss')
        plt.title('Multi-Scale PatchCNN Training History')
        plt.legend()
        plt.grid()
        plt.savefig(f"{output_dir}/CNN_cdpa_multiscale_training.png", dpi=300, bbox_inches='tight')
        plt.close()
        return model_path


    # ── 10. 整图滑窗预测 ───────────────────────────────────────────────────────────

    def predict_entire_map(model, data, halfs=HALFS, batch_size=64):
        _, height, width = data.shape
        half_max = max(12, max(halfs))
        output = np.full((height, width), np.nan, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for i in range(half_max, height - half_max):
                strip = data[:, i-half_max:i+half_max+1, :]
                candidate_cols = np.arange(half_max, width - half_max)
                keep = valid_patch_indices(strip, np.full(len(candidate_cols), half_max),
                                           candidate_cols, half_max)
                valid_cols = candidate_cols[keep]
                for start in range(0, len(valid_cols), batch_size):
                    cols = valid_cols[start:start + batch_size]
                    batch = []
                    for scale, half in enumerate(halfs):
                        patches = np.stack([data[:, i-half:i+half+1, j-half:j+half+1] for j in cols])
                        normalized = (patches - X_means[scale]) / X_stds[scale]
                        batch.append(torch.from_numpy(normalized).to(device))
                    output[i, cols] = denorm_y(model(batch).cpu().numpy())
        return output


    # ── 11. 可视化函数（保持原样，仅去掉重复 print）────────────────────────────────

    def plot_2d(lon_mesh, lat_mesh, data, title, filename, cmap='gist_earth', vmin=None, vmax=None):
        plt.figure(figsize=(12, 8))
        valid = ~np.isnan(data)
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        data_crop = data[r0:r1+1, c0:c1+1]
        lat_vals = lat_mesh[:, 0]
        lon_vals = lon_mesh[0, :]
        if lat_vals[0] > lat_vals[-1]:
            origin = 'upper'
            extent = [lon_vals[c0], lon_vals[c1], lat_vals[r1], lat_vals[r0]]
        else:
            origin = 'lower'
            extent = [lon_vals[c0], lon_vals[c1], lat_vals[r0], lat_vals[r1]]
        p = plt.imshow(data_crop, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin=origin, interpolation='none', aspect='auto')
        plt.colorbar(p, label='Depth (m)')
        plt.xlabel('Longitude (°E)')
        plt.ylabel('Latitude (°N)')
        plt.title(title)
        plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()


    def plot_3d(lon_mesh, lat_mesh, depth, title, filename):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(14, 10))
        ax  = fig.add_subplot(111, projection='3d')
        step = max(1, min(lon_mesh.shape) // 100)

        depth_filled = depth.copy()
        lm   = lon_mesh[::step, ::step]
        latm = lat_mesh[::step, ::step]
        dm   = depth_filled[::step, ::step]
        z_min = float(np.nanmin(depth))
        z_max = float(np.nanmax(depth))

        side_color = plt.get_cmap('gist_earth')(0.02)

        surf = ax.plot_surface(lm, latm, dm,
                               cmap='gist_earth', edgecolor='none',
                               antialiased=False, vmin=z_min, vmax=z_max)

        bx = np.array([[lm[0,0],  lm[0,-1]],  [lm[-1,0],  lm[-1,-1]]])
        by = np.array([[latm[0,0],latm[0,-1]], [latm[-1,0],latm[-1,-1]]])
        ax.plot_surface(bx, by, np.full((2,2), z_min),
                        color=side_color, edgecolor='none')

        def add_wall(xs, ys, zs_top):
            if not np.isfinite(zs_top).all():
                return
            n = len(xs)
            verts = [list(zip(xs, ys, zs_top)) +
                     list(zip(xs[::-1], ys[::-1], [z_min] * n))]
            poly = Poly3DCollection(verts, color=[side_color], linewidth=0)
            ax.add_collection3d(poly)

        add_wall(lm[0, :],  latm[0, :],  dm[0, :])
        add_wall(lm[-1, :], latm[-1, :], dm[-1, :])
        add_wall(lm[:, 0],  latm[:, 0],  dm[:, 0])
        add_wall(lm[:, -1], latm[:, -1], dm[:, -1])

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor('none')
            axis.line.set_lw(0.)
        ax.grid(False)
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.set_zlabel('')

        ax.view_init(elev=30, azim=225)
        ax.set_xlabel('Longitude (°E)', labelpad=12)
        ax.set_ylabel('Latitude (°N)', labelpad=12)
        ax.set_xlim(*lon_range)
        ax.set_ylim(*lat_range)
        ax.set_xticks(np.linspace(lon_min, lon_max, 6))
        ax.set_yticks(np.linspace(lat_min, lat_max, 6))
        ax.set_title(title, pad=20, fontsize=13)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Depth (m)')
        plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D图已保存: {filename}")


    # ── 12. 评估指标函数 ──────────────────────────────────────────────────────────

    def compute_metrics(pred, true):
        """pred, true 为一维数组，NaN 点自动剔除"""
        mask = np.isfinite(pred) & np.isfinite(true)
        p = pred[mask]
        t = true[mask]
        if len(t) == 0:
            return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'N': 0}
        mse  = float(np.mean((p - t) ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(p - t)))
        ss_res = float(np.sum((p - t) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'N': int(len(t))}


    # ── 13. 主流程 ────────────────────────────────────────────────────────────────

    print(f"\n=== 开始训练 Multi-Scale PatchCNN (SE + ResNet, {N_CHANNELS}通道) ===")
    print(f'Experiment: {args.experiment}; channels: {input_channels}')
    model = make_model(args.experiment, N_CHANNELS).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"多尺度 HALF: {HALFS}  →  patch 尺寸: {[2*h+1 for h in HALFS]}")

    model_path = train_model(model, train_loader, val_loader, num_epochs=args.epochs)

    # 保存归一化统计量
    norm_stats = {
        "region": args.region,
        "lon_range": lon_range,
        "lat_range": lat_range,
        "cdpa_dir": str(cdpa_dir.resolve()),
        "halfs": HALFS,
        "architecture": architecture_config(args.experiment),
        "input_channels": input_channels,
        "experiment": args.experiment,
        "comparison_cohort": args.comparison_cohort,
        "seed": SEED, "split_seed": args.split_seed,
        "split_method": "random",
        "y_mean": y_mean,
        "y_std":  y_std,
        "X_means": [m.tolist() for m in X_means],
        "X_stds":  [s.tolist() for s in X_stds],
    }
    stats_path = f"{output_dir}/CNN_cdpa_multiscale_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f)
    print(f"归一化统计量已保存: {stats_path}")

    print("\n=== 加载最佳模型进行整图预测 ===")
    model.load_state_dict(torch.load(model_path, map_location=device))
    predicted_depth = predict_entire_map(model, input_data)
    if not np.isfinite(predicted_depth).any():
        raise RuntimeError("Map prediction contains no finite values.")

    np.save(f"{output_dir}/CNN_cdpa_multiscale_predicted.npy", predicted_depth)
    df_pred = pd.DataFrame(predicted_depth, index=lat_grid, columns=lon_grid)
    df_pred.to_csv(f"{output_dir}/CNN_cdpa_multiscale_predicted.csv", encoding='utf-8')
    print(f"预测水深网格已保存: CNN_cdpa_multiscale_predicted.npy / .csv")

    # ── 14. 评估（只在验证集和测试集上计算）──────────────────────────────────────

    # 验证集预测
    val_pred = predicted_depth[lat_keep[val_idx], lon_keep[val_idx]]
    val_true = y_raw[val_idx]
    val_metrics = compute_metrics(val_pred, val_true)

    # 测试集预测
    test_pred = predicted_depth[lat_keep[test_idx], lon_keep[test_idx]]
    test_true = y_raw[test_idx]
    test_metrics = compute_metrics(test_pred, test_true)

    # 训练集预测（仅用于诊断是否过拟合，不报告为最终结果）
    train_pred = predicted_depth[lat_keep[train_idx], lon_keep[train_idx]]
    train_true = y_raw[train_idx]
    train_metrics = compute_metrics(train_pred, train_true)

    print("\n=== 评估指标（随机划分）===")
    print(f"{'Set':<8} {'N':>6} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    for name, m in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        print(f"{name:<8} {m['N']:>6d} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")

    result = {'experiment': args.experiment, 'channels': N_CHANNELS,
              'input_channels': input_channels, 'parameters': sum(p.numel() for p in model.parameters()),
              'metrics': {'Train': train_metrics, 'Val': val_metrics, 'Test': test_metrics}}
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)

    # 将指标写入文件（最终报告以 Test 为准）
    with open(f"{output_dir}/CNN_cdpa_multiscale_evaluation.txt", 'w', encoding='utf-8') as f:
        f.write("Multi-Scale PatchCNN (SE + ResNet) Bathymetry Evaluation\n")
        f.write("---------------------------------------------------------\n")
        f.write("Dataset split: random samples (approximately 70/15/15)\n")
        f.write(f"Random seed: {SEED}\n\n")
        f.write(f"{'Set':<8} {'N':>6} {'MSE':>12} {'RMSE':>10} {'MAE':>10} {'R2':>10}\n")
        for name, m in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
            f.write(f"{name:<8} {m['N']:>6d} {m['MSE']:>12.4f} {m['RMSE']:>10.4f} "
                    f"{m['MAE']:>10.4f} {m['R2']:>10.4f}\n")
        f.write(f"\nRegion: {region_label}\n")
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Input: {N_CHANNELS} channels: {', '.join(input_channels)}\n")
        f.write(f"Model: Multi-Scale PatchCNN, HALF={HALFS}, "
                f"patch sizes={[2*h+1 for h in HALFS]}\n")
        f.write(f"       SE channel attention + ResNet blocks + feature fusion\n")

    print(f"\n所有结果已保存到 {output_dir}")

    # ── 15. 可视化预测结果 ────────────────────────────────────────────────────────

    vmin = min(np.nanmin(predicted_depth), float(np.nanmin(test_true)))
    vmax = max(np.nanmax(predicted_depth), float(np.nanmax(test_true)))

    plot_2d(lon_mesh, lat_mesh, predicted_depth,
            f'Multi-Scale PatchCNN [{args.experiment}] Predicted Bathymetry ({region_label})',
            'CNN_cdpa_multiscale_predicted.png', vmin=vmin, vmax=vmax)

    plot_3d(lon_mesh, lat_mesh, predicted_depth,
            f'Multi-Scale PatchCNN [{args.experiment}] Predicted Bathymetry ({region_label})',
            'CNN_cdpa_multiscale_predicted_3d.png')

    # ── 16. 剖面图（只使用测试集内的航线，避免泄漏）───────────────────────────────

    print("\n=== 生成剖面图 ===")

    df_all = pd.read_csv(
        SHIP_PATH,
        encoding='utf-8'
    )
    df_all.columns = df_all.columns.str.strip().str.lower()
    if 'survey_name' not in df_all.columns:
        print("Missing survey_name; skipping profile plots.")
        return
    for column in ('longitude', 'latitude', 'bathymetry'):
        df_all[column] = pd.to_numeric(df_all[column], errors='coerce')

    df_region = df_all[
        (df_all['longitude'] >= lon_min) & (df_all['longitude'] <= lon_max) &
        (df_all['latitude'] >= lat_min) & (df_all['latitude'] <= lat_max)   &
        np.isfinite(df_all['bathymetry'])
    ].copy()

    # 只保留属于测试集的点，确保剖面图反映泛化性能
    # Use original CSV row IDs: different observations may share a grid cell.
    df_region['track_index'] = df_region.groupby('survey_name').cumcount()
    in_test = df_region.index.isin(ship_rows[keep_idx[test_idx]])
    df_test = df_region.loc[in_test].copy()
    survey_stats = df_test.groupby('survey_name')['bathymetry'].agg(['count', 'var'])
    candidates = survey_stats[survey_stats['count'] >= 2].dropna(subset=['var'])
    if candidates.empty:
        print("No test survey has at least two valid points; skipping profile plots.")
        return
    best_survey = candidates['var'].idxmax()
    safe_survey = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(best_survey))
    prof = df_test[df_test['survey_name'] == best_survey].reset_index(drop=True)
    p_lats   = prof['latitude'].values.astype(np.float32)
    p_lons   = prof['longitude'].values.astype(np.float32)
    p_depths = prof['bathymetry'].values.astype(np.float32)
    x_axis   = prof['track_index'].to_numpy()

    p_lat_idx = nearest_indices(lat_grid, p_lats)
    p_lon_idx = nearest_indices(lon_grid, p_lons)
    p_pred    = predicted_depth[p_lat_idx, p_lon_idx]

    valid    = np.isfinite(p_pred) & np.isfinite(p_depths)
    if valid.sum() < 2:
        print("Fewer than two finite test profile predictions; skipping plots.")
        return
    x_v      = x_axis[valid]
    depths_v = p_depths[valid]
    pred_v   = p_pred[valid]
    abs_err  = np.abs(pred_v - depths_v)

    fig, ax = plt.subplots(figsize=(14, 5))
    profile_true = np.full(x_v[-1] - x_v[0] + 1, np.nan)
    profile_pred = np.full_like(profile_true, np.nan)
    profile_x = np.arange(x_v[0], x_v[-1] + 1)
    profile_true[x_v - x_v[0]] = depths_v
    profile_pred[x_v - x_v[0]] = pred_v
    ax.plot(profile_x, profile_true, color='steelblue', linewidth=1.5, label='Ship Measurement')
    ax.plot(profile_x, profile_pred,   color='tomato',    linewidth=1.5, label='Multi-Scale PatchCNN')
    ax.set_xlim(x_v[0], x_v[-1])
    # Negative bathymetry is elevation; keep the normal vertical axis.
    ax.set_xlabel('Along-Track Point Index', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Bathymetric Profile — {best_survey}  ({region_label})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/CNN_cdpa_multiscale_profile_{safe_survey}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"剖面图已保存: CNN_cdpa_multiscale_profile_{safe_survey}.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x_v, abs_err, alpha=0.4, color='tomato')
    ax.plot(x_v, abs_err, color='tomato', linewidth=1.2)
    ax.axhline(abs_err.mean(), color='k', linewidth=1.2, linestyle='--',
               label=f'Mean Error = {abs_err.mean():.1f} m')
    ax.set_xlim(x_v[0], x_v[-1])
    ax.set_xlabel('Along-Track Point Index', fontsize=12)
    ax.set_ylabel('Absolute Error (m)', fontsize=12)
    ax.set_title(f'Error Profile — Multi-Scale PatchCNN  |  {best_survey}  ({region_label})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/CNN_cdpa_multiscale_error_{safe_survey}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"误差剖面图已保存")


if __name__ == '__main__':
    main()
