"""Data-free CPU demonstration; its scores are NOT paper results."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gravity_models import MultiScalePatchCNN
from traditional_gravity import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT/'build/synthetic_example')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    rng = np.random.default_rng(42)
    patches = rng.normal(size=(64, 2, 25, 25)).astype(np.float32)
    labels = (-3000+250*patches[:, 0].mean((1, 2))+100*patches[:, 1, 12, 12]).astype(np.float32)
    order = rng.permutation(len(labels))
    train, val, test = order[:44], order[44:54], order[54:]
    mean, std = float(labels[train].mean()), float(labels[train].std())
    x = torch.from_numpy(patches)
    y = torch.from_numpy((labels-mean)/std)
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=11, shuffle=True,
                        generator=torch.Generator().manual_seed(123))
    # Reduced width for a short CPU check; the paper network uses width 128.
    model = MultiScalePatchCNN(in_channels=2, feat_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    def scales(batch):
        return [batch[:, :, 12-h:13+h, 12-h:13+h] for h in (4, 8, 12)]
    best_loss, best_state, history = float('inf'), None, []
    for epoch in range(args.epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.huber_loss(model(scales(bx)), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.nn.functional.huber_loss(model(scales(x[val])), y[val]))
        history.append(dict(epoch=epoch+1, val_loss=score))
        if score < best_loss:
            best_loss = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(scales(x[test])).numpy()*std+mean
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(synthetic=True, device='cpu', training_seed=123, data_seed=42,
                  feature_dim=16, history=history, test=metrics(pred, labels[test]))
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
