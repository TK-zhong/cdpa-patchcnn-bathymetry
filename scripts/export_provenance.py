"""Export unmodified saved metrics with relative paths and source hashes."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result_root = ROOT/'data/result'
    paths = []
    for name in ('architecture_ablation', 'traditional_methods', '\u6d88\u878d'):
        directory = result_root/name
        if directory.exists():
            paths.extend(directory.rglob('metrics.json'))
    records, runs = [], []
    for path in sorted(paths):
        saved = json.loads(path.read_text(encoding='utf-8'))
        if 'metrics' not in saved:
            continue
        relative = path.relative_to(ROOT).as_posix()
        run_id = path.parent.relative_to(result_root).as_posix()
        summary_path = path.parent/'training_summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
        manifest_path = path.parent/'experiment.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        runs.append(dict(run_id=run_id, source_metrics=relative,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         experiment=saved.get('experiment', saved.get('method', '')),
                         region=saved.get('region', manifest.get('region', '')),
                         training_seed=manifest.get('seed', ''), split_seed=saved.get('split_seed', manifest.get('split_seed', '')),
                         best_epoch=summary.get('best_epoch', ''), status='historical_run_not_final_paper_assignment'))
        for subset, values in saved['metrics'].items():
            records.append(dict(run_id=run_id, subset=subset.lower(), **values))
    destination = ROOT/'results'
    (destination/'historical_metrics').mkdir(parents=True, exist_ok=True)
    for path, rows in ((destination/'experiment_manifest.csv', runs),
                       (destination/'historical_metrics/run_metrics.csv', records)):
        if not rows:
            raise RuntimeError('No saved runs found; nothing exported')
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f'Exported {len(runs)} runs, {len(records)} metric rows; values unchanged.')


if __name__ == '__main__':
    main()
