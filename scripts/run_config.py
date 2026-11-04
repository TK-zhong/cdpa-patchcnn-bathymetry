"""Execute a versioned JSON configuration using the current Python interpreter."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'CDPA_split.py', 'run_architecture_ablation.py', 'run_ablation.py', 'run_traditional.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['entrypoint'] not in ALLOWED or not isinstance(config['arguments'], list):
        parser.error('Invalid entrypoint or arguments')
    if not all(isinstance(value, str) for value in config['arguments']):
        parser.error('Arguments must be strings')
    command = [sys.executable, '-u', str(ROOT/config['entrypoint']), *config['arguments']]
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
