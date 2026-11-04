"""Build a curated journal ZIP from the versioned release allowlist."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = json.loads((ROOT/'release_manifest.json').read_text(encoding='utf-8'))['files']
    if len(files) != len(set(files)):
        raise RuntimeError('Duplicate release paths')
    files = sorted(files)
    if not files or 'README.md' not in files:
        raise RuntimeError('Missing source files')
    for name in files:
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError(f'Missing or unsafe release path: {name}')
        if (name.startswith('data/') and not name.endswith('README.md')) or Path(name).suffix in ('.log', '.pth', '.npz', '.npy'):
            raise RuntimeError(f'Unexpected release artifact: {name}')
    output = ROOT/'dist'
    output.mkdir(exist_ok=True)
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    with zipfile.ZipFile(output/'CDPA-journal-source-clean.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            info = zipfile.ZipInfo('CDPA/'+name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT/name).read_bytes())
        archive.writestr('CDPA/SOURCE_HASHES.json', json.dumps(hashes, indent=2))
    (output/'journal_clean_manifest.txt').write_text('\n'.join(files)+'\n', encoding='utf-8')
    print(f'Archived {len(files)} files: {output/"CDPA-journal-source-clean.zip"}')


if __name__ == '__main__':
    main()
