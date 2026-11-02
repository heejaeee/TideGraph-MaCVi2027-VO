"""Restore pinned, hash-verified private Hugging Face artifacts into this checkout.

Authenticate with `hf auth login` first. No credentials are embedded in this repo.
This restores model estimates, not a replacement for raw-video inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def destination(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe artifact path')
    out = (root / name).resolve()
    if not out.is_relative_to(root.resolve()):
        raise ValueError('Artifact destination escapes checkout')
    return out


def expected_paths(lock):
    result = {}
    for a in lock['assets']:
        if a['kind'] == 'zip':
            result.update(a['members_sha256'])
        else:
            result[a['destination']] = a['sha256']
    return result


def install(root, name, stream, checksum, force):
    out = destination(root, name)
    if out.exists():
        if digest(out) == checksum:
            return
        if not force:
            raise ValueError(f'Existing file has different hash: {name}; inspect or use --force')
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.download-', dir=out.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(stream, f)
        if digest(tmp) != checksum:
            raise ValueError(f'Artifact hash mismatch: {name}')
        os.replace(tmp, out)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--force', action='store_true', help='replace mismatching local artifact files')
    args = ap.parse_args()
    root = args.root.resolve()
    lock = json.loads((root / 'artifacts.lock.json').read_text())
    expected = expected_paths(lock)
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        revision = lock['revision']
        if not revision or len(revision) != 40:
            raise ValueError('Artifact revision must be a pinned commit SHA')
        for a in lock['assets']:
            local = hf_hub_download(repo_id=lock['repo_id'], repo_type=lock['repo_type'],
                                    filename=a['path'], revision=revision)
            if digest(local) != a['sha256']:
                raise ValueError(f'Download hash mismatch: {a["path"]}')
            if a['kind'] == 'zip':
                with zipfile.ZipFile(local) as z:
                    members = z.infolist()
                    names = [m.filename for m in members]
                    if len(names) != len(set(names)) or set(names) != set(a['members_sha256']):
                        raise ValueError('Archive member list differs from pinned manifest')
                    for m in members:
                        if m.is_dir() or ((m.external_attr >> 16) & 0o170000) == 0o120000:
                            raise ValueError('Unexpected directory or symlink in archive')
                        with z.open(m) as f:
                            install(root, m.filename, f, a['members_sha256'][m.filename], args.force)
            elif a['kind'] == 'file':
                with open(local, 'rb') as f:
                    install(root, a['destination'], f, a['sha256'], args.force)
            else:
                raise ValueError('Unknown artifact type')
    for name, checksum in expected.items():
        path = destination(root, name)
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f'Missing or changed artifact: {name}')
    print(json.dumps({'verified_files': len(expected), 'repo_id': lock['repo_id'],
                      'revision': lock['revision'], 'status': 'verified'}, indent=2))


if __name__ == '__main__':
    main()
