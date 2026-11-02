"""Download pinned upstream code/weights and write the raw runner asset map.

Setup may use the network. Inference workers never download anything.
The public companion bundle is used by default under its component licenses.
Upstream-only Omega download requires upstream access using your HF account.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--dpvo-weights', type=Path, help='Optional existing dpvo.pth; SHA-256 checked')
    ap.add_argument('--checkpoint-zip', type=Path, help='Verified TideGraph checkpoint bundle; missing models still come from upstream')
    ap.add_argument('--upstream-only', action='store_true', help='Do not use the public companion bundle; Omega upstream access may be required')
    a = ap.parse_args(); root = a.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = json.loads((PACKAGE/'configs/raw_assets.lock.json').read_text())
    if a.upstream_only and a.checkpoint_zip:
        ap.error('--upstream-only and --checkpoint-zip cannot be combined')
    if not a.checkpoint_zip and not a.upstream_only and not a.verify_only:
        from huggingface_hub import hf_hub_download
        bundle = json.loads((PACKAGE/'configs/checkpoint_bundle.json').read_text())
        print('Downloading public checkpoint bundle under its component-specific research terms', flush=True)
        a.checkpoint_zip = Path(hf_hub_download(bundle['repo_id'], bundle['filename'],
                                               revision=bundle['revision'], token=False))
        if a.checkpoint_zip.stat().st_size != bundle['bytes'] or sha256(a.checkpoint_zip) != bundle['sha256']:
            raise ValueError('Companion bundle hash/size mismatch')
    archive = zipfile.ZipFile(a.checkpoint_zip) if a.checkpoint_zip else None
    bundled = {
        'models/omega.pt': 'weights/vggt_omega_1b_416_reproduce.pt',
        'models/pi3x.safetensors': 'legacy_upstream/pi3x/model.safetensors',
        'repos/DPVO/dpvo.pth': 'legacy_upstream/DPVO/dpvo.pth',
        'models/flow_speed_v3/model.pkl': 'trained_models/flow_speed_v3/model.pkl',
    }
    if archive:
        for entry in archive.infolist():
            p = Path(entry.filename)
            if p.parts[:2] == ('TideGraph_checkpoints', 'licenses') and len(p.parts) == 3:
                if entry.file_size > 1024*1024 or p.name in ('.', '..'):
                    raise ValueError('Invalid license attachment')
                dest = root/'licenses'/p.name; dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(entry))
    def git(path, *args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    for name, rec in lock['repositories'].items():
        dest = root/'repos'/name
        if not (dest/'.git').exists():
            if a.verify_only: raise FileNotFoundError(dest)
            subprocess.run(['git', 'clone', '--no-checkout', rec['url'], str(dest)], check=True)
            subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', rec['commit']], check=True)
            subprocess.run(['git', '-C', str(dest), 'submodule', 'update', '--init', '--recursive'], check=True)
        if git(dest, 'rev-parse', 'HEAD') != rec['commit']:
            raise ValueError('Wrong source revision: '+name)
    for rec in lock['files']:
        dest = root/rec['path']
        if not dest.exists():
            if a.verify_only: raise FileNotFoundError(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name(dest.name+'.partial')
            if archive and rec['path'] in bundled:
                member = 'TideGraph_checkpoints/'+bundled[rec['path']]
                if archive.getinfo(member).file_size != rec['bytes']:
                    raise ValueError('Wrong bundled checkpoint size: '+member)
                with archive.open(member) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            elif rec.get('repo'):
                from huggingface_hub import hf_hub_download
                source = hf_hub_download(rec['repo'], rec['filename'], revision=rec['revision'])
                shutil.copyfile(source, temporary)
            elif rec.get('url'):
                with urllib.request.urlopen(rec['url']) as response, temporary.open('wb') as f:
                    shutil.copyfileobj(response, f)
            elif rec['path'] == 'repos/DPVO/dpvo.pth':
                if a.dpvo_weights:
                    shutil.copyfile(a.dpvo_weights, temporary)
                else:
                    import gdown
                    dpvo_archive = root/'dpvo_models.zip'
                    gdown.download(id=lock['dpvo_drive_id'], output=str(dpvo_archive), quiet=False)
                    with zipfile.ZipFile(dpvo_archive) as z:
                        members = [n for n in z.namelist() if Path(n).name == 'dpvo.pth']
                        if len(members) != 1: raise ValueError('Ambiguous DPVO checkpoint archive')
                        with z.open(members[0]) as src, temporary.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
            else: raise ValueError('No source for '+rec['path'])
            if sha256(temporary) != rec['sha256']:
                raise ValueError('Downloaded checkpoint hash mismatch: '+rec['path'])
            temporary.replace(dest)
        if dest.stat().st_size != rec['bytes'] or sha256(dest) != rec['sha256']:
            raise ValueError('Checkpoint hash/size mismatch: '+rec['path'])
        print('VERIFIED', rec['path'], flush=True)
    # Preserve the exact source fixes used in the original inference environment.
    for name, rel in lock['patches'].items():
        repo = root/'repos'/name; patch = PACKAGE/rel
        applied = subprocess.run(['git', '-C', str(repo), 'apply', '--reverse', '--check', str(patch)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not applied:
            if a.verify_only: raise ValueError('Required source patch missing: '+name)
            subprocess.run(['git', '-C', str(repo), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(repo), 'apply', str(patch)], check=True)
    training = root/'models/flow_speed_v3/training_record.json'
    training.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PACKAGE/'configs/flow_speed_training_record.json', training)
    mapped = {r['path']: str(root/r['path']) for r in lock['files']
              if not r['path'].startswith(('repos/', 'models/mask2former/'))}
    mapped['models/flow_speed_v3/training_record.json'] = str(training)
    assets = dict(repositories={n: str(root/'repos'/n) for n in lock['repositories']},
                  files=mapped, segmentation_model=str(root/'models/mask2former'))
    path = root/'assets.json'; path.write_text(json.dumps(assets, indent=2)+'\n')
    if archive:
        archive.close()
    print('Assets ready:', path)
    print('Compile DPVO/lietorch in your CUDA environment before raw inference. See RAW_REPRODUCTION.md.')


if __name__ == '__main__':
    main()
