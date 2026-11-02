#!/usr/bin/env bash
set -euo pipefail
asset_root="$(realpath "${1:?Usage: build_raw_extensions.sh /path/to/raw-assets}")"
dpvo_source="$asset_root/repos/DPVO"
: "${CUDA_HOME:?Set CUDA_HOME to your CUDA 12.8 toolkit}"
export MAX_JOBS="${MAX_JOBS:-4}"
# RTX 5080 uses 12.0. Override for a different GPU before compiling.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
python -m pip install 'wheel==0.45.1'
python -m pip install 'torch-scatter==2.1.2+pt27cu128' -f 'https://data.pyg.org/whl/torch-2.7.0+cu128.html'
if [[ ! -f "$dpvo_source/thirdparty/eigen-3.4.0/Eigen/Core" ]]; then
  python - "$dpvo_source" <<'PY'
import io,sys,urllib.request,zipfile
from pathlib import Path
root=Path(sys.argv[1])/'thirdparty'
with urllib.request.urlopen('https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip') as r:
    archive=zipfile.ZipFile(io.BytesIO(r.read()))
    for info in archive.infolist():
        p=Path(info.filename)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='eigen-3.4.0':
            raise ValueError('Unexpected Eigen archive member')
    archive.extractall(root)
PY
fi
python -m pip install --no-build-isolation --no-deps "$dpvo_source"
python -c 'import torch; import cuda_corr, cuda_ba, lietorch_backends; print("DPVO CUDA extensions imported")'
