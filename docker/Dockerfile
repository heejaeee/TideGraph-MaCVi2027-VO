# First restore Hugging Face artifacts on the host: python scripts/fetch_artifacts.py
# CPU image for modes replay / rebuild (no GPU, no weights, no network at run time).
#   docker build -t tidegraph-cpu -f docker/Dockerfile .
#   docker run --rm --network none -v /path/to/lookout-vo-test-v0.1:/in:ro -v /tmp/out:/out tidegraph-cpu \
#       python repro_check.py /pkg --run --input /in --output /out
FROM python:3.10-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /pkg
WORKDIR /pkg
