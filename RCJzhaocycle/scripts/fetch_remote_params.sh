#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-rcj@10.32.0.172}"
REMOTE_DIR="${REMOTE_DIR:-/home/rcj/RCJzhaocycle}"
LOCAL_DIR="${LOCAL_DIR:-remote_params}"

mkdir -p "${LOCAL_DIR}"
rsync -av "${HOST}:${REMOTE_DIR}/params/" "${LOCAL_DIR}/"
echo "fetched parameter files into ${LOCAL_DIR}/"
