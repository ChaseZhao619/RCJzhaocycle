#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-rcj@10.32.0.172}"
REMOTE_DIR="${REMOTE_DIR:-/home/rcj/RCJzhaocycle}"
PORT="${PORT:-8080}"
CAMERA="${CAMERA:-0}"

rsync -av --delete \
  --exclude .git \
  --exclude build \
  --exclude out \
  --exclude params \
  --exclude remote_params \
  --exclude '*.log' \
  --exclude '*.pid' \
  --exclude __pycache__ \
  ./ "${HOST}:${REMOTE_DIR}/"

ssh "${HOST}" "cd '${REMOTE_DIR}' && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"

ssh "${HOST}" "pkill -x arc_web_tuner 2>/dev/null || true"
ssh "${HOST}" "cd '${REMOTE_DIR}' && { nohup env LD_PRELOAD=/usr/local/libexec/libcamera/v4l2-compat.so ./build/arc_web_tuner --camera '${CAMERA}' --bind 0.0.0.0 --port '${PORT}' --params-dir params > arc_web_tuner.log 2>&1 < /dev/null & echo \$! > arc_web_tuner.pid; }"

host_ip="${HOST#*@}"
echo "arc_web_tuner started on ${HOST}"
echo "open: http://${host_ip}:${PORT}/"
echo "logs: ssh ${HOST} 'tail -f ${REMOTE_DIR}/arc_web_tuner.log'"
