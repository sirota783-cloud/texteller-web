#!/usr/bin/env bash
set -e
PORT="${PORT:-10000}"
RAY_memory_monitor_refresh_ms=0 exec texteller launch -p "$PORT" --use-onnx --ngpu-per-replica 0
