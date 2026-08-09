#!/usr/bin/env bash
set -e
PORT="${PORT:-10000}"
exec texteller launch --host 0.0.0.0 --port "$PORT" --use-onnx
