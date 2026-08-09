#!/usr/bin/env bash
set -e
PORT="${PORT:-10000}"
exec texteller launch --port "$PORT" --use-onnx
