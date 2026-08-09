#!/usr/bin/env bash
set -e
PORT="${PORT:-10000}"
exec texteller web --server.address 0.0.0.0 --server.port "$PORT" --server.headless true
