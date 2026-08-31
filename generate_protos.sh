#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
protoc --proto_path=proto --python_out=. \
  proto/womd_protos/scenario.proto \
  proto/womd_protos/map.proto
touch womd_protos/__init__.py
echo "protos generated"
