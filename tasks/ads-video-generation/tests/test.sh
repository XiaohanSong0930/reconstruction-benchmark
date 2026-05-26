#!/usr/bin/env bash
set -euo pipefail

cd /tests
python -m pytest -q test_outputs.py
