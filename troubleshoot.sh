#!/usr/bin/env bash
# fix_env.sh  – Reinstall compatible Torch & urllib3 inside .venv39

set -euxo pipefail

VENV="/apps/logAnalyzer/.venv39"
PYBIN="$VENV/bin/python"
PIP="$VENV/bin/pip"

source "$VENV/bin/activate"

echo "Python:" $("${PYBIN}" -V)
echo "Binary:" "$PYBIN"

# Remove incompatible wheels
"${PIP}" uninstall -y torch urllib3 || true

# CPU‑only torch
"${PIP}" install --no-cache-dir \
  torch==2.1.2+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html

# urllib3 compatible with OpenSSL 1.0.2
"${PIP}" install --no-cache-dir 'urllib3<2,>=1.26.18'

# Lock versions
"${PIP}" freeze | grep -E '^(torch==|urllib3==)' > requirements.lock

# Verify
"${PYBIN}" - <<'PY'
import torch, urllib3, ssl, sys
print("torch     :", torch.__version__)
print("urllib3   :", urllib3.__version__)
print("OpenSSL   :", ssl.OPENSSL_VERSION)
print("Python    :", sys.version.split()[0])
PY

echo "Environment patch complete."
