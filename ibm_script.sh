set -euxo pipefail

# --- paths ------------------------------------------------------------
VENV="/apps/logAnalyzer/.venv39"
PYBIN="$VENV/bin/python"
PIP="$VENV/bin/pip"

# --- activate the virtual‑env ----------------------------------------
. "$VENV/bin/activate"

echo "Python:" "$($PYBIN -V)"
echo "Binary:" "$PYBIN"

# --- internal mirror (uncomment if required) -------------------------
# export PIP_INDEX_URL="https://artifactory-prxy-b.company.net/artifactory/api/pypi/pypi-python-remote/simple"
# export PIP_TRUSTED_HOST="artifactory-prxy-b.company.net"

# --- ensure latest pip / setuptools ----------------------------------
"$PIP" install --upgrade pip setuptools

# --- remove incompatible wheels --------------------------------------
"$PIP" uninstall -y torch urllib3 || true

# --- install CPU‑only torch (no CUDA / ABI mismatch) -----------------
"$PIP" install --no-cache-dir \
  torch==2.1.2+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html

# --- install urllib3 that works with OpenSSL 1.0.2 -------------------
"$PIP" install --no-cache-dir 'urllib3<2,>=1.26.18'

# --- lock the two versions for reproducibility -----------------------
"$PIP" freeze | grep -E '^(torch==|urllib3==)' > requirements.lock

# --- quick sanity check ----------------------------------------------
"$PYBIN" - <<'PY'
import torch, urllib3, ssl, sys
print("torch     :", torch.__version__)
print("urllib3   :", urllib3.__version__)
print("OpenSSL   :", ssl.OPENSSL_VERSION)
print("Python    :", sys.version.split()[0])
PY

echo "Environment patch complete."
