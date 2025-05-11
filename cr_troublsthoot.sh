#!/bin/bash
set -e

# 0) Activate the exact venv your service uses
source /apps/logAnalyzer/.venv39/bin/activate

echo "Python version:"
python -V        # should say 3.9.x
echo "Python binary:"
which python     # should be /apps/logAnalyzer/.venv39/bin/python

# 1) Remove the bad wheels
echo "Uninstalling torch and urllib3..."
pip uninstall -y torch urllib3

# 2) Install CPU‑only Torch that has no CUDA / ABI headaches
echo "Installing CPU-only torch..."
pip install --no-cache-dir torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 3) Downgrade urllib3 to last 1.x line
echo "Installing compatible urllib3..."
pip install --no-cache-dir 'urllib3<2,>=1.26.18'

# 4) OPTIONAL: freeze versions so the pipeline never upgrades them again
echo "Freezing torch and urllib3 versions in requirements.txt..."
pip freeze | grep -E 'torch|urllib3' >> requirements.txt

# 5) Confirm installation
echo "Installed versions:"
python -c "import torch; print('torch:', torch.__version__); import urllib3; print('urllib3:', urllib3.__version__)"

echo "Environment fix complete."
