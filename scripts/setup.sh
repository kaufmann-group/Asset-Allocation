#!/bin/bash
set -e

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

arch -arm64 python3 -m venv "$REPO/ocean"
source "$REPO/ocean/bin/activate"

python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir --force-reinstall -r "$REPO/scripts/requirements.txt"

python3 -m ipykernel install --user --name=repo-env --display-name "Asset Allocation Repository Environment"


echo ""
echo "Setup complete."
echo "Activate the environment from the project root with:"
echo "source ocean/bin/activate"