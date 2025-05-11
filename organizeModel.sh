#!/bin/bash
set -e

echo "Current working directory: $(pwd)"

if [ -f all-mpnet-base.zip ]; then
    echo "Found model zip: all-mpnet-base.zip"
else
    echo "ERROR: all-mpnet-base.zip not found in $(pwd)"
    exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p breakfix-kb-model/all-mpnet-base-v2

# Unzip the model into the target directory
unzip -o all-mpnet-base.zip -d breakfix-kb-model/all-mpnet-base-v2

# If the zip contains a subfolder, move files up one level (optional cleanup)
if [ -d breakfix-kb-model/all-mpnet-base-v2/all-mpnet-base-v2 ]; then
    mv breakfix-kb-model/all-mpnet-base-v2/all-mpnet-base-v2/* breakfix-kb-model/all-mpnet-base-v2/
    rmdir breakfix-kb-model/all-mpnet-base-v2/all-mpnet-base-v2
fi

echo "Model extracted to breakfix-kb-model/all-mpnet-base-v2/"
