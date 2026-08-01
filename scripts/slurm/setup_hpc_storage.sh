#!/bin/bash
# Helper script to prepare HPC Scratch storage (/scr/$USER) and symlink output directories.
# Run this once on the login node or automatically before starting batch jobs.

set -e

USER_NAME=${USER:-$(whoami)}

if [ -d "/scr" ]; then
    SCRATCH_BASE="/scr/$USER_NAME"
    mkdir -p "$SCRATCH_BASE" 2>/dev/null || true
elif [ -d "/scratch" ]; then
    SCRATCH_BASE="/scratch/$USER_NAME"
    mkdir -p "$SCRATCH_BASE" 2>/dev/null || true
else
    SCRATCH_BASE="/scr/$USER_NAME"
fi

echo "=================================================="
echo "HPC Scratch Storage Environment Setup"
echo "Target User Scratch Directory: $SCRATCH_BASE"
echo "=================================================="

if [ ! -d "$SCRATCH_BASE" ]; then
    echo "[Info] Scratch directory $SCRATCH_BASE not found or not mounted locally."
    echo "[Info] Proceeding with standard relative outputs and artifacts directories."
    exit 0
fi

# Create subdirectories under scratch
mkdir -p "$SCRATCH_BASE/m5_outputs"
mkdir -p "$SCRATCH_BASE/m5_artifacts"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/slurm


# Symlink outputs directory if not already linked
if [ ! -L "outputs" ]; then
    if [ -d "outputs" ] && [ "$(ls -A outputs 2>/dev/null)" ]; then
        echo "[Copying] Moving existing outputs to $SCRATCH_BASE/m5_outputs/..."
        cp -r outputs/* "$SCRATCH_BASE/m5_outputs/"
        rm -rf outputs
    else
        rm -rf outputs
    fi
    ln -s "$SCRATCH_BASE/m5_outputs" outputs
    echo "[Linked] Symlinked ./outputs -> $SCRATCH_BASE/m5_outputs"
else
    echo "[OK] ./outputs is already a symlink."
fi

# Symlink artifacts directory if not already linked
if [ ! -L "artifacts" ]; then
    if [ -d "artifacts" ] && [ "$(ls -A artifacts 2>/dev/null)" ]; then
        echo "[Copying] Moving existing artifacts to $SCRATCH_BASE/m5_artifacts/..."
        cp -r artifacts/* "$SCRATCH_BASE/m5_artifacts/"
        rm -rf artifacts
    else
        rm -rf artifacts
    fi
    ln -s "$SCRATCH_BASE/m5_artifacts" artifacts
    echo "[Linked] Symlinked ./artifacts -> $SCRATCH_BASE/m5_artifacts"
else
    echo "[OK] ./artifacts is already a symlink."
fi

echo "=================================================="
echo "HPC Storage setup completed successfully!"
echo "Outputs   : $(readlink -f outputs)"
echo "Artifacts : $(readlink -f artifacts)"
echo "=================================================="
