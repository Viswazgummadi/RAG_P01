#!/bin/bash
#SBATCH --job-name=LLM-finetune                 # Job name
#SBATCH --output=TEST_01.txt                # Output file
#SBATCH --ntasks=1                           # Run a single task
#SBATCH --time=0:10:00                       # Increase time limit
#SBATCH --partition=gpu                 # Specify correct GPU partition
#SBATCH --gres=gpu:1                         # Request one GPU
#SBATCH --nodelist=node2                     # Run on node2
#SBATCH --mem=2G
#SBATCH --account=iitdh_acc1     # Explicit account declaration


echo "================ SLURM JOB INFO ================"
echo "SLURM_JOBID = $SLURM_JOBID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_NNODES = $SLURM_NNODES"
echo "SLURMTMPDIR = $SLURMTMPDIR"
echo "Date = $(date)"
echo "Hostname = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "================================================"

# Create custom temporary directories
echo "Setting up temporary directories..."
export RUNTIME_DIR="$HOME/runtime-$SLURM_JOBID"
export TEMP_DIR="$HOME/slurm_tmp_$SLURM_JOBID"
mkdir -p $RUNTIME_DIR $TEMP_DIR
chmod 700 $RUNTIME_DIR $TEMP_DIR

# Configure temporary directory variables
export XDG_RUNTIME_DIR=$RUNTIME_DIR
export TMPDIR=$TEMP_DIR
export TMP=$TEMP_DIR
export TEMP=$TEMP_DIR

# Initialize Miniconda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate rag_env

echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"

# Install CPU-only PyTorch if CUDA is problematic
echo "Installing CPU-only PyTorch..."
pip install torch torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers
echo "Installing sentence-transformers..."
pip install sentence-transformers

# Run vector_db.py
echo "Running vector_db.py..."
python vector_db.py

# Clean up temporary directories
rm -rf $RUNTIME_DIR $TEMP_DIR
