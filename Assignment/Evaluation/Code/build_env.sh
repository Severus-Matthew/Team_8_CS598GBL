#!/bin/bash
set -e # this will stop the script on first error

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Building environment '$ENV_NAME'"


  # - biopython>=1.85
  # - biotite=0.36
  # - pandas>=2.2.3
  # - foldseek=8.ef4e960=pl5321hb365157_0
  # - matplotlib<3.10
  # - pymol-open-source  # PyMOL from conda-forge
  # - gputil
  # - deepspeed
  # - gcc_linux-64
  # - libgcc-ng
  # - cudatoolkit=11.8
  # - cuda-nvcc=11.8
  # - cudnn

mamba install -c bioconda -c conda-forge -c pytorch -c nvidia -c defaults biopython=1.85 biotite=0.36 pandas>=2.2.3 foldseek=8.ef4e960=pl5321hb365157_0 pymol-open-source gputil deepspeed gcc_linux-64 libgcc-ng cudatoolkit=11.8 cuda-nvcc=11.8 cudnn -y