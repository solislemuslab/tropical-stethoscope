#!/bin/bash

# mini conda
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda-3.6.0-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH
conda install conda

conda env create -f environment.yml
source activate tf-gpu

# run your script
# for Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
export TF_FORCE_GPU_ALLOW_GROWTH='true'
cp /staging/ysun299/whole_data_1110.hdf5 ./
python3 classification_fixed_size.py
rm whole_data_1110.hdf5
rm model_group.hdf5
