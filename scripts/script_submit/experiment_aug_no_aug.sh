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

# run the script
cp /staging/ysun299/whole_data_1110.hdf5 ./
python3 experiment_aug_no_aug.py
rm whole_data_1110.hdf5
rm model_group.hdf5
