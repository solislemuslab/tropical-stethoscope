The files in this folder provide similar functions as those in the `example` folder.  The files here are used to run similar jobs on HTCondor repeatedly for the data analyses presented in the manuscript. If you are interested to try out our methods or use our neural network model in your own data, please follow the steps in the `example` folder.

## Files description

### Jupyter Notebooks:

`dataset_creation.ipynb`: the notebook to create the dataset in hdf5 format.

`classification_with_augmentation.ipynb`: the notebook for classifiaction model with augmentation functions.

`figures.ipynb`: the notebook to plot figures.

### Python scripts:

Those are the files used for experiments.

`classification_fix_6_sono.py`: the file that run on the condor for the experiment to randomly select sonotypes, use same (fixed) or original size (specified with parameter in main) for all the sonotypes in one run, augment, trian, and check classification performance. Note that we only compute auc/roc and accuracy in the script but we record necessary data for all the other methods for performance evaluation.

`experiment_2_to_6.py`: the file that run on the condor for the experiment to select sonotypes with fixed sample size (49), train with and without augmentation, and check classification performance.

### script_submit folder

Inside are files that are used to submit jobs to the HTCondor to run the python scripts. The  `sub`  files are used to submit the jobs to HTCondor to run the scripts. The `sh` files can be used to set up the environment and run the python scripts with requirements listed in `environment.yml`.

## Notes

Data are not pushed to this repo.

The files`classification_fix_6_sono.py`, `experiment_2_to_6.py` and `classification_with_augmentation.ipynb` are similar with slight differences for different experiments.

