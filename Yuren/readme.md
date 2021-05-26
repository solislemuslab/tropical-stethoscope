## Files description

### Jupyter Notebooks:

dataset_creation.ipynb: the notebook to create the dataset in hdf5 format

classification_with_augmentation.ipynb: the notebook for classifiaction model with augmentation functions.

figures.ipynb: the notebook to plot figures

### Python scripts:

Those are the files used for experiments.

classification_with_augmentation.py: the file that run on the condor for the experiment to randomly select sonotypes without fixing sizes, augment, trian, and check classification performance.

classification_fixed_size.py: the file that run on the condor for the experiment to randomly select sonotypes, use same (fixed) size for all the sonotypes in one run, augment, trian, and check classification performance.

experiment_aug_no_aug.py: the file that run on the condor for the experiment to select sonotypes with fixed sample size, train with and without augmentation, and check classification performance.

### script_submit folder

Inside are files that are used to submit jobs to the condor to run the python scripts.

## Notes

Data are not pushed to this repo

The files classification_with_augmentation.py, classification_fixed_size.py, experiment_aug_no_aug.py and classification_with_augmentation.ipynb are similar with slight differences for different experiments.

