## Files description

### Jupyter Notebooks:

dataset_creation.ipynb: the notebook for create the dataset in hdf5 format

classification.ipynb: the notebook for the classification model without augmentation functions.

classification_with_augmentation.ipynb: the notebook for classifiaction model with augmentation functions.

### Python files:

classification_with_augmentation.py: the file that run on the condor for the experiment to randomly select sonotypes, augment, trian, and check classification performance.

classification_fixed_size.py: the file that run on the condor for the experiment to randomly select sonotypes, use same size for all the sonotypes in one run, augment, trian, and check classification performance.

experiment_aug_no_aug.py: the file that run on the condor for the experiment to select sonotypes with fixed number, train with and without augmentation, and check classification performance.

### script_submit folder

Inside are files that are used to submit jobs to the condor to run the python files.

## Notes

Data are not pushed to this repo

