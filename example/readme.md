# Running the neural network model on a sample dataset or with your own data

We provide a sample dataset and a jupyter notebook to exemplify the steps to run the neural network model. The sample dataset could be found [here](https://drive.google.com/file/d/101Mnahr0ZPVz1eFyBniNPNJ0NltlVhk6/view?usp=sharing) on Google Drive and is about 480mb in size. The sample dataset contains data on 6 birds (the classes that we want to classify with the neural network model) and noise data that are used to augment the dataset.

If you want to try out your own data to train the model, you need to make sure to put your data in the same format which is a HDF5 format to run the sample classification. 

To do so, you need to follow the following instructions:

1. Make sure that your labe file have the same (or at least similar) format as `sample_labels.txt`. You also need corresponding wav files for recordings.

2. You will run the `sample_dataset_creation.ipynb` jupyter notebook file maybe with some slight changes based on your data format, to create the labels. Details about this process could be found in the `Data structure` section below. To run this notebook file, you can follow the instructions in the `Instructions to create dataset and fit the neural network model` section.

Once you have your data in the hdf5 format as described above, you can follow the instructions in `Instructions to create dataset and fit the neural network model` section to run the `sample_classification_with_augmentation.ipynb` jupyter notebook for the training process. The training speed might differ in the laptop processers but should probably take less than 2 hours.


## Data structure

We provide an example jupyter notebook with our approach to create the create the hdf5 dataset, named `sample_dataset_creation.ipynb`. The file will read and parse the `sample_labels.txt` file to retrieve info from to plot and chop the spectrogram to create the dataset. The labels file was created using Raven Pro 1.6 (please see methods in the publication). The wav file used in this example could be found [here](https://drive.google.com/file/d/1b0KzSFkvSakbIoQhLk9VHDX8Wi17d2xk/view?usp=sharing). 
Note that the label and wav data in this example is different from the sample dataset provided above as the sample dataset is created with multiple label files and wav files, which are too large to include in the example.

The end result for `sample_dataset_creation.ipynb` is a sample dataset stored in hdf5 format consisting of 5 arrays (lists). Each element in each array are described below.
- image (spectrogram) of the sonotype with size 244 * 244 * 3 ( 3 channel 224 * 224 pixels image) as float. 
- sonotype number as float
- time range: 2 floats for start and end time
- frequency range: 2 floats for minimum and maximum frequency
- taxonomic group as string (in this case, we only provide sonotypes of 6 birds so all the groups are "b"). This is not used in training/validation/testing but only for analysis about the model performance on different taxonomic group so it is not necessary to have it in order to run the example.

The element at the same index in each array belongs to the same sample and the five arrays should have the same length, i.e, the nth image in image array, nth time range, nth frequency, and nth taxomic group represent the same sample of nth sonotype number.


## Instructions to create dataset and fit the neural network model

You need to follow the steps (mainly aimed at Mac or Linux users):

1. Open the terminal and move to a path where you want to store the scripts and data.

2. Type `git clone https://github.com/solislemuslab/tropical-stethoscope.git` which will download the entire GitHub repository.

3. Download `JupyterLab`. Instructions could be found [here](https://jupyter.org/), but one way to do it is by typing: `conda install -c conda-forge jupyterlab`.

4. Download the necessary packages by typing the following in the terminal: 
```  
pip3 install -r requirements.txt
```
The `requirements.txt` is inside the `example` folder that was cloned and it lists the requirements below:
```
numpy
opencv
h5py
pandas
tensorflow-gpu==1.14.0
tensorflow==1.14.0
cudatoolkit=10.0
keras==2.3.1
```
If you want to run the `sample_dataset_creation.ipynb` to create the dataset, include the following lines in the `requirement.txt`:
```
scipy
soundfile
```

5. Run `JupyterLab` by typing `jupyter-lab` in the terminal. A browser page will be opened automatically where you will find the `sample_classification_with_augmentation.ipynb` and "sample_dataset_creation.ipynb" jupyter notebook from the folder you just cloned.

6. Change the path in the 3rd code block (Read the dataset) in `sample_classification_with_augmentation.ipynb` to where you stored your hdf5 file and run all the code blocks. If you want to run `sample_dataset_creation.ipynb`, change the path in the top three lines of 3rd code block (Parse the label file) to where you stored your label file, the name of your label file, and where you stored your wav file respectively. Also, change the path in the 4th code block (Creat H5 data storage) to where you want to store the hdf5 dataset.


Note that one alternative to use `JupyterLab` is to use Google Colab, which is a online version for jupyter notebook. Instructions could be found [here](https://colab.research.google.com/). When using Colab to fit the model with `sample_classification_with_augmentation.ipynb`, please remember to use the GPU run time when fitting the model, which could be found at Runtime > Change runtime type and select GPU as Hardware accelerator (otherwise, the training process might be extremely slow). Google Colab limit the resources that we can use for free within a period of time, including the GPU, but the limited resources is enough for this example. You do not need to do this when running `sample_dataset_creation.ipynb` as CPU runtime is enough for dataset creation.




