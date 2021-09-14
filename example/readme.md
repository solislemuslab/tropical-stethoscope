# Fitting the neural network model on a sample dataset

We provide a sample dataset to fit the neural network model on Google Drive [here](https://drive.google.com/file/d/101Mnahr0ZPVz1eFyBniNPNJ0NltlVhk6/view?usp=sharing). The sample dataset contains data on 6 sonotypes of birds (the classes that we want to classify with the neural network model) and noise data that are used to augment the dataset of bird sonotypes.

The sample dataset is stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and is about 480mb in size format. It consists of 5 arrays (lists). The elements in each array are described below.
- Image (spectrogram) of the sonotype with size 244 * 244 * 3 ( 3 channel 224 * 224 pixels image) as float. 
- Sonotype number as float
- Time range: 2 floats for start and end time
- Frequency range: 2 floats for minimum and maximum frequency
- Taxonomic group as string (in this case, we only provide 6 sonotypes of birds so all the groups are "b"). It is not necessary to have this array to run the example. The taxonomic group array is not used for training/validation/testing but only for evaluating model performance on different taxonomic groups.

The elements of the same index in each array belong to the same sample. The five arrays should have the same length, i.e, the nth image in image array, nth time range, nth frequency, and nth taxonomic group represent the same sample of nth sonotype number.

For a more detailed description of the raw data used to create the HDF5 file, see the section below on "Fitting the neural network model on your own data".


## Instructions to fit the neural network model on the sample data

You need to follow the steps (mainly aimed at Mac or Linux users):

1. Open the terminal and move to a path where you want to store the scripts and data.

2. Type `git clone https://github.com/solislemuslab/tropical-stethoscope.git` to download the entire GitHub repository.

3. Download `JupyterLab` by typing: `conda install -c conda-forge jupyterlab`. More instructions can be found [here](https://jupyter.org/).

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
scipy
soundfile
scikit-learn
```

5. Run `JupyterLab` by typing `jupyter-lab` in the terminal. A browser page will be opened automatically where you will find the `sample_classification_with_augmentation.ipynb` jupyter notebook from the folder you have just cloned.

6. Change the path in the 3rd code block (Read the dataset) in `sample_classification_with_augmentation.ipynb` to where you stored the sample HDF5 file and run all the code blocks. The training speed might differ in the laptop processors but it should take less than 2 hours.

Note that one alternative to use `JupyterLab` is to use Google Colab, an online version for jupyter notebook. Instructions can be found [here](https://colab.research.google.com/). When using Colab to fit the model with `sample_classification_with_augmentation.ipynb`, please remember to use the GPU run time when fitting the model, which could be found at Runtime > Change runtime type and select GPU as Hardware accelerator (otherwise, the training process might be extremely slow). Google Colab limits the resources that we can use for free within a period of time, including the GPU, but the limited resources is enough for this example. You do not need to do this when running `sample_dataset_creation.ipynb` as CPU runtime is enough for dataset creation.

# Fitting the neural network model on your own data

If you want to try out your own data to train the model, make sure to put your data in HDF5 format to run the sample classification. 

You need to have two types of files:
1. A label file with the sonotype classification for each sound file. This file needs to have the same format as `sample_labels.txt`.
2. Sound files in wav format all stored in a folder

We provide an example jupyter notebook with our approach to create the hdf5 dataset: `sample_dataset_creation.ipynb`. The file will read and parse the `sample_labels.txt` file to retrieve info from to plot and chop the spectrogram to create the dataset. The labels file was created using Raven Pro 1.6 (please see Methods in the publication). The wav file used in this example can be found [here](https://drive.google.com/file/d/1b0KzSFkvSakbIoQhLk9VHDX8Wi17d2xk/view?usp=sharing) (around 150mb). The expected output of the jupyter notebook from the label file and wav file can be found [here](https://drive.google.com/file/d/1IiNqZQEcxwT8BECapfM7bJS4Y9zmJUaw/view?usp=sharing).
Note that the label and sound files in this example are different from the sample dataset provided above as the sample dataset was created with multiple label files and sound files, which are too large to include in the current example.

## Creating the HDF5 input file

You need to follow the steps (mainly aimed at Mac or Linux users):

1. Open the terminal and move to a path where you want to store the scripts and data.
2. Type `git clone https://github.com/solislemuslab/tropical-stethoscope.git` to download the entire GitHub repository.
3. Download `JupyterLab` by typing: `conda install -c conda-forge jupyterlab`. More instructions can be found [here](https://jupyter.org/).
4. Download the necessary packages by typing the following in the terminal: 
```  
pip3 install -r requirements.txt
```
Note that the `requirements.txt` is inside the `example` folder that was cloned. It lists the requirements to create HDF5 input file and fit the neural network model. For this step, only the packages below are required, but we can install all the required packaged, which will be needed to fit the model later, with the above commend.
```
numpy
scipy
soundfile
h5py
opencv
```

5. Run `JupyterLab` by typing `jupyter-lab` in the terminal. A browser page will be opened automatically where you will find the `sample_dataset_creation.ipynb` jupyter notebook from the folder you have just cloned.

6. Change the path in the top three lines of 3rd code block (Parse the label file) in the notebook `sample_dataset_creation.ipynb` to where you stored your label file, the name of your label file, and the name of the path where you stored your wav file respectively. Also, change the path in the 4th code block (Create H5 data storage) to where you want to store the hdf5 dataset.
7. Once you have your data in the hdf5 format, you can open and follow the `sample_classification_with_augmentation.ipynb` jupyter notebook for the training process. Training time will depend on the size of your data.

