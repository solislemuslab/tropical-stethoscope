# Running the neural network model on a sample dataset

## Data structure

We provide a sample dataset to exemplify the steps to run the neural network model. The sample dataset could be found [here](https://drive.google.com/file/d/1kEW1FscJROzEVm3R8_1Yl4RyOw9K5_jL/view?usp=sharing) on Google Drive and is about 480mb in size.
The sample dataset contains data on 6 birds (the classes that we want to classify with the neural network model) and noise data.

[need to add here what was the structure of the files before they were converted to hdf5. We might want to add one example of how the data looks like before the hdf5 in the same shared drive.]

You need to make sure to put your data in the same format which is a HDF5 format.

To do so, you need to follow the following instructions:

1. Make sure that your datafiles have this format [need details here, but I don't know thw raw format of the data]

2. You will read your datafiles into python with the following commands: [add here] and then you will save as hdf5 with the following commands: [add here]


The end result is a sample dataset stored in hdf5 format consisting of 5 unordered arrays (lists). Each element in each array are described below.
- image (spectrogram) of the sonotype with size 244 * 244 * 3 ( 3 channel 224 * 224 pixels image) as float. 
- sonotype number as float
- time range: 2 floats for start and end time
- frequency range: 2 floats for minimum and maximum frequency
- taxonomic group as string (in this case, we only provide sonotypes of 6 birds so all the groups are "b"). This is not used in training/validation/testing but only for analysis about the model performance on different taxonomic group so it is not necessary to have it in order to run the example.

The element at the same index in each array belongs to the same sample and the five arrays should have the same length, i.e, the nth image in image array, nth time range, nth frequency, and nth taxomic group represent the same sample of nth sonotype number.

## Fitting the neural network model

Once you have your data in the hdf5 format as described above, you need to follow the instructions in the `sample_classification_with_augmentation.ipynb` jupyter notebook. The training speed might differ in the laptop processers but should probably take less than 2 hours.

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
keras==2.3.1
```

5. Run `JupyterLab` by typing `jupyter-lab` in the terminal. A browser page will be opened automatically where you will find the `sample_classification_with_augmentation.ipynb` jupyter notebook from the folder you just cloned.

6. Change the path in the 3rd code block (Read the dataset) to where you stored your hdf5 file and run all the code blocks.


Note that one alternative to use `JupyterLab` is to use Google Colab, which is a online version for jupyter notebook. Instructions could be found [here](https://colab.research.google.com/). When using Colab, please remember to use the GPU run time, which could be found at Runtime > Change runtime type and select GPU as Hardware accelerator (otherwise, the training process might be extremely slow). Google Colab limit the resources that we can use for free, including the GPU, but the limited resources is enough for this example.




