# Example with a sample dataset

Here we provide a jupyter notebook with a sample dataset containing data for 6 birds and some noises that are used for data augmentation.

The sample dataset could be found [here](https://drive.google.com/file/d/1kEW1FscJROzEVm3R8_1Yl4RyOw9K5_jL/view?usp=sharing) on Google Drive and is about 480mb in size.

Below are some instructions to get started with the codes. Once you are able to run the jupyter notebook, just follow the cells inside the notebook to tryout our methods. The training speed might differ in the laptop processers but should probably take less than 2 hours.

## Get Started

As the example is in a jupyter notebook, there are two options to get started with the code. Once you get started with the notebook, you just need to change the paths to the location on you own laptop/computer/google drive (marked in the jupyter notebook) and follow the order of cells to try out this example

1. Download the jupyterLab or use the online version of of it. Instructions could be found [here](https://jupyter.org/). After downloading the jupyterLab, you might need to download a list of necessary packages by typing the codes below in the terminal.

    ```  
    pip3 install -r requirements.txt
    ```

   with a list of packages in the requirements.txt as below
   
   ```
   numpy
   opencv
   h5py
   pandas
   tensorflow-gpu==1.14.0
   tensorflow==1.14.0
   keras==2.3.1
   ```

2. Use the Google Colab, which is a online version for jupyter notebook. Instructions could be found [here](https://colab.research.google.com/). When using Colab, please remember to use the GPU run time, which could be found at Runtime > Change runtime type and select GPU as Hardware accelerator (otherwise, the training process might be extremely slow). Google Colab limit the resources that we can use for free, including the GPU, but the limited resources is enough for this example.



## Dataset Format

The sample dataset is stored in hdf5 format and consist 5 unordered arrays (lists). Each element in each array are described below.
- image (spectrogram) of the sonotype with size 244 * 244 * 3 ( 3 channel 224 * 224 pixels image) as float. 
- sonotype number as float
- time range: 2 floats for start and end time
- frequency range: 2 floats for minimum and maximum frequency
- taxonomic group as string (in this case, we only provide sonotypes of 6 birds so all the groups are "b"). This is not used in training/validation/testing but only for analysis about the model performance on different taxonomic group so it is not necessary to have it in order to run the example

The element at the same index in each array belongs to the same sample and the five arrays should have the same length, i.e, the nth image in image array, nth time range, nth frequency, and nth taxomic group represent the same sample of nth sonotype number.

If you want to use your own dataset, it might be easier to store it in hdf5 format with arrays described above, but it is not necessary and you just need to have the data as the five arrays to try out this example.


