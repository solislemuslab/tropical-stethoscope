# Tropical stethoscope
Analyzing sound data to estimate biodiversity, in collaboration witn Zuzana Burivalova

- Yuren Sun

# Steps to clone the repo locally

1. In the terminal, go to the folder where you want to put the local repository using cd

2. Type:
```shell
git clone https://github.com/solislemuslab/tropical-stethoscope.git
```
3. Inside this folder, create a subfolder called `data` where you will put the data. This folder is ignored in the `.gitignore` file, so nothing inside this folder will be pushed to 


# Data results from Yuren analyses

From Yuren:

`classification_random_6_Feb10.csv`
This is the csv file for all the classification for random 6 sonotypes. It is formated in the way like “sonotypes separated by “; “, sample sizes separated by “; “, test loss, test accuracy” for each line
For example, in the first line
```
90.0; 95.0; 219.0; 49.0; 498.0; 295.0, 40; 33; 24; 14; 12; 4, 0.02718656708796819, 1.0
```
90.0; 95.0; 219.0; 49.0; 498.0; 295.0 are the sonotypes that I use
40; 33; 24; 14; 12; 4 are the sample sizes
0.02718656708796819 is the test loss (that I did not use for the plot)
1.0 is the accuracy
The four part in each line is separated by “,”


Here is the result for classification among random 15 sonotypes. It is formated in the same way as above: `classification_15_feb_20.csv`

I need to replace the `;` for `,`:
```shell
sed -i'' -e 's/;/,/g' classification_random_6_Feb10.csv
sed -i'' -e 's/;/,/g' classification_15_feb_20.csv
```

Code to make plots in `plots.Rmd`.
