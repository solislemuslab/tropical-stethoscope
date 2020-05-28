# Notes

## Links to Google Colab files

viewable jupyter notebook on google drive with UW-Madison account

Classification: https://colab.research.google.com/drive/1mUK81IBL4Dz1X76jQjlJbFwwMPGUxReP?usp=sharing

Dataset creation: https://colab.research.google.com/drive/1h7tLkUp5yOKkKi2C4Kb4VR_VSStb8Fx8?usp=sharing



## Histogram of data

Uploaded on May 27th.

Whole histogram:<img src="imgs/histogram_whole.png" alt="histogram_whole" style="zoom:50%; display: inline;" />

Top 50 data:<img src="imgs/histogram_top_50.png" alt="histogram_50" style="zoom:50%; display: inline;" />

Top 150 frequency: 142 120 113  92  72  69  57  45  43  40  35  33  32  32  30  28  28  27  26  26  26  25  24  24  24  24  24  23  22  22  21  21  19  18  17  17  16  16  16  16  16  16  15  15  15  15  15  15  14  14  14  14  14  14  13  13  13  12  12  12  12  12  11  10  10  10  10  10  10  10  10  10   9   9   9   9   9   9   9   9   8   8   8   8   8   7   7   7   7   7   7   7   7   7   6   6   6   6   6   6

## Data

Tested with sonotypes with highest k number of data

### Only birds

Sonotypes (order or decreasing number): 52, 138, 463, 86, 139, 220

| Types | number of data | Accuracy (%) |
| :---- | -------------- | ------------ |
| 2     | 120            | 95.83        |
| 3     | 92             | 92.86        |
| 4     | 57             | 60.87        |
| 5     | 45             | 43.48        |
| 6     | 43             | 26.92        |

2 types, 120 each

loss: 10.2101 - accuracy: 0.9444 - val_loss: 24.5293 - val_accuracy: 0.9583

3 types, 92 each

loss: 0.0228 - accuracy: 0.9879 - val_loss: 0.7307 - val_accuracy: 0.9286

4 types, 57 each

accuracy: 0.8683 - val_loss: 2.6672 - val_accuracy: 0.6087

5 types, 45 each

accuracy: 0.3713 - val_loss: 1.4765 - val_accuracy: 0.4348

### All groups

Sonotypes (order or decreasing number): 52, 138, 25, 463, 236

#### balanced classes with time and frequency:

| Types | number of data | Accuracy (%) |
| :---- | -------------- | ------------ |
| 2     | 120            | 95.83        |
| 3     | 113            | 44.12        |
| 4     | 92             | 37.84        |
| 5     | 72             | 13.89        |

2 types, 120 each

loss: 10.2101 - accuracy: 0.9444 - val_loss: 24.5293 - val_accuracy: 0.9583

3 types, 113 each

loss: 328.8753 - accuracy: 0.3574 - val_loss: 53.3833 - val_accuracy: 0.4412

4 types, 92 each

loss: 368.9658 - accuracy: 0.3958 - val_loss: 191.2737 - val_accuracy: 0.3784

5 types, 72 each

3s 9ms/step - loss: 461.9938 - accuracy: 0.3086 - val_loss: 326.3062 - val_accuracy: 0.1389



#### inbalanced classes with time and frequency:

| Types | number of data | Accuracy (%) |
| :---- | -------------- | ------------ |
| 2     | 120            | 81.48        |
| 3     | 113            | 34.21        |
| 4     | 92             | 19.15        |

2 types,

loss: 116.8172 - accuracy: 0.7447 - val_loss: 74.5958 - val_accuracy: 0.8148

3 types,

loss: 314.5453 - accuracy: 0.3650 - val_loss: 140.8449 - val_accuracy: 0.3421

4 types,

loss: 560.8641 - accuracy: 0.2405 - val_loss: 157.1247 - val_accuracy: 0.1915



#### balanced classes without times and frequency

2 types, 120 each: loss: 9.5630 - accuracy: 0.5463 - val_loss: 3.9478 - val_accuracy: 0.3750

not work



### MINST balanced

| Types | number of data | Accuracy (%) |
| :---- | -------------- | ------------ |
| 2     | 120            | 81.48        |
| 3     | 113            | 100          |
| 4     | 92             | 100          |
| 5     | 72             | 95           |

2 types, 120 each

loss: 0.0494 - accuracy: 0.9907 - val_loss: 0.0021 - val_accuracy: 1.0000

3 types, 113 each

loss: 0.0284 - accuracy: 0.9967 - val_loss: 0.0239 - val_accuracy: 1.0000

4 =types, 92 each

loss: 0.0589 - accuracy: 0.9849 - val_loss: 0.0161 - val_accuracy: 1.0000

5 types, 72 each

loss: 0.0930 - accuracy: 0.9723 - val_loss: 0.0757 - val_accuracy: 0.9500