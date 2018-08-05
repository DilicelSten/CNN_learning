# Char CNN Implementation
Implement a Character-level Convolutional Neural Network on the dataset called 20newsgroup.

Using the model from paper: **Character-level Convolutional Networks for Text Classification**


## Main codes
* config.py

Contain the **parameters** in model and training which is beneficial for **tuning**.

* data_process.py

**Extract data from csv file** ,**split the text into single char** which is represented by an **alphabet dict**, and finally **generate input vector**

* char_cnn_model.py

**CharCNN** model with all the process in it, which contains **six convolutional layers**, followed by **three fully connected layers**

* train_char_cnn.py

**Training** and **predicting** process

## Operating environment
Based on python3.4 and mian tools as follow:

* tensorflow

* numpy

* pandas

* sklearn


## Operation instructions
### first step
Download the **processed data** from :
https://github.com/DilicelSten/CNN_learning/tree/master/data/20newsgroup
### second step
run train_char_cnn.py

## Warning

When I set the learning rate as 0.001, the dev accuracy just decreased suddenly from 70% to zero, so I turn it to 0.005, which shows a better performance.


