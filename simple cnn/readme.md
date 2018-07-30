# Single CNN Implementation
Implement a single Convolutional Neural Network on the dataset called 20newsgroup, which contains three models as follows:**(updating....)**

* **CNN-rand**

* **CNN-static**

* **CNN-non-static**


## Main codes
* data_process.py

**Extract data from csv file** and provide **label_dict** for transforming text and **batch_iter** for generating batches

* text_cnn.py

**TextCNN** model with all the process in it, which contains **a convolutional layer**, **a maxpooling layer** and **a fully connected layer**

* simple_cnn_rand.py

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
run simple_cnn_rand.py

## Warning

When I run the code, something wrong happened. So there is something you need to pay attention to.

* computer memory
```
tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
```
When you see this warning, you need to split your data into batches for training or validation.

