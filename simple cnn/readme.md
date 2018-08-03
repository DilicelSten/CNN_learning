# Single CNN Implementation
Implement a single Convolutional Neural Network on the dataset called 20newsgroup, which contains three models as follows:

* **CNN-rand**

* **CNN-static**

* **CNN-non-static**

## Data
* **20newsgroup**

You can download from here: http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz

* **GoogleNews-vectors-negative300.bin**

You can download from here: https://pan.baidu.com/s/1kU73HlX

## Main codes
* data_process.py

**Extract data from csv file** and provide **label_dict** for transforming text and **batch_iter** for generating batches

* data_2_vector.py

Using **pretrained vector** to represent our data in order to bulid the embedding layer.

* text_cnn.py

**TextCNN** model with all the process in it, which contains **a convolutional layer**, **a maxpooling layer** and **a fully connected layer**, but it can just be used in **CNN-rand model**.

* simple_cnn_rand.py

**Training** and **predicting** process about **CNN-rand**.

* text_cnn_static.py

Different from CNN-rand model, it contains **CNN-static and CNN-nonstatic models**, which can be used by changing a parameter called **static flag**.

* cnn_static_or_not.py

**Training** and **predicting** process about **CNN-static and CNN-nonstatic models**.

## Operating environment
Based on python3.4 and mian tools as follow:

* tensorflow

* numpy

* pandas

* sklearn


## Operation instructions
## CNN-rand
### first step
Download the **processed data** from :
https://github.com/DilicelSten/CNN_learning/tree/master/data/20newsgroup
### second step
run simple_cnn_rand.py

## CNN-static-or-not
### first step
run data_2_vector.py
### second step
run simple_cnn_rand.py

## Warning

When I run the code, something wrong happened. So there is something you need to pay attention to.

* computer memory
```
tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
```
When you see this warning, you need to split your data into batches for training or validation.

* OOM
```
Tensorflow Deep MNIST: Resource exhausted: OOM when allocating tensor with shape[10000,32,28,28]
```
When I run the static model, I first saw this warning which confused me a lot, and after debuging I found the embedding size should be consitent with the google vector dimension.
