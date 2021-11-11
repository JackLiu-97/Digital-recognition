# [Digital-recognition](https://github.com/JackLiu-97/Digital-recognition)
基于机器学习方法的MNIST手写数字识别

1.本项目为分别使用SVM、决策树、神经网络的方法进行手写数字识别，并比较三种方法的准确率；

2.用到的编程语言为python>3.6；

3.代码位于Code文件夹中，数据集在Dataset文件夹中，部分结果图在res文件夹中；

4.所需的库：

```
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils import to_categorical
import pylab
import _pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from sklearn import svm
```

5.数据集介绍：
MNIST数据集是机器学习领域非常经典的一个数据集。MNIST数据集来自美国国家标准与技术研所，National Institute of Standard and Technology(NIST)。

MNIST是一个包含数字0-9的手写体图片数据集，图片已归一化为以手写数字为中心的28*28规格的图片。

训练集60,000个手写体图片及对应标签 ；测试集10,000个手写体图片及对应标签。

它包含四个部分：

- train-images-idx3-ubyte.gz：训练集图像（9912422字节）

- 训练集图像，共60000张图像

- train-labels-idx1-ubyte.gz：训练集标签（28881 字节）

训练标签集

- t10k-images-idx3-ubyte.gz：测试集图像（1648877字节）

测试集图像，共10000张图像

- t10k-labels-idx1-ubyte.gz：测试集标签（4542字节）

测试标签集
