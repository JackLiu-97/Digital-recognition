from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)

#使用Keras快速搭建神经网络。
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#对数据进行了简单归一化
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 25

#将label转换为独热编码格式
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("before change:" ,test_labels[0])
print("after change: ", test_labels[0])
#将数据输入网络进行训练
network.fit(train_images, train_labels, epochs=10, batch_size = 128)