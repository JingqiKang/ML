import numpy as np
import pandas as pd
import tensorflow as tf
import skflow

train = pd.read_csv('../Dataset/MNIST/train.csv')
print(train.shape)
test = pd.read_csv('../Dataset/MNIST/test.csv')
print(test.shape)

# 将训练集中的数据特征与对应标记分离。
y_train = train['label']
X_train = train.drop('label', 1)

# 准备测试特征
X_test = test

# 使用skflow中已经封装好的基于tensorflow搭建的线性分类器TensorFlowLinearClassifier进行学习预测。
classifier = skflow.TensorFlowLinearClassifier(
    n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
classifier.fit(X_train, y_train)
linear_y_predict = classifier.predict(X_test)

linear_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': linear_y_predict})
linear_submission.to_csv('../Dataset/MNIST/linear_submission.csv', index=False)

# 使用skflow中已经封装好的基于tensorflow搭建的全连接深度神经网络TensorFlow DNNClassifier进行学习预测。
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[200, 50, 10], n_classes=10,
                                            steps=5000, learning_rate=0.01, batch_size=50)
classifier.fit(X_train, y_train)
dnn_y_predict = classifier.predict(X_test)

dnn_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': dnn_y_predict})
dnn_submission.to_csv('../Dataset/MNIST/dnn_submission.csv', index=False)


# 使用Tensorflow中的算子自行搭建更为复杂的卷积神经网络,并使用skflow的程序接口从事MNIST数据的学习与预测。
def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


classifier = skflow.TensorFlowEstimator(
    model_fn=conv_model, n_classes=10, batch_size=100, steps=20000,
    learning_rate=0.001)
classifier.fit(X_train, y_train)

# 由于Tensorflow会同时对所有测试样本进行矩阵运算，一次对28000个测试图片进行计算会消耗#大量的内存和计算资源。
# 这里所采取的是逐批次地对样本进行预测,最后拼接全部预测结果
conv_y_predict = []
for i in np.arange(100, 28001, 100):
    conv_y_predict = np.append(conv_y_predict, classifier.predict(X_test[i - 100:i]))
print(conv_y_predict[:10])

conv_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': np.int32(conv_y_predict)})
conv_submission.to_csv('../Dataset/MNIST/conv_submission.csv', index=False)
