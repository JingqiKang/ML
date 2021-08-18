"""显示手写体数字图片经PCA压缩后的二维空间分布"""
# 导入pandas用于数据读取和处理
import pandas as pd
import numpy as np
# 从sklearn.decomposition导入PCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# 导入基于线性核的支持向量机分类器。
from sklearn.svm import LinearSVC
# 从sklearn.metrics导入classification_report用于更加细致的分类性能分析。
from sklearn.metrics import classification_report

# 使用pandas分别读取训练数据和测试数据集
digits_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
digits_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)
# 从训练与测试数据集上都分离出 64 维度的像素特征，和 1 维度的数字目标
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

# 初始化一个可以将高维度特征向量（64维）压缩到2个维度的PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)
# 显示10类手写体数字图谱经PCA压缩后的2维空间分布


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.values == i]
        py = X_pca[:, 1][y_digits.values == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principle Component')
    plt.ylabel('Second Principle Component')
    plt.show()


plot_pca_scatter()

"""使用原始像素特征和经PCA压缩重建的低维特征,在相同配置的支持向量机(分类)模型上分别进行图像识别"""
# 对训练数据、测试数据进行特征向量（图片像素）与分类目标的分隔。
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在y_predict中。
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)

# 使用PCA将原64维的图像数据压缩到20个维度。
estimator = PCA(n_components=20)

# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征。
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化（transform）。
pca_X_test = estimator.transform(X_test)

# 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_y_predict中。
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

# 对使用原始图像高维像素特征训练的支持向量机分类器的性能作出评估。
print(svc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))

# 对使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能作出评估。
print(pca_svc.score(pca_X_test, y_test))
print(classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))
