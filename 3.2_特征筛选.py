"""使用Titanic数据集,通过特征筛选的方法一步步提升决策树的预测性能"""
# 导入pandas并且更名为pd
import pandas as pd
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import feature_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pylab as pl
from sklearn import feature_selection

# 从本地文件读取titanic数据。
titanic = pd.read_csv('../Dataset/Titanic/train.csv')
# 分离数据特征与预测目标
X = titanic.drop(['Name', 'Survived'], axis=1)
y = titanic['Survived']
# 对缺失数据进行填充
X['Age'].fillna(X['Age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)
# 分割数据，采样25%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 类别型特征向量化
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
# 输出处理后特征向量的维度
print(len(vec.feature_names_))

# 使用决策树模型依靠所有特征进行预测，并作性能评估
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
dt.score(X_test, y_test)
print(dt.score(X_test, y_test))

# 筛选前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)
print(dt.score(X_test_fs, y_test))

# 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print(results)

# 找到体现最佳性能的特征筛选的百分比
opt = np.where(results == results.max())[0]
print('Optimal number of features %d' % percentiles[opt[0]])
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)
print(dt.score(X_test_fs, y_test))
