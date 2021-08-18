"""美国波士顿地区房价预测"""
# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 导入numpy并重命名为np
import numpy as np
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.linear_model 导入 LinearRegression 和 SGDRegression
from sklearn.linear_model import LinearRegression, SGDRegressor
# 从sklearn.metrics依次导入r2_score, mean_squared_error以及mean_absolute_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取数据存储在变量boston中
boston = load_boston()
# 输出数据描述
print(boston.DESCR)
# 该数据共有506条美国波士顿地区房价的数据,每条数据包括对指定房屋的13项数值型特征描述和目标房价。
# 另外,该数据中没有缺失的属性/特征值(Missing Attribute Values),更加方便了后续的分析。

X = boston.data
y = boston.target
# 对原始数据进行分割，25%的乘客数据用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 分析回归目标值的差异
print("The max target value is", np.max(boston.target))
print("The min target value is", np.min(boston.target))
print("The average target value is", np.mean(boston.target))
# 我们发现预测目标房价之间的差异较大,因此需要对特征以及目标值进行标准化处理。

'''数据标准化'''
# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
# 分别对训练和测试数据的特征及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

'''使用线性回归模型LinearRegression 和 SGDRegressor分别对美国波士顿地区房价进行预测'''
# 使用默认配置初始化线性回归器
lr = LinearRegression()
# 使用训练数据进行参数估计
lr.fit(X_train, y_train)
# 对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)

sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)

'''性能评测'''
# 使用LinearRegression模型自带的评估模块，并输出评估结果
print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))
# 使用mean_squared_error模块，并输出评估结果
print('The value of mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
# 使用mean_absolute_error模块，并输出评估结果
print('The value of mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用SGDRegression模型自带的评估模块，并输出评估结果
print('The value of default measurement of SGDRegression is', sgdr.score(X_test, y_test))
# 使用mean_squared_error模块，并输出评估结果
print('The value of mean squared error of SGDRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
# 使用mean_absolute_error模块，并输出评估结果
print('The value of mean absolute error of SGDRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
