"""使用两种不同配置的K近邻回归模型对美国波士顿房价数据进行回归预测"""
# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors 导入 KNeighborRegression
from sklearn.neighbors import KNeighborsRegressor
# 从sklearn.metrics依次导入r2_score, mean_squared_error以及mean_absolute_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

'''准备数据'''
boston = load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

'''训练模型'''
# 初始化K近邻回归器，并且调整配置，使预测方式为平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)
# 初始化K近邻回归器，并且调整配置，使预测方式为距离加权回归
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

'''评估性能'''
print('R-squared value of uniform-weighted KNeighborRegression is',
      uni_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighborRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The mean absolute error of uniform-weighted KNeighborRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('R-squared value of distanced-weighted KNeighborRegression is',
      dis_knr.score(X_test, y_test))
print('The mean squared error of distanced-weighted KNeighborRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The mean absolute error of distanced-weighted KNeighborRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
# 输出表明:相比之下,采用加权平均的方式回归房价具有更好的预测性能。
