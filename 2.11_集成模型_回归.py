"""使用三种集成回归模型对美国波士顿房价训练数据进行学习,并对测试数据进行预测"""
import numpy as np
# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.ensemble中导入RandomForestRegressor, ExtraTreesRegressor以及GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
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
# 使用RandomForestRegressor训练模型，并对测试数据做出预测
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train.ravel())
rfr_y_predict = rfr.predict(X_test)

# 使用ExtraTreesRegressor训练模型，并对测试数据做出预测
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train.ravel())
etr_y_predict = etr.predict(X_test)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train.ravel())
gbr_y_predict = gbr.predict(X_test)

'''评估性能'''
print('R-squared value of RandomForestRegressor is',
      rfr.score(X_test, y_test))
print('The mean squared error of RandomForestRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absolute error of RandomForestRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

print('R-squared value of ExtraTreesRegressor is',
      etr.score(X_test, y_test))
print('The mean squared error of ExtraTreesRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absolute error of ExtraTreesRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度
print(np.sort(list(zip(etr.feature_importances_, boston.feature_names)), axis=0))

print('R-squared value of GradientBoostingRegressor is',
      gbr.score(X_test, y_test))
print('The mean squared error of GradientBoostingRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The mean absolute error of GradientBoostingRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))