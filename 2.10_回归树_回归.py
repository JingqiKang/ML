"""使用回归树对美国波士顿房价训练数据进行学习,并对测试数据进行预测"""
# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.tree中导入DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
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
# 使用默认配置初始化DecisionTreeRegressor
dtr = DecisionTreeRegressor()
# 用波士顿房价的训练数据构建回归树
dtr.fit(X_train, y_train)
# 使用默认配置的单一回归树对测试数据进行预测
dtr_y_predict = dtr.predict(X_test)

'''评估性能'''
print('R-squared value of DecisionTreeRegressor is',
      dtr.score(X_test, y_test))
print('The mean squared error of DecisionTreeRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The mean absolute error of DecisionTreeRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
# 该代码的输出结果优于线性回归器一节 LinearRegression 与SGDRegressor 的性能表现。
# 因此,可以初步判断“美国波士顿房价预测”问题的特征与目标值之间存在一定的非线性关系。
