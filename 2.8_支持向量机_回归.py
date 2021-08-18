"""使用三种不同核函数配置的支持向量机回归模型进行训练,并且分别对测试数据做出预测"""
# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从 sklearn.svm 中导入支持向量机（回归）模型
from sklearn.svm import SVR
# 从sklearn.metrics依次导入r2_score, mean_squared_error以及mean_absolute_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取数据存储在变量boston中
boston = load_boston()
# 该数据共有506条美国波士顿地区房价的数据,每条数据包括对指定房屋的13项数值型特征描述和目标房价。
X = boston.data
y = boston.target
# 对原始数据进行分割，25%的乘客数据用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

'''数据标准化'''
# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
# 分别对训练和测试数据的特征及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

'''使用三种不同核函数配置的支持向量机回归模型进行训练﹐并且分别对测试数据做出预测'''
# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
ploy_svr = SVR(kernel='poly')
ploy_svr.fit(X_train, y_train)
ploy_svr_y_predict = ploy_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

'''对三种核函数配置下的支持向量机回归模型在相同测试集上进行性能评估'''
print('The value of default measurement of Linear SVR is',
      linear_svr.score(X_test, y_test))
print('The value of mean squared error of Linear SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The value of mean absolute error of Linear SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print('The value of default measurement of Poly SVR is',
      ploy_svr.score(X_test, y_test))
print('The value of mean squared error of Poly SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(ploy_svr_y_predict)))
print('The value of mean absolute error of Poly SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(ploy_svr_y_predict)))

print('The value of default measurement of RBF SVR is',
      rbf_svr.score(X_test, y_test))
print('The value of mean squared error of RBF SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The value of mean absolute error of RBF SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
# 通过三组性能测评我们发现,不同配置下的模型在相同测试集上存在着非常大的性能差异。并且在使用了径向基(Radialbasis function)核函数对特征进行非线性映射之后,支持向量机展现了最佳的回归性能。
