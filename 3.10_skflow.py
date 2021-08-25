"""使用skflow内置的LinearRegreesor、DNN以及Scikit-learn中的集成回归模型对“美国波士顿房价”数据进行回归预测"""
# 一次性导入sklearn中的多个模块。
from sklearn import datasets, metrics, preprocessing, model_selection
# 导入 skflow
import skflow
from sklearn.ensemble import RandomForestRegressor

# 使用datasets.load_boston读取美国波士顿房价数据。
boston = datasets.load_boston()

# 获取房屋数据特征以及对应房价。
X, y = boston.data, boston.target

# 分割数据，随机采样25%作为测试样本。
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=33)

# 对数据特征进行标准化处理。
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用skflow中的LinearRegressor
tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
tf_lr.fit(X_train, y_train)
tf_lr_y_predict = tf_lr.predict(X_test)

# 输出skflow中LinearRegressor模型的回归性能。
print('The mean absolute error of Tensorflow Linear Regressor on boston dataset is',
      metrics.mean_absolute_error(tf_lr_y_predict, y_test))
print('The mean squared error of Tensorflow Linear Regressor on boston dataset is',
      metrics.mean_squared_error(tf_lr_y_predict, y_test))
print('The R-squared value of Tensorflow Linear Regressor on boston dataset is',
      metrics.r2_score(tf_lr_y_predict, y_test))

# 使用skflow的DNNRegressor，并且注意其每个隐层特征数量的配置。
tf_dnn_regressor = skflow.TensorFlowDNNRegressor(hidden_units=[100, 40], steps=10000, learning_rate=0.01, batch_size=50)
tf_dnn_regressor.fit(X_train, y_train)
tf_dnn_regressor_y_predict = tf_dnn_regressor.predict(X_test)

# 输出skflow中DNNRegressor模型的回归性能。
print('The mean absolute error of Tensorflow DNN Regressor on boston dataset is',
      metrics.mean_absolute_error(tf_dnn_regressor_y_predict, y_test))
print('The mean squared error of Tensorflow DNN Regressor on boston dataset is',
      metrics.mean_squared_error(tf_dnn_regressor_y_predict, y_test))
print('The R-squared value of Tensorflow DNN Regressor on boston dataset is',
      metrics.r2_score(tf_dnn_regressor_y_predict, y_test))

# 使用Scikit-learn的RandomForestRegressor。
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 输出Scikit中RandomForestRegressor模型的回归性能。
print('The mean absolute error of Sklearn Random Forest Regressor on boston dataset is',
      metrics.mean_absolute_error(rfr_y_predict, y_test))
print('The mean squared error of Sklearn Random Forest Regressor on boston dataset is',
      metrics.mean_squared_error(rfr_y_predict, y_test))
print('The R-squared value of Sklearn Random Forest Regressor on boston dataset is',
      metrics.r2_score(rfr_y_predict, y_test))