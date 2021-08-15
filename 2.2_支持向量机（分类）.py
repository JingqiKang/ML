# 从sklearn.datasets里导入手写体数字加载器
from sklearn.datasets import load_digits
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.svm里导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC
# 从sklearn.metric里导入classification_report模块
from sklearn.metrics import classification_report


'''示例：手写数字识别分类'''
# 从通过数据加载器获得手写体数字的数码图像数据并存储在digits变量中
digits = load_digits()
# 检视数据规模和特征维度
print(digits.data.shape)
# 随机选取75%的数据作为训练样本，其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
# 分别检视训练与测试数据规模
print(y_train.shape)
print(y_test.shape)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# 初始化线性假设的支持向量机分类器 LinearSVC
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(X_train, y_train)
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果存储在变量y_predict中
y_predict = lsvc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评
print('The Accuracy of Linear SVC is', lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
# 通过测试结果可以知道,支持向量机(分类)模型的确能够提供比较高的手写体数字识别性能。平均而言,各项指标都在95%上下。
