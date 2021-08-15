# 从sklearn.datasets导入iris数据加载器
from sklearn.datasets import load_iris
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier,即K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report

'''读取鸢尾(Iris)数据集'''
iris = load_iris()
# 查验数据规模
print(iris.data.shape)
# 查看数据说明
print(iris.DESCR)
# Iris数据集共有150朵鸢尾数据样本,并且均匀分布在3个不同的亚种;每个数据样本被4个不同的花瓣、花萼的形状特征所描述。

'''数据分割'''
# 随机选取75%的数据作为训练样本，其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

'''使用K近邻分类器对鸢尾花数据进行类别预测'''
# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
# 使用K近邻分类器对测试数据进行类别预测，预测结果存储在变量y_predict中
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

'''性能评估'''
print("The accuracy of K-Nearest Neighbor Classifier is", knc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=iris.target_names))
# K近邻分类器对38条鸢尾花测试样本分类的准确性约为71.053%,平均精确率、召回率以及F1指标分别为0.86,0.71和0.70。
