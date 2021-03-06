# 导入 pandas 与 numpy 工具包
import pandas as pd
import numpy as np
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing里导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.linear_model里导入LogisticRegression 与 SGDClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
# 从sklearn.metric里导入classification_report模块
from sklearn.metrics import classification_report

'''
原始数据共有699条样本，共11列，
第1列用于检索的id，9列与肿瘤有关的医学特征（1~10），最后1列表征肿瘤类型的数值（2表示良性，4表示恶性）
'''

'''良/恶性乳腺癌肿瘤数据预处理'''
# 创建特征列表
column_names=['sample code number','1','2','3','4','5','6','7','8','9','class']
# 使用pandas.read_csv函数从互联网读取指定数据
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)
# 将？替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how='any')
# 输出data的数据量和维度
print(data.shape) # 输出：(699, 11)

'''准备良/恶性乳腺癌肿瘤训练、测试数据'''
# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
# 查验训练样本的数量和类别分布
print(y_train.value_counts()) # 训练样本共有512条（344条良性，168条恶性）
# 查验测试样本的数量和类别分布
print(y_test.value_counts()) # 测试样本共有171条（100条良性，71条恶性）

'''使用线性分类模型从事良/恶性肿瘤预测任务'''
# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# 初始化LogisticRegression与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()
# 调用LogisticRegression中的fit函数/模块来训练模型参数
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果存储在变量lr_y_predict中
lr_y_predict = lr.predict(X_test)
# 调用SGDClassifier中的fit函数/模块来训练模型参数
sgdc.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果存储在变量lr_y_predict中
sgdc_y_predict = sgdc.predict(X_test)
# 预测结果称为准确性（Accuracy），作为评估分类器模型的一个重要性能指标

'''使用线性分类器模型从事良/恶性肿瘤预测任务的性能分析'''
# 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
# 利用classification_report模块获得LogisticRegression其他三个指标的结果
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
print("finish")
# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of SGD Classifier:', sgdc.score(X_test, y_test))
# 利用classification_report模块获得SGDClassifier其他三个指标的结果
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
print("finish")
# 我们可以发现:LogisticRegression比起SGDClassifier在测试集上表现有更高的准确性(Accuracy)。
# 这是因为Scikit-learn中采用解析的方式精确计算LogisticRegression的参数,而使用梯度法估计SGDClassifier的参数。
