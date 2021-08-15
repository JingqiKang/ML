'''集成模型对泰坦尼克号乘客是否生还的预测'''
import pandas as pd
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 使用sklearn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report

# 读取数据
titanic = pd.read_csv('Dataset/Titanic/train.csv')
# 人工选取Pclass、Age以及Sex作为判别乘客是否能够生还的特征
X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']
# 对于缺失的年龄信息，我们使用全体乘客的平均年龄代替，这样可以在保证顺利训练模型的同时，尽可能不影响预测任务
X['Age'].fillna(X['Age'].mean(), inplace=True)
# 对原始数据进行分割，25%的乘客数据用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 对类别型特征进行转化，成为特征向量
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

'''性能评价'''
# 输出单一决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print("The accuracy of Decision Tree Classifier is", dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))
# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print("The accuracy of Random Forest Classifier is", rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))
# 输出梯度提升决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标
print("The accuracy of Gradient Tree Boosting is", gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))
# 在相同的训练和测试数据条件下,仅仅使用模型的默认配置，梯度上升决策树具有最佳的预测性能,其次是随机森林分类器,最后是单一决策树。
# 大量在其他数据上的模型实践也证明了上述结论的普适性。一般而言,工业界为了追求更加强劲的预测性能,经常使用随机森林分类模型作为基线系统(Baseline System)。
