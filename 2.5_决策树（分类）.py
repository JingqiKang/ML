# 导入pandas用于数据分析
import pandas as pd
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 使用sklearn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
# 从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report

'''读取数据'''
# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
titanic = pd.read_csv('Dataset/Titanic/train.csv')
# 观察前几行数据
print(titanic.head())
# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()查看数据的统计特性
print(titanic.info())
# 该数据共有1313条乘客信息﹐并且有些特征数据是完整的(如 pclass, name),有些则是缺失的;有些是数值类型的,有些则是字符串。

'''使用决策树模型预测泰坦尼克号乘客的生还情况'''
# 特征选择：决定分类的关键特征因素
X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']
# 查看当前选择的特征
print(X.info())

# 数据处理任务：1.填补缺失数据；2.转化数据特征
X['Age'].fillna(X['Age'].mean(), inplace=True)
# 查看补充完的数据
print(X.info())

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 特征抽取
vec = DictVectorizer(sparse=False)
# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)
# 同样对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

'''使用单一决策树训练数据集'''
# 使用默认设置初始化决策树分类器
dtc = DecisionTreeClassifier()
# 使用分割到的训练数据学习模型
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测
y_predict = dtc.predict(X_test)

'''性能评价'''
print("The accuracy of Decision Tree Classifier is", dtc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=['Died', 'Survived']))
