"""对比随机决策森林以及XGBoost模型对泰坦尼克号上的乘客是否生还的预测能力"""
# 导入pandas用于数据分析。
import pandas as pd
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction导入DictVectorizer。
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

titanic = pd.read_csv('../Dataset/Titanic/train.csv')
# 选取Pclass、Age以及Sex作为训练特征。
X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']

# 对缺失的Age信息，采用平均值方法进行补全，即以Age列已知数据的平均数填充。
X['Age'].fillna(X['Age'].mean(), inplace=True)

# 对原数据进行分割，随机采样25%作为测试集。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)

# 对原数据进行特征向量化处理。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 采用默认配置的随机森林分类器对测试集进行预测。
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('The accuracy of Random Forest Classifier on testing set:', rfc.score(X_test, y_test))

# 采用默认配置的XGBoost模型对相同的测试集进行预测。
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print(xgbc.fit(X_train, y_train))
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test))