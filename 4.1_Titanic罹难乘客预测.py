# 导入pandas方便数据读取和预处理
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
# 从sklearn.ensemble中导入RandomForestClassifier。
from sklearn.ensemble import RandomForestClassifier
# 从XGBoost导入XGBClassifier用于处理分类预测问题
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 分别从本地读取训练数据和测试数据
train = pd.read_csv("../Dataset/Titanic/train.csv")
test = pd.read_csv("../Dataset/Titanic/test.csv")
# 先分别输出训练与测试数据的基本信息
print(train.info())
print(test.info())

# 按照我们之前对Titanic事件的经验，人工选取对预测有效的特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']

# 通过我们之前对数据的总体观察，得知Embarked特征存在缺失值，需要补完
print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())

# 对于Embarked这种类别型的特征，我们使用出现频率最高的特征值来填充
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
# 对于Age这种数值类型的特征，我们使用求平均值或者中位数来填充缺失值
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# 重新对处理后的训练和测试数据进行查验，发现一切就绪
X_train.info()

# 使用DictVectorizer对特征向量化
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 使用默认配置初始化RandomForestClassifier
rfc = RandomForestClassifier()
# 也使用默认配置初始化XGBClassifier。
xgbc = XGBClassifier()

# 使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier以及XGBClassifier进行性能评估，并获得平均分类准确性的得分。
cross_val_score(rfc, X_train, y_train, cv=5).mean()
print('RandomForestClassifier:', cross_val_score(rfc, X_train, y_train, cv=5).mean())
cross_val_score(xgbc, X_train, y_train, cv=5).mean()
print('XGBClassifier:', cross_val_score(xgbc, X_train, y_train, cv=5).mean())

# 使用默认配置的RandomForestClassifier进行预测操作。
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
# 将默认配置的RandomForestClassifier对测试数据的预测结果存储在rfc_submission.csv文件中。
rfc_submission.to_csv('../Dataset/Titanic/rfc_submission.csv', index=False)

# 使用默认配置的XGBClassifier进行预测操作。
xgbc.fit(X_train, y_train)
print(xgbc.fit(X_train, y_train))
xgbc_y_predict = xgbc.predict(X_test)
# 将默认配置的XGBClassifier对测试数据的预测结果存储在xgbc_submission.csv文件中。
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('../Dataset/Titanic/xgbc_submission.csv', index=False)

# 使用并行网格搜索的方式寻找更好的超参数组合，以期待进一步提高XGBClassifier的预测性能。
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}

xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train, y_train)
print(gs.fit(X_train, y_train))

# 查验优化之后的XGBClassifier的超参数配置以及交叉验证的准确性。
print(gs.best_score_)
print(gs.best_params_)

# 使用经过优化超参数配置的XGBClassifier堆测试数据的预测结果存储在文件xgbc_best_submission中。
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('../Dataset/Titanic/xgbc_best_submission.csv', index=False)