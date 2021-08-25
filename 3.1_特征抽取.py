"""DictVectorizer 对使用字典存储的数据进行特征抽取与向量化"""
# 从sklearn.feature_extraction导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# 从sklearn.datasets中导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text导入CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 从sklearn.naive_bayes中导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
# 从sklearn.metrics中导入classification_report
from sklearn.metrics import classification_report

# 定义一组字典列表，用来表示多个数据样本（每个字典代表一个数据样本）
measurements = [{'city': "Dubai", 'temperature': 33.},
                {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]
# 初始化DictVectorizer特征抽取器
vec = DictVectorizer()
# 输出转化之后的特征矩阵
print(vec.fit_transform(measurements).toarray())
# 输出各个维度的特征含义
print(vec.get_feature_names())

"""使用CountVectorizer并且不去掉停用词的条件下,对文本特征进行量化的朴素贝叶斯分类性能测试"""
# 从互联网上即时下载新闻样本，subset='all'参数代表下载全部近2万条文本
news = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
count_vec = CountVectorizer()
# 只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)
# 使用默认的配置对分类器进行初始化
mnb_count = MultinomialNB()
# 使用朴素贝叶斯分类器对CountVectorizer（不去除停用词）后的训练样本进行参数学习
mnb_count.fit(X_count_train, y_train)
# 输出模型准确性结果
print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer without filtering stopwords):', mnb_count.score(X_count_test, y_test))
# 将分类预测的结果保存在变量y_count_predict中
y_count_predict = mnb_count.predict(X_count_test)
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_count_predict, target_names=news.target_names))

"""使用TfidfVectorizer并且不去掉停用词的条件下,对文本特征进行量化的朴素贝叶斯分类性能测试"""
tfidf_vec = TfidfVectorizer()
# 使用tfidf的方式将原始训练和测试文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)
# 使用默认的配置对分类器进行初始化
mnb_tfidf = MultinomialNB()
# 使用朴素贝叶斯分类器对TfidfVectorizer（不去除停用词）后的训练样本进行参数学习
mnb_tfidf.fit(X_count_train, y_train)
# 输出模型准确性结果
print('The accuracy of classifying 20newsgroups using Naive Bayes (TfidfVectorizer without filtering stopwords):',
      mnb_tfidf.score(X_tfidf_test, y_test))
# 将分类预测的结果保存在变量y_tfidf_predict中
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))

"""分别使用CountVectorizer 与TfidfVectorizer,并且去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试"""
# 分别使用停用词过滤配置初始化CountVectorizer 与TfidfVectorizer
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer='word', stop_words='english'), \
                                     TfidfVectorizer(analyzer='word', stop_words='english')
# 使用带有停用词过滤的CountVectorizer将原始训练和测试文本转化为特征向量
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
# 使用带有停用词过滤的TfidfVectorizer将原始训练和测试文本转化为特征向量
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

# 使用默认的配置对分类器进行初始化
mnb_count_filter = MultinomialNB()
# 使用朴素贝叶斯分类器对CountVectorizer（不去除停用词）后的训练样本进行参数学习
mnb_count_filter.fit(X_count_filter_train, y_train)
# 输出模型准确性结果
print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer by filtering stopwords):', mnb_count_filter.score(X_count_filter_test, y_test))
# 将分类预测的结果保存在变量y_count_filter_predict中
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))

# 使用默认的配置对分类器进行初始化
mnb_tfidf_filter = MultinomialNB()
# 使用朴素贝叶斯分类器对TfidfVectorizer（不去除停用词）后的训练样本进行参数学习
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
# 输出模型准确性结果
print('The accuracy of classifying 20newsgroups using Naive Bayes (TfidfVectorizer by filtering stopwords):', mnb_tfidf_filter.score(X_tfidf_filter_test, y_test))
# 将分类预测的结果保存在变量y_tfidf_filter_predict中
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))
