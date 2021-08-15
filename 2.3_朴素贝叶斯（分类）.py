# 从sklearn.datasets里导入新闻数据抓取器 fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
# 使用sklearn.model_selection里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report


'''读取数据'''
# 与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
# 查验数据规模和细节
print(len(news.data))
print(news.data[0])
# 该数据共有18846条新闻;不同于前面的样例数据,这些文本数据既没有被设定特征,也没有数字化的量度。
# 因此,在交给朴素贝叶斯分类器学习之前,要对数据做进一步的处理。不过在此之前,我们仍然需要对数据进行分割并且随机采样出一部分用于测试。

'''数据分割'''
# 随机选取75%的数据作为训练样本，其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

'''使用朴素贝叶斯分类器对新闻文本数据进行类别预测'''
# 首先将文本转化为特征向量,然后利用朴素贝叶斯模型从训练数据中估计参数,最后利用这些概率参数对同样转化为特征向量的测试新闻样本进行类别预测。
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
# 使用默认设置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对测试样本进行类别预测，结果存储在变量y_predict中
y_predict = mnb.predict((X_test))

'''性能评估'''
print("The accuracy of Naive Bayes Classifier is", mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))
# 通过输出结果﹐我们获知朴素贝叶斯分类器对4712条新闻文本测试样本分类的准确性约为83.977%,平均精确率、召回率以及F1指标分别为0.86,0.84和0.82。

