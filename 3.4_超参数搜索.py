"""使用单线程对文本分类的朴素贝叶斯模型的超参数组合执行网格搜索"""
# 从sklearn.datasets中导入20类新闻文本抓取器。
from sklearn.datasets import fetch_20newsgroups
# 导入numpy，并且重命名为np。
import numpy as np
# 从sklearn.model_selection中导入train_test_split用来分割数据。
from sklearn.model_selection import train_test_split
# 导入支持向量机（分类）模型。
from sklearn.svm import SVC
# 导入TfidfVectorizer文本抽取器。
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入PipeLine
from sklearn.pipeline import Pipeline
# 从sklearn.grid_search中导入网格搜索模块GridSearchCV。
from sklearn.model_selection import GridSearchCV

# 使用新闻抓取器从互联网上下载所有数据，并且存储在变量news中。
news = fetch_20newsgroups(subset='all')
# 对前3000条新闻文本进行数据分割，25%文本用于未来测试。
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25,
                                                    random_state=33)
# 使用Pipeline简化系统搭建流程，将文本抽取和分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
# 这里需要试验的2个超参数的的个数分别是4、3，svc__gamma的参数共有10^-2, 10^-1...。这样我们一共有12种的超参数组合，12个不同参数下的模型。
parameters={'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
# 将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV。注意refit=True
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索。
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性
print(gs.score(X_test, y_test))
# 结论：使用单线程的网格搜索技术对朴素贝叶斯模型在文本分类任务中的超参数组合进行调优，共有12组超参数\times 3折交叉验证=36项独立运行的计算任务。
# 该任务一共进行了7分23秒，寻找到的最佳的超参数组合在测试集上所能达成的最高分类准确性为82.27%。


"""使用多线程对文本分类的朴素贝叶斯模型的超参数组合执行并行化的网格搜索"""
# 初始化配置并行网格搜索，n_jobs=-1代表使用该计算机全部的CPU。
gs_2 = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)
# 执行多线程并行网格搜索。
gs_2.fit(X_train, y_train)
gs_2.best_params_, gs.best_score_
# 输出最佳模型在测试集上的准确性。
print(gs_2.score(X_test, y_test))
# 结论：同样是网格搜索，使用多线程并行搜索技术对朴素贝叶斯模型在文本分类任务中的超参数组合进行调优，执行同样的36项计算任务一共只花费了1分13秒，
# 寻找到的最佳的超参数组合在测试集上所能达成的最高分类准确性依然为82.27%。
# 发现在没有影响验证准确性的前提下，通过并行搜索基础有效地利用了16核心（CPU）的计算资源，几乎6倍地提升了运算速度，节省了最佳超参数组合的搜索时间。
