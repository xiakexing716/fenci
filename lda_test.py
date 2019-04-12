import numpy as np
import pandas as pd
# import lda
import re
import jieba.posseg as pseg
import codecs
from gensim import corpora,models,similarities
from connmongo import MongoConn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 连接mongo
conn = MongoConn()
res = conn.db['getnews'].find({},{'title':1,'content':1,'_id':0})
title = []
content = []
for i in res:
    if i.get('title'):
        title.append(i['title'].strip())
        content.append(re.sub('[a-zA-Z0-9]*','',','.join(i['content']).strip()))
print(len(title))
print(len(content))

stop_words = 'stop.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]

stop_flag = ['x','c','u','d','p','t','uj','m','f','r']

# 去停用词
def ridtext(text):
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

# 所有的词语
all_words = ','.join(content)

vocab = tuple(set(ridtext(all_words)))
print("res",vocab)
print("len(res)",len(vocab))

def getres(contents):
    result = []
    for cont in contents:
        cont2 = ridtext(cont)
        inres = []
        for i in vocab:
            inres.append(cont2.count(i))
        result.append(inres)
    return result

res2 = getres(content)
res3 = np.array(res2)
print("res3",res3)
print("res3:type",res3.shape)

# fit the model

# model = lda.LDA(n_topics=8,n_iter=500, random_state=1)
# model.fit(res3)

# topic_word = model.topic_word_

# n = 5
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
#     print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
#
# doc_topic = model.doc_topic_
# for n in range(10):
#     topic_most_pr = doc_topic[n].argmax()
#     print("doc: {} topic: {}\n{}...".format(n,topic_most_pr,title[n][:50]))

# TF-IDF
column_sum = [float(len(np.nonzero(res3[:,i])[0])) for i in range(res3.shape[1])]
column_sum = np.array(column_sum)
column_sum = res3.shape[0] / column_sum
idf = np.log(column_sum)
idf = np.diag(idf)

res4 = []
for doc_v in res3:
    if doc_v.sum() == 0:
        doc_v = doc_v/1
    else:
        doc_v = doc_v/(doc_v.sum())
    res4.append(doc_v)
tfidf = np.dot(res4,idf)
print("tfidf:",tfidf)

# KMEANS 聚类
kmeans = KMeans(n_clusters=10)
kmeans.fit(tfidf)

# 打印出各个族的中心点
print(kmeans.cluster_centers_)
for index, label in enumerate(kmeans.labels_,1):
    print("index:{},label:{}".format(index,label))

print("inertia:{}".format(kmeans.inertia_))


# 可视化
tsne = TSNE(n_components=2)
decomposition_data = tsne.fit_transform(tfidf)

x = []
y = []

for i in decomposition_data:
    x.append(i[0])
    y.append(i[1])

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
plt.scatter(x,y,c=kmeans.labels_, marker="x")
plt.xticks(())
plt.yticks(())
plt.show()
