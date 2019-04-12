# -*- coding:UTF-8 -*-

import jieba.posseg as pseg
import codecs
from gensim import corpora,models,similarities

stop_words = 'stop.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]

stop_flag = ['x','c','u','d','p','t','uj','m','f','r']

# 对停用词
def tokenization(filename):
    result = []
    with open(filename,'r') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

filenames = ["巨头对新锐.txt","阅文集团.txt"]
corpus = []
for each in filenames:
    corpus.append(tokenization(each))
print("corpus:",len(corpus))
print("corpus[0]:",corpus[0])

# 建立词袋模型
dictionary = corpora.Dictionary(corpus)
print(dictionary)

doc_vectors = [dictionary.doc2bow(text) for text in corpus]
print(len(doc_vectors))
print(doc_vectors)

# 建立TF-IDF模型
tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]
print(len(tfidf_vectors))
print(len(tfidf_vectors[0]))

query = tokenization('阅文集团2.txt')
query_bow = dictionary.doc2bow(query)
print(len(query_bow))
print(query_bow)


index = similarities.MatrixSimilarity(tfidf_vectors)
sims = index[query_bow]
print("sims",list(enumerate(sims)))


# 构建lsi模型，设置主题数为2
lsi = models.LsiModel(tfidf_vectors,id2word = dictionary, num_topics = 2)
lsi.print_topics(2)

lsi_vector = lsi[tfidf_vectors]
for vec in lsi_vector:
    print(vec)

query = tokenization('阅文集团2.txt')
query_bow = dictionary.doc2bow(query)
print("query_bow",query_bow)

query_lsi = lsi[query_bow]
print(query_lsi)

index = similarities.MatrixSimilarity(lsi_vector)
sims = index[query_lsi]
print(list(enumerate(sims)))
