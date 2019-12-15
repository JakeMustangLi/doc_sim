import jieba
from gensim import corpora, models, similarities
import os
# 将分好词的文档转化为list
with open("doc/o1.txt", "r", encoding='UTF-8') as f:
    f1 = f.read()
doc1 = f1.split()

with open("doc/o2.txt", "r", encoding='UTF-8') as f:
    f2 = f.read()
doc2 = f2.split()

with open("doc/o3.txt", "r", encoding='UTF-8') as f:
    f3 = f.read()
doc3 = f3.split()

with open("doc/o4.txt", "r", encoding='UTF-8') as f:
    f4 = f.read()
doc4 = f4.split()

with open("doc/o5.txt", "r", encoding='UTF-8') as f:
    f5 = f.read()
doc5 = f5.split()

# TEST文本
with open("doc/ot.txt", "r", encoding='UTF-8') as f:
    ft = f.read()
docT = ft.split()

all_doc = [doc1, doc2, doc3, doc4, doc5]

print(all_doc)
# 根据文档建立词典 (词袋模型)
dictionary = corpora.Dictionary(all_doc)
print('词典：', dictionary)
feature_cnt = len(dictionary.token2id)
print('词典特征数：%d' % feature_cnt)
doc_vectors = [dictionary.doc2bow(doc) for doc in all_doc]
print(len(doc_vectors))
print(doc_vectors)

# TF-IDF
tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]
print(len(tfidf_vectors))
print(len(tfidf_vectors[0]))
test_bow = dictionary.doc2bow(docT)
print(len(test_bow))
print(test_bow)

index = similarities.MatrixSimilarity(tfidf_vectors)
sims = index[test_bow]
print(list(enumerate(sims)))

# LSI模型
lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=2)
lsi.print_topics(2)
lsi_vector = lsi[tfidf_vectors]
for vec in lsi_vector:
    print(vec)

test_lsi = lsi[test_bow]
print(test_lsi)

index = similarities.MatrixSimilarity(lsi_vector)
sims = index[test_lsi]
print(list(enumerate(sims)))