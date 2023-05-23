from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import os
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics

train = '20news-bydate-train'
data = {}
word_cnt = {}
doc_cnt = -1
data_1 = []
line_to_name = {}
max_f = 500

for dirname in os.listdir(train):
    for file_name in os.listdir(os.path.join(train, dirname)):
        file_path = os.path.join(os.path.join(train, dirname), file_name)
        try:
            f = open(file_path, 'r', encoding='utf-8', errors='ignore')
        except:
            print(f'read {file_path} error!')
        doc_cnt += 1
        dic = {}
        appeared = []
        s = ''
        for line in f:
            s = s + re.sub('[^a-zA-Z ]', '', line) + ' '
        l = s.split(' ')
        s = ''
        for word in l:
            if word == '':
                continue
            if word not in dic.keys():
                dic[word] = 1
            else:
                dic[word] += 1
            if word not in appeared:
                appeared.append(word)
            if s == '':
                s = word
            else:
                s = s + ' ' + word
        for word in appeared:
            if word not in word_cnt.keys():
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1
        data[os.path.join(dirname, file_name)] = dic
        data_1.append(s)
        line_to_name[doc_cnt] = os.path.join(dirname, file_name)

print("word_cnt")
ls = list(word_cnt.items()) # 将键值对以元组的形式存放，最后保存在一个列表中

    # ls = [('张三', 23), ('李四', 18), ('王五', 20), ('刘六', 25)]
ls.sort(key=lambda x:x[1],reverse=True)
my_vocabulary = []
for i in range(200, 200 + max_f):
    print(ls[i])
    my_vocabulary.append(ls[i][0])

vectorizer = CountVectorizer(max_features=max_f, vocabulary=my_vocabulary) #列数为5000
# 该类会统计每个词语的tf-idf权值
tf_idf_transformer = TfidfTransformer()
# 将文本转为词频矩阵并计算tf-idf
X = tf_idf_transformer.fit_transform(vectorizer.fit_transform(data_1))
# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

# 建立文本聚类模型
k = 20
model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=1, verbose=0)

model.fit(X)

X_embedded = TSNE(n_components=2).fit_transform(X)


print(X_embedded)
print(model.labels_)

print(X_embedded[:, 0])
cb = ['#F0F8FF',
'#FAEBD7',
'#00FFFF',
'#7FFFD4',
'#F0FFFF',
'#F5F5DC',
'#FFE4C4',
'#000000',
'#FFEBCD',
'#0000FF',
'#8A2BE2',
'#A52A2A',
'#DEB887',
'#5F9EA0',
'#7FFF00',
'#D2691E',
'#FF7F50',
'#6495ED',
'#FFF8DC',
'#DC143C',
'#00FFFF',
'#00008B'
]
c = []
for i in model.labels_:
    c.append(cb[i])

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color=c)
plt.show()

pgjg1 = metrics.silhouette_score(X, model.labels_, metric='euclidean')   #轮廓系数
print('聚类结果的轮廓系数=', pgjg1)
