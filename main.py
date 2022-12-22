import os
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('TkAgg')
#
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)
#
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import numpy as np
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt
from sklearn_som.som import SOM

text_path = "texts/rand/"
#doc_names = ['1000.docx', '1001.docx', '1002.docx', '1003.docx', '1004.docx', '2000.docx',
#             '3000.docx', '3001.docx', '3002.docx', '3003.docx', '3004.docx', ]
doc_names = []
MAX_TITLE_LEN = 70

# Получение путей к файлам
def get_doc_names():
    dirs = os.listdir(text_path)
    for file in dirs:
        if file.find('.docx') == -1:
            dirs.remove(file)
    # ограничение количества файлов для проверки результатов
    doc_names.append(file)
    return doc_names

def doc_to_string(doc_name):
    active_path = text_path + doc_name
    # чтение текста из документа
    doc = docx.Document(active_path)
    all_paras = doc.paragraphs
    text_from_doc = ''
    for paragraph in all_paras:
        # разделение абзацев пробелом для корректного результата лемматизации
        text_from_doc += paragraph.text + ' '
    return text_from_doc

def normalization(text_from_doc):
    # лемматизация
    spacy.prefer_gpu()
    nlp = spacy.load("ru_core_news_sm")
    lemmatization = nlp(text_from_doc)
    # список типов слов, которые необходимо удалить
    stop_types = ['PUNCT', 'SPACE', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'SCONJ', 'SYM']
    # создание чистого текста без стоп слов
    clear_text = ''
    for token in lemmatization:
        # token.is_alpha проверят является ли строка буквенной, потому что числа могут расцениваться как прилагательные
        if token.pos_ not in stop_types and (token.is_stop is not True) and token.is_alpha:
            clear_text += token.lemma_ + ' '
    return clear_text

def result(clear_texts):
    vectorizer = TfidfVectorizer(ngram_range=(3, 4), min_df=0.01, max_df=0.95, sublinear_tf=True, norm='l2')
    vectors = vectorizer.fit_transform(clear_texts)
    Y = TSNE(n_components=2, random_state=0, perplexity=15, init='pca', learning_rate=200).fit_transform(np.array(vectors.toarray()))
    vec_x = Y[:, 0]
    vec_y = Y[:, 1]
    # plt.scatter(vec_x, vec_y, marker='o')
    # plt.show()

    X_principal = pd.DataFrame(Y)
    X_principal.columns = ['P1', 'P2']
    plt.figure(figsize=(8, 8))

    plt.title('Дендрограмма алгоритма кластеризации Agglomerative Clustering')
    Dendrogram = shc.dendrogram(shc.linkage(X_principal, method='ward'))


    ac = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')

    plt.figure(figsize=(6, 6))
    plt.scatter(X_principal['P1'], X_principal['P2'], c=ac.fit_predict(X_principal), cmap='rainbow', marker='o')
    plt.title('Agglomerative Clustering\nКол-во кластеров: {}'.format(ac.n_clusters))
    plt.show()


    X_principal = pd.DataFrame(Y)
    X_principal.columns = ['P1', 'P2']
    Y_som = SOM(m=2, n=2, dim=3)
    # m - The shape along dimension 0 (vertical) of the SOM
    # n - The shape along dimesnion 1 (horizontal) of the SOM
    # dim - The dimensionality (number of features) of the input space
    # Ir - The initial step size for updating the SOM weights.

    Y_som.fit(Y)
    predictions = Y_som.predict(Y)
    plt.scatter(X_principal['P1'], X_principal['P2'], c=predictions, cmap='rainbow', marker='o')
    plt.title('Self-organizing maps\nКол-во кластеров:{}'.format(Y_som.m))
    plt.show()

doc_names_list = get_doc_names()
clear_texts = []
for doc in doc_names_list:
    text = doc_to_string(doc)
    clear_doc_text = normalization(text)
    clear_texts.append(clear_doc_text)
    print(doc)
result(clear_texts)




