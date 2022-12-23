import os
import docx
import sklearn.preprocessing
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
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn import datasets


docs_path = "texts\small\\"

# Получение путей к файлам
def get_docs_info():
    doc_data = []
    for dirname, _, filenames in os.walk(docs_path):
        text_id = 0
        TSNE_ = 0.0
        for filename in filenames:
            doc_theme = os.path.join(dirname.replace(docs_path, ""))
            doc_name = filename
# DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE [DEBUG_TEXTS]
            if doc_theme == 'CULTUR' and (int(doc_name.replace('.docx', '')) < 30):
# DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE DELETE [DEBUG_TEXTS]
                doc_text = doc_to_string(doc_name, doc_theme)
                doc_data.append([doc_theme, doc_text, text_id, TSNE_])
                text_id += 1
    docs_df = pd.DataFrame(doc_data, columns=['theme', 'text', 'text_id', 'TSNE'])
    return docs_df

def doc_to_string(doc_name, doc_theme):
    active_path = docs_path + doc_theme + '\\' + doc_name
    # чтение текста из документа
    try:
        doc = docx.Document(active_path)
        all_paras = doc.paragraphs
        text_from_doc = ''
        for paragraph in all_paras:
            # разделение абзацев пробелом для корректного результата лемматизации
            text_from_doc += paragraph.text + ' '
    except:
        print(f"[DEBUG EXCEPTION] {doc_theme}//{doc_name}")
        return ''
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

def tfidf(clear_texts):
    vectorizer = TfidfVectorizer(ngram_range=(3, 4), sublinear_tf=True, norm='l2')
    # min_df=0.01, max_df=0.95
    vectors = vectorizer.fit_transform(clear_texts)
    x_embedded = TSNE(n_components=2, random_state=0, perplexity=15, init='pca', learning_rate=200).\
        fit_transform(np.array(vectors.toarray()))
    #vec_x = x_embedded[:, 0]
    #vec_y = x_embedded[:, 1]
    x_principal = pd.DataFrame(x_embedded)
    x_principal.columns = ['P1', 'P2']
    return vectors, x_embedded, x_principal

def dendrogramm(x_principal):
    x_principal.columns = ['P1', 'P2']
    plt.figure(figsize=(8, 8))
    plt.title('Дендрограмма алгоритма кластеризации Agglomerative Clustering')
    Dendrogram = shc.dendrogram(shc.linkage(x_principal, method='ward'))

def agglomerative(x_principal):
    ac = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
    plt.figure(figsize=(6, 6))
    plt.scatter(x_principal['P1'], x_principal['P2'], c=ac.fit_predict(x_principal), cmap='rainbow', marker='o')
    plt.title('Agglomerative Clustering\nКол-во кластеров: {}'.format(ac.n_clusters))
    plt.show()

def maps(x_embedded):

    '''
    m - The shape along dimension 0 (vertical) of the SOM
    n - The shape along dimesnion 1 (horizontal) of the SOM
    dim - The dimensionality (number of features) of the input space
    Ir - The initial step size for updating the SOM weights.
    '''

    x_principal = pd.DataFrame(x_embedded)
    x_principal.columns = ['P1', 'P2']
    x_som = SOM(m=3, n=1, dim=2)
    x_som.fit(x_embedded)
    predictions = x_som.predict(x_embedded)
    plt.scatter(x_principal['P1'], x_principal['P2'], c=predictions,
                cmap='rainbow', marker='.')
    plt.title('Self-organizing maps\nКол-во кластеров:{}'.format(x_som.m))
    plt.show()

def kmeans(vectors,x_embedded):
    km = KMeans(
        n_clusters=5, init='random',
        n_init=10, max_iter=500,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(vectors)
    plt.scatter(
        x_embedded[y_km == 0, 0], x_embedded[y_km == 0, 1],
        s=50, c='yellow',
        marker='o', edgecolor='black',
        label='Кластер 1'
    )
    plt.scatter(
        x_embedded[y_km == 1, 0], x_embedded[y_km == 1, 1],
        s=50, c='red',
        marker='o', edgecolor='black',
        label='Кластер 2'
    )
    plt.scatter(
        x_embedded[y_km == 2, 0], x_embedded[y_km == 2, 1],
        s=50, c='blue',
        marker='o', edgecolor='black',
        label='Кластер 3'
    )
    plt.scatter(
        x_embedded[y_km == 3, 0], x_embedded[y_km == 3, 1],
        s=50, c='green',
        marker='o', edgecolor='black',
        label='Кластер 4'
    )
    # plt.scatter(
    #     x_embedded[y_km == 4, 0], x_embedded[y_km == 4, 1],
    #     s=50, c='pink',
    #     marker='o', edgecolor='black',
    #     label='Кластер 5'
    # )
    plt.legend(scatterpoints=1)
    plt.title('Алгоритм: K-Means\nКол-во кластеров: {}\nКол-итераций:{}'.format(km.n_clusters, km.max_iter))
    plt.grid()
    plt.show()

def result_out(clear_text):
    with open('clear_text.txt', 'w+', encoding='utf-8') as o_file:
        for doc in clear_text:
           o_file.write(doc + '\n')

def result_get():
    clear_texts = []
    with open('clear_text.txt', 'r+', encoding='utf-8') as o_file:
        for doc in o_file:
            clear_texts.append(doc)
    return clear_texts

def new_result_out(docs_df):
    docs_df.to_pickle('culture_small_25.pkl')

def new_result_get():
    return pd.read_pickle('culture_small_25.pkl')


if __name__ == '__main__':
    # Не используется при загрузке данных из файла
    '''
    docs_df = get_docs_info()
    for text in docs_df['text']:
        clear_text = normalization(text)
        docs_df['text'] = docs_df['text'].replace(text, clear_text)
    '''
    # Загрузка данных из файла
    docs_df = new_result_get()

    # код для будущего понимания, не используется
    '''
    # create a object of term-frequency, inverse document frequency
    tfidf = TfidfVectorizer(min_df=5, lowercase=True, max_features=21, ngram_range=(1, 2), sublinear_tf=True)
    # transform each lebel into vector
    feature = tfidf.fit_transform(docs_df.text).toarray()
    labels = docs_df.text_id
    '''
    # код для будущего понимания, не используется. Готовый датасет от sklearn
    '''
    data = datasets.load_digits()
    print(f'{data.data}\n############\n{data.target}')
    https://learn.saylor.org/mod/book/view.php?id=55626&chapterid=41483
    '''

    # Старое обращение к функциям
    '''
    vectors, x_embedded, x_principal = tfidf(clear_texts)
    dendrogramm(x_principal)
    agglomerative(x_principal)
    maps(x_embedded)
    kmeans(vectors, x_embedded)
    '''
