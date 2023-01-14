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
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt
from sklearn_som.som import SOM
from sklearn.cluster import KMeans
from yellowbrick.text import TSNEVisualizer

docs_path = "texts\small\\"

# Получение путей к файлам
def get_docs_info():
    doc_data = []
    text_metadata = [0, 0.0]
    for dirname, _, filenames in os.walk(docs_path):
        for filename in filenames:
            doc_theme = os.path.join(dirname.replace(docs_path, ""))
            doc_name = filename
            extension = os.path.splitext(filename)[1]
            if extension != '.docx':
                continue
            doc_text = doc_to_string(doc_name, doc_theme)
            doc_data.append([doc_theme, doc_text, text_metadata[0], text_metadata[1],
                             text_metadata[1], text_metadata[1], text_metadata[1], text_metadata[1]])
            text_metadata = [data+1 for data in text_metadata]
    docs_df = pd.DataFrame(doc_data, columns=['theme', 'text', 'text_id', 'text_x', 'text_y',
                                              'ac', 'som', 'km'])
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

def tfidf(docs_df):
    # sublinear_tf = replace tf with 1 + log(tf)
    tfidf = TfidfVectorizer(min_df=0.1, max_df=0.9, lowercase=True, sublinear_tf=True,)
    tfidf_vectors = tfidf.fit_transform(docs_df.text)
    # tfidf_vectors.to_numpy()
    # np.savetxt('tfidf.csv', tfidf_vectors, delimiter=",")
    tfidf_array = tfidf_vectors.toarray()
    vectors = TSNE(n_components=2, random_state=0, perplexity=15, init='pca').fit_transform(tfidf_array)
    # np.savetxt('tsne.csv', vectors, delimiter=",")
    counter = 0.0
    for temp in vectors:
        docs_df['text_x'] = docs_df['text_x'].replace(counter, temp[0])
        docs_df['text_y'] = docs_df['text_y'].replace(counter, temp[1])
        counter += 1
    # Create the visualizer and draw the vectors
    tsne = TSNEVisualizer()
    test = tsne.fit(tfidf_array)
    tsne.poof()
    return tfidf_vectors

def agglomerative(docs_df):
    text_xy = docs_df[['text_x', 'text_y']].copy()
    # дендрограмма
    plt.figure(figsize=(8, 8))
    z = shc.linkage(text_xy, 'ward')
    plt.title('Дендрограмма алгоритма кластеризации Agglomerative Clustering')
    Dendrogram = shc.dendrogram(z, labels=docs_df[['theme']].to_numpy())
    plt.show()
    # визуализация
    ac = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
    ac_array = ac.fit_predict(text_xy)
    plt.figure(figsize=(6, 6))
    plt.scatter(text_xy['text_x'], text_xy['text_y'], c=ac_array, cmap='rainbow', marker='o')
    plt.title('Agglomerative Clustering\nКол-во кластеров: {}'.format(ac.n_clusters))
    plt.show()
    # сохранение номера кластера в датафрейме
    counter = 0.0
    for cluster in ac_array:
        docs_df['ac'] = docs_df['ac'].replace(counter, cluster)
        counter += 1

def maps(docs_df):
    xy_df = docs_df[['text_x', 'text_y']].copy()
    xy_array = xy_df.to_numpy()
    Y_som = SOM(dim=2)
    Y_som.fit(xy_array)
    predictions = Y_som.predict(xy_array)
    plt.scatter(xy_df['text_x'], xy_df['text_y'], c=predictions, cmap='rainbow', marker='o')
    # plt.title('Self-organizing maps\nКол-во кластеров: {}'.format(Y_som.m))
    plt.title('Self-organizing maps')
    plt.show()

    counter = 0.0
    for cluster in predictions:
        docs_df['som'] = docs_df['som'].replace(counter, cluster)
        counter += 1

    '''
    m - The shape along dimension 0 (vertical) of the SOM
    n - The shape along dimesnion 1 (horizontal) of the SOM
    dim - The dimensionality (number of features) of the input space
    Ir - The initial step size for updating the SOM weights.
    '''

def kmeans(docs_df, vectors):
    km = KMeans(
        n_clusters=3, init='random',
        n_init=10, max_iter=500,
        tol=1e-04, random_state=0
    )

    y_km = km.fit_predict(vectors)

    xy_df = docs_df[['text_x', 'text_y']].copy()
    y = xy_df.to_numpy()

    plt.scatter(
        y[y_km == 0, 0], y[y_km == 0, 1],
        s=50, c='yellow',
        marker='o', edgecolor='black',
        label='Кластер 1'
    )
    plt.scatter(
        y[y_km == 1, 0], y[y_km == 1, 1],
        s=50, c='red',
        marker='o', edgecolor='black',
        label='Кластер 2'
    )
    plt.scatter(
        y[y_km == 2, 0], y[y_km == 2, 1],
        s=50, c='blue',
        marker='o', edgecolor='black',
        label='Кластер 3'
    )
    # plt.scatter(
    #     y[y_km == 3, 0], y[y_km == 3, 1],
    #     s=50, c='green',
    #     marker='o', edgecolor='black',
    #     label='Кластер 4'
    # )
    # plt.scatter(
    #     y[y_km == 4, 0], y[y_km == 4, 1],
    #     s=50, c='pink',
    #     marker='o', edgecolor='black',
    #     label='Кластер 5'
    # )
    # plt.scatter(
    #     y[y_km == 5, 0], y[y_km == 5, 1],
    #     s=50, c='orange',
    #     marker='o', edgecolor='black',
    #     label='Кластер 6'
    # )
    plt.legend(scatterpoints=1)
    plt.title('Алгоритм: K-Means\nКол-во кластеров: {}\nКол-итераций:{}'.format(km.n_clusters, km.max_iter))
    plt.grid()
    plt.show()

    counter = 0.0
    for cluster in y_km:
        docs_df['km'] = docs_df['km'].replace(counter, cluster)
        counter += 1

def result_out(docs_df, filename):
    docs_df.to_csv(filename)

def result_get(filename):
    return pd.read_csv(filename)


if __name__ == '__main__':
    docs_df = get_docs_info()
    counter = 1
    # Не используется при загрузке данных из файла
    for text in docs_df['text']:
        print(f'text {counter}')
        counter += 1
        clear_text = normalization(text)
        docs_df['text'] = docs_df['text'].replace(text, clear_text)
    # сохранение результата без кластеризации
    result_out(docs_df, '600_texts.csv')
    #
    #docs_df = result_get('600_texts.csv')
    #vectors = np.fromfile('tfidf.csv')
    vectors = tfidf(docs_df)
    agglomerative(docs_df)
    kmeans(docs_df, vectors)
    maps(docs_df)
    # сохранение результатов с кластеризацией
    result_out(docs_df, '600_full.csv')
