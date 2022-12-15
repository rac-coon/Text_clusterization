import os
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Получение путей к файлам
path = "texts/PYTTYR/"
dirs = os.listdir(path)
for file in dirs:
    if file.find('.docx') == -1:
        dirs.remove(file)

# ограничение количества файлов для проверки результатов
dirs = ['198.docx', '199.docx', '200.docx']

# Работа с каждым файлом
docs = []
for file in dirs:
    active_path = path + file

    # чтение текста из документа
    doc = docx.Document(active_path)
    all_paras = doc.paragraphs
    text_from_doc = ''
    for paragraph in all_paras:
        # разделение абзацев пробелом для корректного результата лемматизации
        text_from_doc += paragraph.text + ' '

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
    # добавление отформатированного текста в массив
    docs.append(clear_text)

# tf-tdf
tfidf = TfidfVectorizer()
# This is equivalent to fit followed by transform, but more efficiently
X = tfidf.fit_transform(docs)
print(X)

