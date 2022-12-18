import os
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

text_path = "texts/small/CULTUR/"

# Получение путей к файлам
def get_doc_names():
    dirs = os.listdir(text_path)
    for file in dirs:
        if file.find('.docx') == -1:
            dirs.remove(file)
    # ограничение количества файлов для проверки результатов
    doc_names = ['1.docx', '2.docx', '3.docx']
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

def tfidf_result(clear_texts):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(clear_texts)
    # print('\nWord indexes:')
    # print(tfidf.vocabulary_)
    # display tf-idf values
    print('\ntf-idf values:')
    print(vectors)

doc_names_list = get_doc_names()
clear_texts = []
for doc in doc_names_list:
    text = doc_to_string(doc)
    clear_doc_text = normalization(text)
    clear_texts.append(clear_doc_text)
tfidf_result(clear_texts)

