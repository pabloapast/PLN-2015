from collections import Counter
import re

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID, CLEAN_REGEX,\
                         STOPWORDS
from wikify.features import Context, current_keyword, NLeftWord, NRightWord,\
                            NTopWord
from wikify.utils import clean_keywords, clean_text

from lxml import etree
from nltk.stem.snowball import SnowballStemmer
from featureforge.vectorizer import Vectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


class Disambiguation:

    def __init__(self, xml, n=3, surround=6, top=10,
                 classifier='LogisticRegression'):
        self.n = n
        self.surround = surround
        self.top = top

        classifiers = {'LogisticRegression': LogisticRegression(),
                       'LinearSVC': LinearSVC(),
                       'MultinomialNB': MultinomialNB(),
                       'DecisionTreeClassifier': DecisionTreeClassifier(),
                       }

        features = []
        features.append(current_keyword)
        for i in range(self.surround):
            features.append(NLeftWord(i))
            features.append(NRightWord(i))
        for i in range(self.top):
            features.append(NTopWord(i))
        self.clf = clf = Pipeline([('vect', Vectorizer(features)),
                                   ('clf', classifiers[classifier])])
        clf = clf.fit(self.texts_context(xml), self.texts_ids(xml))


    def texts_context(self, xml):
        context_list = []
        xml_reader = etree.iterparse(xml, tag='article')

        for event, elem in xml_reader:
            text_iterator = elem.iterchildren(tag='text')
            text = next(text_iterator).text
            # assert text is not None

            count_words = Counter([word for word in text.split()
                                   if not word.isdigit() and word not in STOPWORDS])
            top_words, _ = zip(*count_words.most_common(self.top))
            top_words = list(top_words)

            keyword_iterator = elem.iterchildren(tag='keyword')

            context_list += self.text_context(keyword_iterator, top_words)

            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        del xml_reader

        return context_list


    def text_context(self, keyword_iterator, top_words):
        context_list = []

        for keyword in keyword_iterator:
            key_name = keyword.attrib['name']
            l_words = keyword.attrib['l_words'].split()[-self.surround:]
            l_words = ['']*(self.surround - len(l_words)) + l_words
            r_words = keyword.attrib['r_words'].split()[:self.surround]
            r_words = r_words + ['']*(self.surround - len(r_words))

            # print(Context(key_name, l_words, r_words, top_words))
            context_list.append(Context(key_name, l_words, r_words, top_words))

        return context_list


    def texts_ids(self, xml):
        key_id_list = []
        xml_reader = etree.iterparse(xml, tag='article')

        for event, elem in xml_reader:

            keyword_iterator = elem.iterchildren(tag='keyword')

            key_id_list += self.text_ids(keyword_iterator)

            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del xml_reader

        return key_id_list


    def text_ids(self, keyword_iterator):
        key_id_list = []

        for keyword in keyword_iterator:
            key_id = keyword.attrib['id']
            key_id_list.append(key_id)

        return key_id_list


    def desambiguate(self, c):
        return self.clf.predict([c])
