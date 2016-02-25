from collections import Counter
from heapq import nlargest
from math import ceil
from multiprocessing import Pool
import re
import time

from lxml import etree
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID,\
                         PUNCTUATION
from wikify.utils import fast_xml_iter, clean_keywords


NONWORDS = ['ref', 'http', 'https', 'lt', 'gt', 'quot', 'wbr', 'shy', 'www',\
            'com', 'url', 'ref', 'st', 'll']
STOPWORDS = stopwords.words('english') + NONWORDS + ['']


class Keyphraseness:

    def __init__(self, xml, n=3, processes=4, ratio=0.1):
        self.n = n
        self.ratio = ratio

        start_time = time.time()
        # Vocabulary of keywords with their occurrence count
        _vocabulary = Counter()
        # Iterates over articles in xml and extract keywords
        for _, elem in etree.iterparse(xml, tag='article'):
            keywords = self.extract_keywords(elem)
            _vocabulary.update(keywords)
            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        # Delete keywords that occurs less than 5 times
        for key, value in list(_vocabulary.items()):
            if value < 5:
                del _vocabulary[key]
        for w in STOPWORDS:
            if w in _vocabulary:
                del _vocabulary[w]

        print("--- %s sec: extract_keywords ---" % (time.time() - start_time))
        self._vocabulary_len = len(_vocabulary)
        print('vocabulary len = ', self._vocabulary_len)

        start_time = time.time()
        self._vectorizer = CountVectorizer(ngram_range=(1, self.n),
                                           vocabulary=list(_vocabulary.keys()))
        # self._vectorizer.fit([])
        _keywords_counts = np.zeros((1, self._vocabulary_len), dtype=np.int)
        # Iterates over articles in xml and count keywords
        for _, elem in etree.iterparse(xml, tag='article'):
            _keywords_counts += self.count_keywords(elem)
            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        # _keywords_counts = self.count_keywords(xml, processes)
        # print(self._vectorizer.get_feature_names())
        # print(_keywords_counts)
        self._index_to_feature = self._vectorizer.get_feature_names()
        print("--- %s sec: count_keywords ---" % (time.time() - start_time))

        # print(np.sum(_keywords_counts))
        l = [n for i, n in enumerate(self._index_to_feature)
             if _keywords_counts[0, i] == 0]
        print(l)
        # print(self._index_to_feature[24])
        self._keyphraseness = np.zeros((1, self._vocabulary_len))
        for index, keyword in enumerate(self._index_to_feature):
            self._keyphraseness[0, index] = _vocabulary[keyword] /\
                                            _keywords_counts[0, index]


    def extract_keywords(self, article):
        keywords = article.iterchildren(tag='keyword')

        return [keyword.attrib['name'] for keyword in keywords
                if len(keyword.attrib['name'].split()) <= self.n]


    def count_keywords(self, article):
        text = article.iterchildren(tag='text')
        text = next(text).text

        return self._vectorizer.transform([text])


    def rank(self, text, ratio=None):
        if ratio is None:
            ratio = self.ratio

        total_words = ceil(ratio * len(text.split()))

        m = self._vectorizer.transform([text])
        rows, cols = m.nonzero()
        keyword_with_prob = list(zip(cols, self._keyphraseness[0, cols]))
        top_keywords = nlargest(total_words, keyword_with_prob,
                                key=lambda t: t[1])

        return [self._index_to_feature[index] for index, prob in top_keywords]
