from collections import Counter, defaultdict
from heapq import nlargest
from math import ceil
import pickle
import re

from lxml import etree
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID,\
                         PUNCTUATION
from wikify.utils import fast_xml_iter, clean_keywords


NONWORDS = ['ref', 'http', 'https', 'lt', 'gt', 'quot', 'wbr', 'shy', 'www',\
            'com', 'url', 'ref', 'st', 'll']
STOPWORDS = set(stopwords.words('english') + NONWORDS + [''])


class Keyphraseness:

    def __init__(self, xml, n=3, ratio=0.04):
        self.n = n
        self.ratio = ratio

        # keywords name with their occurrence count
        _names_count = Counter()
        # Iterates over articles in xml and extract keywords
        for _, article in etree.iterparse(xml, tag='article'):
            _names_count.update(self.extract_keywords(article))
            # Clear data read
            article.clear()
            while article.getprevious() is not None:
                del article.getparent()[0]

        # Delete undesirable keywords or that occurs less than 5 times
        for key, value in list(_names_count.items()):
            if value < 5 or key in STOPWORDS:
                del _names_count[key]

        self._names_len = len(_names_count)
        print('vocabulary len = ', self._names_len)


        self._vectorizer = CountVectorizer(ngram_range=(1, self.n),
                                           vocabulary=list(_names_count.keys()))
        _keywords_counts = np.zeros((1, self._names_len), dtype=np.int)
        # Iterates over articles in xml and count keywords
        for _, article in etree.iterparse(xml, tag='article'):
            _keywords_counts += self.count_keywords(article)
            # Clear data read
            article.clear()
            while article.getprevious() is not None:
                del article.getparent()[0]

        self._vocabulary = self._vectorizer.get_feature_names()

        self._keyphraseness = np.zeros((1, self._names_len))
        for index, keyword in enumerate(self._vocabulary):
            self._keyphraseness[0, index] = _names_count[keyword] /\
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

        return [self._vocabulary[index] for index, prob in top_keywords]
