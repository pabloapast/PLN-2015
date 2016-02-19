from collections import Counter
from heapq import nlargest
from math import ceil
from multiprocessing import Pool
import re
import time

from lxml import etree
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID,\
                         PUNCTUATION
from wikify.utils import fast_xml_iter, clean_keywords


class Keyphraseness:

    def __init__(self, xml, processes=4, ratio=None):
        # Iterates over xml and extract keywords
        start_time = time.time()
        # Vocabulary of keywords with their occurrence count
        _vocabulary = self.extract_keywords(xml, processes)

        # Delete keywords that occurs less than 5 times
        for key, value in list(_vocabulary.items()):
            if value <= 5:
                del _vocabulary[key]
        print("--- %s sec: extract_keywords ---" % (time.time() - start_time))
        self._vocabulary_len = len(_vocabulary)
        print('vocabulary len = ', self._vocabulary_len)
        # print(_vocabulary.keys())


        start_time = time.time()
        self._vectorizer = CountVectorizer(ngram_range=(1, 3),
                                           vocabulary=list(_vocabulary.keys()))
        self._vectorizer.fit([])
        _keywords_counts = self.count_keywords(xml, processes)
        # print(self._vectorizer.get_feature_names())
        # print(_keywords_counts)
        self._index_to_feature = self._vectorizer.get_feature_names()
        print("--- %s sec: count_keywords ---" % (time.time() - start_time))

        # print(np.sum(_keywords_counts))
        # l = [n for i, n in enumerate(self._index_to_feature)
        #      if _keywords_counts[0, i] == 0]
        # print(l)
        # print(self._index_to_feature[24])
        self._keyphraseness = np.zeros((1, self._vocabulary_len))
        for index, keyword in enumerate(self._index_to_feature):
            self._keyphraseness[0, index] = _vocabulary[keyword] /\
                                            _keywords_counts[0, index]


    def extract_keywords(self, xml, processes):
        pool = Pool(processes=processes)
        # Regular expression used to extract keywords inside '[[ ]]'
        extract_regex = re.compile('\[\[([^][]+)\]\]', re.IGNORECASE)

        vocabulary = Counter()
        context = etree.iterparse(xml, tag=PAGE_TAG)
        for event, elem in context:
            iterator = elem.iterchildren(tag=NAMESPACE_TAG)
            namespace_id = next(iterator).text

            if namespace_id == ARTICLE_ID:
                # Text in the article
                iterator = elem.iterdescendants(tag=TEXT_TAG)
                text = next(iterator).text

                # Find words inside '[[ ]]'
                keywords = extract_regex.findall(text)
                cleaned_keywords = pool.map(clean_keywords, keywords)

                # Update dictionary only with nonempty keywords
                vocabulary.update(filter(None, cleaned_keywords))

            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context

        return vocabulary


    def count_keywords(self, xml, processes):
        pool = Pool(processes=processes)
        keywords_counts = np.zeros((1, self._vocabulary_len), dtype=np.int)
        # count = 0
        context = etree.iterparse(xml, tag=PAGE_TAG)
        articles = []
        for i, (event, elem) in enumerate(context):
            iterator = elem.iterchildren(tag=NAMESPACE_TAG)
            namespace_id = next(iterator).text

            if namespace_id == ARTICLE_ID:
                iterator = elem.iterdescendants(tag=TEXT_TAG)
                text = next(iterator).text
                articles.append([text])

                if len(articles) == 500:
                    m = pool.map(self._vectorizer.transform, articles)
                    for count in m:
                        keywords_counts += count
                    articles = []

            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context

        m = pool.map(self._vectorizer.transform, articles)
        for count in m:
            keywords_counts += count

        return keywords_counts


    def rank(self, text, ratio=None):
        if ratio is None:
            ratio = ceil(0.06 * len(text.split()))

        m = self._vectorizer.transform([text])
        rows, cols = m.nonzero()
        zipped = list(zip(cols, self._keyphraseness[0, cols]))
        top = nlargest(ratio, zipped, key=lambda t: t[1])

        return [self._index_to_feature[index] for index, prob in top]
