from collections import Counter
from heapq import nlargest
from multiprocessing import Pool
import re
import time

from lxml import etree
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID,\
                         PUNCTUATION
from wikify.utils import fast_xml_iter, clean_keywords


class Keyphraseness:

    def __init__(self, xml, processes=4):
        # Iterates over xml and extract keywords
        start_time = time.time()
        # Vocabulary of keywords with their occurrence count
        _vocabulary = self.extract_keywords(xml, processes)

        # Delete keywords that occurs less than 5 times
        for key, value in list(_vocabulary.items()):
            if value <= 5 or len(key.split()) > 3 or len(key) < 2:
                del _vocabulary[key]
        print("--- %s sec: extract_keywords ---" % (time.time() - start_time))
        self._vocabulary_len = len(_vocabulary)
        print('vocabulary len = ', self._vocabulary_len)
        # print(_vocabulary.keys())


        start_time = time.time()
        self._vectorizer = CountVectorizer(ngram_range=(1, 3),
                                           analyzer=lambda doc: self._vectorizer._word_ngrams(list(filter(lambda word: word not in PUNCTUATION, word_tokenize(' '.join(doc.lower().split('|'))))), None),
                                           vocabulary=list(_vocabulary.keys()))
        _keywords_counts = self.count_keywords(xml)
        # print(_keywords_counts)
        self._index_to_feature = self._vectorizer.get_feature_names()
        l = [(name, _keywords_counts[0, i])
             for i, name in enumerate(self._index_to_feature)
             if _keywords_counts[0, i] == 0]
        print(l)
        print("--- %s sec: count_keywords ---" % (time.time() - start_time))

        # print(np.sum(_keywords_counts))
        # print(_keywords_counts[])
        # print(self._index_to_feature[24])
        self._keyphraseness = np.zeros((1, self._vocabulary_len))
        for index, keyword in enumerate(self._index_to_feature):
            self._keyphraseness[0, index] = _vocabulary[keyword] /\
                                            _keywords_counts[0, index]


    def custom_analyzer(self):
        # preprocess = lambda doc: ' '.join(doc.lower().split('|'))
        # tokenize = word_tokenize
        return lambda x: x.split()
        return lambda doc: self._vectorizer._word_ngrams(filter(lambda word: word not in PUNCTUATION, word_tokenize(' '.join(doc.lower().split('|')))), None)
                # None)


    def extract_keywords(self, xml, processes=4):
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


    def count_keywords(self, xml):
        keywords_counts = np.zeros((1, self._vocabulary_len), dtype=np.int)
        # count = 0
        context = etree.iterparse(xml, tag=PAGE_TAG)
        for event, elem in context:
            iterator = elem.iterchildren(tag=NAMESPACE_TAG)
            namespace_id = next(iterator).text

            if namespace_id == ARTICLE_ID:
                # Text in the article
                iterator = elem.iterdescendants(tag=TEXT_TAG)
                text = next(iterator).text
                m = self._vectorizer.transform([text])
                # if 'Ottoman' in text:
                #     print(m[0, 298])
                #     print(text)
                #     break
                # count += 1
                # if count == 2:
                #     print(text)
                #     break
                keywords_counts += m

            # Clear data read
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context

        return keywords_counts


    def rank(self, text, ratio=None):
        # ratio = int(len(text.split(' ')) * 0.06)
        m = self._vectorizer.transform(text)
        rows, cols = m.nonzero()
        zipped = list(zip(cols, self._keyphraseness[0, cols]))
        top = nlargest(ratio, zipped, key=lambda t: t[1])
        return [self._index_to_feature[i] for i in top]

# 1148.8907270431519 sec: extract_keywords 3 cores
