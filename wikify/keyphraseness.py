from collections import Counter
import re
import time
from heapq import nlargest

from lxml import etree
from nltk.tokenize import word_tokenize
import numpy as np

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID
from wikify.utils import fast_xml_iter, clean_text


class Keyphraseness:

    def __init__(self, wiki_dump):
        # Ignore keywords starting with this names
        self._ignored_keywords = ['image:', 'file:', 'category:', 'wikipedia:']
        # Regular expression used to extract keywords inside '[[ ]]'
        self._extract_regex = re.compile('\[\[([^][]+)\]\]', re.IGNORECASE)

        # Iterates over xml and extract keywords
        start_time = time.time()
        xml_iterator = etree.iterparse(wiki_dump, tag=NAMESPACE_TAG)
        # Vocabulary of keywords with their occurrence count
        _vocabulary = Counter()
        fast_xml_iter(xml_iterator, self.extract_keywords, _vocabulary)

        # Delete keywords that occurs less than 5 times
        for key, value in list(_vocabulary.items()):
            if value <= 5:
                del _vocabulary[key]
        print("--- %s sec: extract_keywords ---" % (time.time() - start_time))

        start_time = time.time()
        _token_counts = np.zeros((1, len(_vocabulary.keys())))
        self._vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 3),
                                           vocabulary=_vocabulary.keys(),
                                           tokenizer=word_tokenize)
        fast_xml_iter(xml_iterator, self.count_keywords, _token_counts)
        self._index_to_feature = vectorizer.get_feature_names()
        print("--- %s sec: count_keywords ---" % (time.time() - start_time))

        start_time = time.time()
        self._keyphraseness = np.zeros((1, len(vocabulary.keys())))
        for index, keyword in enumerate(self._index_to_feature):
            self._keyphraseness[0, index] = _vocabulary[keyword] /\
                                            _token_counts[0, index]
        print("--- %s sec: keyphraseness ---" % (time.time() - start_time))


    def extract_keywords(self, elem, dest):
        iterator = elem.iterchildren(tag=NAMESPACE_TAG)
        namespace_id = next(iterator).text

        if namespace_id == ARTICLE_ID:
            keywords = []
            # Text in the article
            iterator = elem.iterdescendants(tag=TEXT_TAG)
            text = next(iterator).text
            # Find words inside '[[ ]]'
            words = self._extract_regex.findall(text)

            for word in words:
                word = clean_text(word.split('|')[-1])
                if not any(x in word for x in self._ignored_keywords) and\
                   len(word) > 0:
                    keywords.append(word)

            dest.update(keywords)


    def count_keywords(self, elem, dest):
        iterator = elem.iterchildren(tag=NAMESPACE_TAG)
        namespace_id = next(iterator).text

        if namespace_id == ARTICLE_ID:

        # Text in the article
        iterator = elem.iterdescendants(tag=TEXT_TAG)
        text = next(iterator).text
        m = self._vectorizer.transform(text)
        dest = np.add(dest, m)


    def rank(self, text, ratio):
        # ratio = int(len(text.split(' ')) * 0.06)
        m = self._vectorizer.transform(text)
        rows, cols = m.nonzero()
        zipped = list(zip(cols, self._keyphraseness[0, cols]))
        top = nlargest(ratio, zipped, key=lambda t: t[1])
        return [self._index_to_feature[i] for i in top]
