from collections import Counter
from heapq import nlargest
from math import ceil

from lxml import etree
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from wikify.const import ARTICLE_TAG, IGNORED_KEYWORDS, KEYWORD_TAG
from wikify.utils import article_text, clear_xml_node


class Keyphraseness:

    def __init__(self, xml, n=3, ratio=0.04):
        self.n = n
        self.ratio = ratio

        # Keywords names with their occurrence count
        names_count = Counter()
        # Iterates over articles in xml and extract keywords
        for _, article in etree.iterparse(xml, tag=ARTICLE_TAG):
            names_count.update(self.extract_keywords(article))
            # Clear data read
            clear_xml_node(article)

        # Delete undesirable keywords or that occurs less than 5 times
        for key, value in list(names_count.items()):
            if value < 5 or key in IGNORED_KEYWORDS:
                del names_count[key]
        print('vocabulary len = ', len(names_count))

        # Used to count keywords occurrences in articles
        self._vectorizer = CountVectorizer(ngram_range=(1, self.n),
                                           vocabulary=list(names_count.keys()))

        # Iterates over articles in xml and count keywords
        keywords_counts = np.zeros((1, len(names_count)), dtype=np.int)
        for _, article in etree.iterparse(xml, tag=ARTICLE_TAG):
            keywords_counts += self.count_keywords(article)
            # Clear data read
            clear_xml_node(article)

        self.vocabulary = self._vectorizer.get_feature_names()

        self._keyphraseness = np.zeros((1, len(names_count)))
        for index, keyword in enumerate(self.vocabulary):
            self._keyphraseness[0, index] = names_count[keyword] /\
                                            keywords_counts[0, index]

    def extract_keywords(self, article):
        """Extract keywords from articles
        Only extracts n-grams between (1, n)

        article -- article contained in a xml node
        """
        keywords = article.iterchildren(tag=KEYWORD_TAG)

        return [keyword.attrib['name'] for keyword in keywords
                if len(keyword.attrib['name'].split()) <= self.n]

    def count_keywords(self, article):
        """Count keywords occurrences in text

        article -- article contained in a xml node
        """
        text = article_text(article)

        return self._vectorizer.transform([text])

    def rank(self, text, ratio=None):
        """Filter most important keywords from a text

        text -- text string
        ratio -- relation between number of keywords and number of words in
                 text, if it's None uses default number
        """
        if ratio is None:
            ratio = self.ratio

        count = self._vectorizer.transform([text])
        rows, cols = count.nonzero()
        keyword_with_prob = list(zip(cols, self._keyphraseness[0, cols]))

        # Extract index of m keywords with major probability
        m = ceil(ratio * len(text.split()))
        top_keywords = nlargest(m, keyword_with_prob, key=lambda t: t[1])

        return [self.vocabulary[index] for index, prob in top_keywords]
