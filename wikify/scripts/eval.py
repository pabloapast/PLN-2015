import pickle
from collections import Counter
import re
import time
from heapq import nlargest

from lxml import etree
from nltk.tokenize import word_tokenize
import numpy as np

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID
from wikify.utils import fast_xml_iter, clean_text


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':

    # load the data
    test_data = etree.iterparse('./wiki-dump/mini/enwiki-test1gb', tag=PAGE_TAG)

    with open('./wiki-dump/mini/Keyphraseness', 'rb') as f:
        model = pickle.load(f)

    for event, elem in test_data:
        iterator = elem.iterchildren(tag=NAMESPACE_TAG)
        namespace_id = next(iterator).text

        if namespace_id == ARTICLE_ID:
            keywords = set()
            # Text in the article
            iterator = elem.iterdescendants(tag=TEXT_TAG)
            text = next(iterator).text
            # Find words inside '[[ ]]'
            words = self._extract_regex.findall(text)

            for word in words:
                word = clean_text(word)
                if not any(x in word for x in self._ignored_keywords) and\
                   len(word) > 0:
                    keywords.add(word)



