import pickle
from collections import Counter
from math import ceil
from multiprocessing import Pool
import re
import time
from heapq import nlargest
import sys

from lxml import etree
from nltk.tokenize import word_tokenize
import numpy as np

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID
from wikify.utils import fast_xml_iter, clean_keywords


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

    count_articles = 0
    pool = Pool(processes=4)
    extract_regex = re.compile('\[\[([^][]+)\]\]', re.IGNORECASE)
    presicion, recall, f1 = 0, 0, 0
    hits, gold_total, predicted_total = 0, 0, 0
    for event, elem in test_data:
        iterator = elem.iterchildren(tag=NAMESPACE_TAG)
        namespace_id = next(iterator).text

        if namespace_id == ARTICLE_ID:
            # Text in the article
            iterator = elem.iterdescendants(tag=TEXT_TAG)
            text = next(iterator).text

            # Find words inside '[[ ]]'
            keywords = extract_regex.findall(text)
            cleaned_keywords = pool.map(clean_keywords, keywords)

            gold_keywords = set(filter(None, cleaned_keywords))
            gold_keywords = gold_keywords.intersection(model._index_to_feature)

            ratio = len(gold_keywords) / len(text.split())

            # print(gold_keywords)
            if len(gold_keywords) > 0:
            # if 0.05 <= ratio <= 0.07:
                predicted_keywords = model.rank(text, ceil(ratio*len(text.split())))
                count_articles += 1

                hits += len(gold_keywords.intersection(predicted_keywords))
                gold_total += len(gold_keywords)
                predicted_total += len(predicted_keywords)

                presicion =  hits / gold_total
                recall = hits / predicted_total
                f1 = 2 * ((presicion * recall) / (presicion + recall))

                progress('{} articles processed - precision: {:2.2f}% - recall: {:2.2f}% - F1: {:2.2f}%'.format(
                          count_articles, presicion * 100, recall * 100, f1 * 100))

        # Clear data read
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context
