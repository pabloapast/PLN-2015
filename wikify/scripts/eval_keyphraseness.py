"""Eval perfomance of Keyphraseness method
Usage:
  eval_keyphraseness.py -i <file> -d <file> [-n <n>]
  eval_keyphraseness.py -h | --help
Options:
  -i <file>     Trained model to evaluate
  -d <file>     XML to test prediction
  -n <n>        Number of articles to analyze [default: float('inf')]
  -h --help     Show this screen.
"""
from docopt import docopt
# from math import ceil
import pickle
# import re
import sys

from lxml import etree

from wikify.const import ARTICLE_TAG
from wikify.utils import article_text, clear_xml_node


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load trained model
    with open(opts['-i'], 'rb') as f:
        model = pickle.load(f)

    # XML file
    xml = opts['-d']

    n = eval(opts['-n'])

    vocabulary = set(model.vocabulary)

    count_articles, hits, gold_total, predicted_total = 0, 0, 0, 0
    for _, article in etree.iterparse(xml, tag=ARTICLE_TAG):
        # Article text
        text = article_text(article)
        # Extract keywords form article
        keywords = model.extract_keywords(article)
        # Keep only keywords contained in the vocabulary
        gold_keywords = set([keyword for keyword in keywords
                             if keyword in vocabulary])

        article_ratio = len(gold_keywords) / len(text.split())
        if model.ratio - 0.01 <= article_ratio <= model.ratio + 0.01:
            predicted_keywords = model.rank(text)

            hits += len(gold_keywords.intersection(predicted_keywords))
            gold_total += len(gold_keywords)
            predicted_total += len(predicted_keywords)

            precision = hits / gold_total
            recall = hits / predicted_total
            f1 = 2 * ((precision * recall) / (precision + recall))

            count_articles += 1
            progress('{} Articles - P: {:2.2f}% - R: {:2.2f}% - F1: {:2.2f}%'
                     .format(count_articles, precision * 100, recall * 100,
                             f1 * 100))

            if count_articles >= n:
                break

        # Clear data read
        clear_xml_node(article)

    print('\n')
