"""Eval perfomance of disambiguation method
Usage:
  eval_disambiguation.py -i <file> -d <file> -t <file> [-n <n>]
  eval_disambiguation.py -h | --help
Options:
  -i <file>     Trained model to evaluate
  -d <file>     XML dump to test prediction
  -t <file>     Wikipedia titles
  -n <n>        Number of articles to analyze [default: float('inf')]
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from lxml import etree

from wikify.const import ARTICLE_TAG
from wikify.utils import article_keywords, clear_xml_node


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # XML file
    xml = opts['-d']

    n = eval(opts['-n'])

    # Load wikipedia article titles
    with open('wiki-dump/mini/enwiki-test1gb-clean-titles', 'rb') as f:
        titles = pickle.load(f)

    # Load trained model
    with open(opts['-i'], 'rb') as f:
        model = pickle.load(f)

    hits, total = 0, 0
    for i, (_, article) in enumerate(etree.iterparse(xml, tag=ARTICLE_TAG)):
        keywords = article_keywords(article)
        keywords = [keyword for keyword in keywords
                    if keyword.attrib['name'] in model.vocabulary and
                    keyword.attrib['id'] in titles]

        # Gold data, keywords with correct sense
        gold_key_id = model.keyword_ids(keywords)

        # Predicted data
        context_list = model.keyword_context(keywords)
        predicted_key_id = [model.disambiguate(k, s) for k, s in context_list]

        hits += sum([g.lower() == p.lower() for g, p in zip(gold_key_id,
                                                            predicted_key_id)])
        total += len(gold_key_id)
        presicion = hits / total

        progress('{} articles processed - precision: {:2.2f}%'
                 .format(i, presicion * 100))

        if i >= n:
            break
        clear_xml_node(article)

    print('\n')
