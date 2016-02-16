from string import punctuation

from lxml import etree
from nltk.tokenize import word_tokenize


punctuation += "\'\'\"\""


# Iteration over xml with low ram consumption
def fast_xml_iter(context, func, dest):
    for event, elem in context:
        # Execute operations over xml node
        func(elem, dest)

        # Clear data read
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


# Tokenize keywords
def clean_text(keyword):
    keyword = keyword.split('|')[-1].lower()
    tokens = word_tokenize(keyword)
    return ' '.join([token for token in tokens if not token in punctuation])
