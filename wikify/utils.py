from lxml import etree
from nltk.tokenize import word_tokenize

from wikify.const import IGNORED_KEYWORDS, PUNCTUATION, PAGE_TAG,\
                         NAMESPACE_TAG, TEXT_TAG, ARTICLE_ID


# Iteration over xml with low ram consumption
def fast_xml_iter(xml, func, dest, tag=None):
    context = etree.iterparse(xml, tag=tag)

    for event, elem in context:
        # Execute operations over xml node
        func(elem, dest)

        # Clear data read
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


def clean_keywords(keyword):
    cleaned_keyword = ''
    keyword = keyword.split('|')[-1].lower()
    if not keyword.startswith(IGNORED_KEYWORDS):
        tokens = word_tokenize(keyword)
        cleaned_keyword = ' '.join([token for token in tokens
                                    if not token in PUNCTUATION])

    return cleaned_keyword
