from lxml import etree
# from nltk.tokenize import

from wikify.const import IGNORED_KEYWORDS, PUNCTUATION, PAGE_TAG,\
                         NAMESPACE_TAG, TEXT_TAG, ARTICLE_ID, CLEAN_REGEX


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
        cleaned_keyword = ' '.join(CLEAN_REGEX.tokenize(keyword))
        if len(cleaned_keyword.split()) > 3:
            cleaned_keyword = ''

    return cleaned_keyword
