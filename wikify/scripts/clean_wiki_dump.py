"""Parse wikipedia corpus, extract keywords and normalize article text
Usage:
  clean_wiki_dump.py -i <file> -o <file>
  clean_wiki_dump.py -h | --help
Options:
  -i <file>     XML dump to clean.
  -o <file>     Output cleaned xml.
  -h --help     Show this screen.
"""
from docopt import docopt
import re

from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from wikify.const import NAMESPACE, ARTICLE_ID, NAMESPACE_TAG,\
                         PAGE_TAG, TITLE_TAG, REDIRECT_TAG, TEXT_TAG


MATCH_KEYWORDS = re.compile('\[\[([^][]+)\]\]', re.IGNORECASE)
CLEAN_TEXT = re.compile('\{\{.*\}\}|\{\{.*\\n|.*\\n\}\}')
TOKENIZE_TEXT = RegexpTokenizer(r'(?u)\b\w\w+\b')
IGNORED_KEYWORDS = ('image:', 'file:', 'category:', 'wikipedia:')
STOPWORDS = stopwords.words('english')
NONWORDS = ['ref', 'http', 'https', 'lt', 'gt', 'quot', 'wbr', 'shy', 'www',\
            'com', 'url', 'ref', 'st', 'll']


def extract_node_text(elem, tag):
    node = elem.iterdescendants(tag=tag)
    return next(node).text


def extract_keywords(text):
    return MATCH_KEYWORDS.findall(text)


def clean_text(text):
    text = text.lower()  # Convert all words to lowercase
    text = CLEAN_TEXT.sub('', text)  # Delete text between '{{' '}}'
    # Only alphanumeric elements and delete undesirable words
    tokens = [token for token in TOKENIZE_TEXT.tokenize(text)
              if token not in NONWORDS]
    return ' '.join(tokens)


def extract_keyword_id(keyword):
    return keyword.split('|')[0]


def extract_keyword_name(keyword):
    return keyword.split('|')[-1].lower()


def clean_keyword_name(keyword_name):
    return ' '.join(TOKENIZE_TEXT.tokenize(keyword_name))


if __name__ == '__main__':
    opts = docopt(__doc__)

    # root_node = etree.Element("wikipedia")

    # Reader that iterates over each wikipedia page in the xml
    xml_reader = etree.iterparse(opts['-i'], tag=PAGE_TAG)

    with etree.xmlfile(opts['-o']) as xf:
        with xf.element('wikipedia'):

            for event, elem in xml_reader:
                # Extract page ID (page type), we only want to parse articles
                namespace_id = extract_node_text(elem, NAMESPACE_TAG)
                # Check if is a redirect article, we want to exclude this article
                # because doesn't have valuable information
                try:
                    _ = extract_node_text(elem, REDIRECT_TAG)
                    is_redirect = True
                except StopIteration:
                    article_title = extract_node_text(elem, TITLE_TAG)
                    is_redirect = False

                if namespace_id == ARTICLE_ID and not is_redirect:
                    # Text in the article
                    text = extract_node_text(elem, TEXT_TAG)
                    # Find keywords, they are between '[[' ']]'
                    keywords = set(extract_keywords(text))
                    # Now clean text
                    text = clean_text(text)

                    # Build xml nodes
                    # Article node
                    with xf.element('article', title=article_title):
                    # article_node = etree.Element('article', title=article_title)

                        # Text node
                        with xf.element('text'):
                        # text_node = etree.Element('text')
                        # text_node.text = text
                        # article_node.append(text_node)
                            xf.write(text)

                        # Keywords nodes
                        for keyword in keywords:
                            key_id = extract_keyword_id(keyword)
                            key_name = extract_keyword_name(keyword)

                            # Exclude stopword and some specific wikipedia keywords
                            if key_name not in STOPWORDS and\
                                not keyword.startswith(IGNORED_KEYWORDS):
                                # Clean keyword name
                                key_name = clean_keyword_name(key_name)
                                # Build node
                                keyword_node = etree.Element('keyword', id=key_id,
                                                             name=key_name)
                                xf.write(keyword_node)
                        #         article_node.append(keyword_node)

                        # root_node.append(article_node)

                # Clear xml node
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

    del xml_reader

    # # Write to xml file
    # with open(opts['-o'], 'wb') as f:
    #     f.write(etree.tostring(root_node))
