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
NONWORDS = ['ref', 'http', 'https', 'lt', 'gt', 'quot', 'wbr', 'shy', 'www',\
            'com', 'url', 'ref', 'st', 'll']
STOPWORDS = stopwords.words('english') + NONWORDS


def extract_node_text(elem, tag):
    node = elem.iterdescendants(tag=tag)
    return next(node).text


def extract_keywords(text):
    return set(MATCH_KEYWORDS.findall(text))


def clean_text(text):
    text = text.lower()  # Convert all words to lowercase
    text = CLEAN_TEXT.sub('', text)  # Delete text between '{{' '}}'
    # Only alphanumeric elements and delete undesirable words
    tokens = [token for token in TOKENIZE_TEXT.tokenize(text)
              if token not in NONWORDS]
    return ' '.join(tokens)


def clean_surround(text):
    text = text.lower()  # Convert all words to lowercase
    text = CLEAN_TEXT.sub('', text)  # Delete text between '{{' '}}'
    # Only alphanumeric elements and delete stop words
    tokens = [token for token in TOKENIZE_TEXT.tokenize(text)
              if not token.isdigit() and token not in STOPWORDS]
    return ' '.join(tokens)


def extract_surround_words(keyword, text):
    surround_words = text.split(keyword)
    l_words = clean_surround(surround_words[0][-150:])
    r_words = clean_surround(surround_words[1][:150])
    return l_words, r_words


def extract_keyword_id(keyword):
    return keyword.split('|')[0]


def extract_keyword_name(keyword):
    return keyword.split('|')[-1].lower()


def clean_keyword_name(keyword_name):
    return ' '.join(TOKENIZE_TEXT.tokenize(keyword_name))


if __name__ == '__main__':
    opts = docopt(__doc__)

    xml_header = b'<enwiki>\n'
    xml_tail = b'</enwiki>\n'

    out = open(opts['-o'], 'ab')

    out.write(xml_header)

    # Reader that iterates over each wikipedia page in the xml
    xml_reader = etree.iterparse(opts['-i'], tag=PAGE_TAG)

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
            if text is not None:
                # Find keywords, they are between '[[' ']]'
                keywords = [key for key in extract_keywords(text)
                            if not key.lower().startswith(IGNORED_KEYWORDS)]

                if len(keywords) > 0:
                    # Clean text
                    cleaned_text = clean_text(text)

                    # Build xml nodes
                    # Article node
                    article_node = etree.Element('article', title=article_title)

                    # Text node
                    text_node = etree.Element('text')
                    text_node.text = cleaned_text
                    article_node.append(text_node)

                    # Keywords nodes
                    for keyword in keywords:
                        key_id = extract_keyword_id(keyword)
                        key_name = extract_keyword_name(keyword)
                        # Clean keyword name
                        key_name = clean_keyword_name(key_name)
                        # Left and right words
                        l_words, r_words = extract_surround_words(keyword, text)
                        # Build node
                        keyword_node = etree.Element('keyword', id=key_id,
                                                     name=key_name,
                                                     l_words=l_words,
                                                     r_words=r_words)
                        article_node.append(keyword_node)

                    out.write(etree.tostring(article_node, pretty_print=True))

        # Clear xml node
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    del xml_reader

    out.write(xml_tail)
    out.close()

    # # Write to xml file
    # with open(opts['-o'], 'wb') as f:
    #     f.write(etree.tostring(root_node))
