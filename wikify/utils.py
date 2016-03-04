from nltk.util import ngrams

from wikify.const import IGNORED_KEYWORDS, MATCH_KEYWORDS, CLEAN_TEXT,\
                         TOKENIZE_TEXT


# ---- Utils to extract data from xml nodes ----
def article_keywords(article):
    keywords = article.getchildren()[1:]

    return keywords


def article_text(article):
    text_node = article.getchildren()[0]

    return text_node.text


def clear_xml_node(xml_node):
    xml_node.clear()
    while xml_node.getprevious() is not None:
        del xml_node.getparent()[0]


def extract_node_text(xml_node, tag):
    node = xml_node.iterdescendants(tag=tag)
    return next(node).text


# ---- Utils to clean data extracted ----
def extract_keywords(text):
    return set(MATCH_KEYWORDS.findall(text))


def clean_text(text):
    text = text.lower()  # Convert all words to lowercase
    return ' '.join(TOKENIZE_TEXT.tokenize(text))


def clean_surround(text):
    text = text.lower()  # Convert all words to lowercase
    text = CLEAN_TEXT.sub('', text)  # Delete text between '{{' '}}'
    # Only alphanumeric elements and delete stop words
    tokens = [token for token in TOKENIZE_TEXT.tokenize(text)
              if not token.isdigit() and token not in IGNORED_KEYWORDS]
    return ' '.join(tokens)


def extract_surround_words(keyword, text):
    surround_words = text.split(keyword)
    l_words = clean_surround(surround_words[0][-150:])
    r_words = clean_surround(surround_words[1][:150])
    return l_words, r_words


def extract_keyword_id(keyword):
    return keyword.split('|')[0]


def extract_keyword_name(keyword):
    return keyword.split('|')[-1]


def clean_keyword_name(keyword_name):
    return ' '.join(TOKENIZE_TEXT.tokenize(keyword_name.lower()))


def extract_n_grams(sent, ngram_range):
    clean_to_original = dict()

    for n in range(1, ngram_range + 1):
        ngrams_list = ngrams(sent, n)

        for ngram in ngrams_list:
            keyword = ' '.join(ngram)
            # Clean keyword
            cleaned_keyword = clean_keyword_name(keyword)
            # Save mappings between cleaned and original keywords
            clean_to_original[cleaned_keyword] = keyword

    return clean_to_original
