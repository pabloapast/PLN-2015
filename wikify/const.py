from string import punctuation
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


NAMESPACE = '{*}'  # Wildcard
ARTICLE_ID = '0'  # Id = 0 is assigned to wikipedia articles
NAMESPACE_TAG = NAMESPACE + 'ns'
PAGE_TAG = NAMESPACE + 'page'
TITLE_TAG = NAMESPACE + 'title'
REDIRECT_TAG = NAMESPACE + 'redirect'
TEXT_TAG = NAMESPACE + 'text'

# Ignore keywords starting with this names
IGNORED_KEYWORDS = ('image:', 'file:', 'category:', 'wikipedia:')

PUNCTUATION = punctuation + "\'\'\"\""

STOPWORDS = stopwords.words('english')

CLEAN_REGEX = RegexpTokenizer(r'(?u)\b\w\w+\b')
