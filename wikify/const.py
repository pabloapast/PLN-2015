# import re
# from string import punctuation

# from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


ARTICLE_TAG = 'article'
TEXT_TAG = 'text'
KEYWORD_TAG = 'keyword'

# English stopwords (i, me, my, myself, we, our ...)
STOPWORDS = set(stopwords.words('english'))

# Words that appear linked in wikipedia but are ignored
NONWORDS = set(['ref', 'http', 'https', 'lt', 'gt', 'quot', 'wbr', 'shy',
                'www', 'com', 'url', 'ref', 'st', 'll', ''])

# Ignored words, not include in vocabulary: STOPWORDS + NONWORDS
IGNORED_KEYWORDS = STOPWORDS.union(NONWORDS)

# Ignore keywords starting with this names
# IGNORED_KEYWORDS = ('image:', 'file:', 'category:', 'wikipedia:')

# PUNCTUATION = punctuation + "\'\'\"\""

# STOPWORDS = stopwords.words('english')

# CLEAN_REGEX = RegexpTokenizer(r'(?u)\b\w\w+\b')

# MATCH_KEYWORDS = re.compile('\[\[([^][]+)\]\]', re.IGNORECASE)
