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
